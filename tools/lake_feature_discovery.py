#!/usr/bin/env python3
"""Inspect feature (column) schemas of parquet datasets in the S3 lake.

Discovers every dataset under s3://<bucket>/<prefix>/ -- anything extracted and
converted to parquet appears automatically, nothing is hardcoded -- then offers
an interactive menu to pick datasets and reports:

  1. full column list + types, per table, per dataset
  2. size analysis: rows, columns, dimensions, file size per table + totals
  3. table presence matrix across the chosen datasets
  4. column-level comparison for tables shared by 2+ datasets

Reads parquet footers only -- no data is downloaded.

Usage:
    eval "$(aws configure export-credentials --format env)"   # once per terminal
    python3 lake_features.py                        # menu, report to stdout
    python3 lake_features.py > features_report.txt  # menu still visible (stderr)
    python3 lake_features.py --non_interactive      # all datasets, no menu
    python3 lake_features.py --datasets cic-ids-2017,cic-ids-2018
"""

import argparse
import sys
from collections import defaultdict

import duckdb

DEFAULT_BUCKET = "ids-datalakec48eb2cab942494ba5059fac3b3527d9"
DEFAULT_PREFIX = "parquets"
DEFAULT_REGION = "eu-north-1"
EXT_REPO = "https://extensions.duckdb.org"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket of the lake")
    p.add_argument("--prefix", default=DEFAULT_PREFIX,
                   help="key prefix holding one sub-folder per parquet dataset")
    p.add_argument("--region", default=DEFAULT_REGION, help="bucket region")
    p.add_argument("--datasets", default=None,
                   help="comma-separated dataset names to inspect (skips the menu)")
    p.add_argument("--non_interactive", action="store_true",
                   help="no menu; inspect all discovered datasets")
    return p.parse_args()


def connect(region):
    con = duckdb.connect()
    con.execute(f"INSTALL httpfs FROM '{EXT_REPO}'; LOAD httpfs;")
    con.execute(f"INSTALL aws FROM '{EXT_REPO}'; LOAD aws;")
    con.execute(
        "CREATE OR REPLACE SECRET aws_lake "
        f"(TYPE s3, PROVIDER credential_chain, CHAIN 'env', REGION '{region}');"
    )
    return con


def discover(con, bucket, prefix):
    """Return {dataset: {table: s3_path}} for every parquet under the prefix.

    The first path segment under the prefix is the dataset name; anything
    deeper is part of the table name, so nested layouts still work.
    """
    root = f"s3://{bucket}/{prefix}/"
    files = [r[0] for r in con.execute(
        "SELECT file FROM glob(?)", [root + "**/*.parquet"]).fetchall()]
    catalog = defaultdict(dict)
    for path in sorted(files):
        rel = path[len(root):]
        dataset, _, table_path = rel.partition("/")
        if not table_path:  # parquet sitting directly under the prefix
            dataset, table_path = "(root)", rel
        catalog[dataset][table_path.removesuffix(".parquet")] = path
    return dict(catalog)


def choose_datasets(catalog, args):
    names = sorted(catalog)
    if args.datasets:
        chosen = [d.strip() for d in args.datasets.split(",") if d.strip()]
        unknown = [d for d in chosen if d not in catalog]
        if unknown:
            sys.exit(f"unknown dataset(s): {', '.join(unknown)}; "
                     f"available: {', '.join(names)}")
        return chosen
    if args.non_interactive:
        return names

    err = sys.stderr
    print(f"\nDatasets in s3://{args.bucket}/{args.prefix}/:", file=err)
    for i, ds in enumerate(names, 1):
        print(f"  [{i}] {ds:<28} ({len(catalog[ds])} tables)", file=err)
    print("Select datasets [Enter = all, e.g. 1,3]: ", end="", file=err, flush=True)
    reply = input().strip()
    if not reply:
        return names
    picked = []
    for tok in reply.replace(",", " ").split():
        if not tok.isdigit() or not (1 <= int(tok) <= len(names)):
            sys.exit(f"invalid selection: {tok!r}")
        picked.append(names[int(tok) - 1])
    return picked


def load_schemas(con, catalog, selected):
    """Return {dataset: {table: [(column, type), ...]}} preserving column order."""
    schemas = {}
    for ds in selected:
        tables = {}
        for table, path in sorted(catalog[ds].items()):
            rows = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
            tables[table] = [(r[0], r[1]) for r in rows]
        schemas[ds] = tables
    return schemas


def load_stats(con, catalog, selected):
    """Return {dataset: {table: {rows, row_groups, bytes}}} from parquet footers."""
    stats = {}
    for ds in selected:
        paths = [catalog[ds][table] for table in sorted(catalog[ds])]
        meta = con.execute(
            "SELECT file_name, num_rows, num_row_groups, file_size_bytes "
            "FROM parquet_file_metadata(?)", [paths]).fetchall()
        by_path = {fname: (nrows, ngroups, nbytes)
                   for fname, nrows, ngroups, nbytes in meta}
        stats[ds] = {
            table: {"rows": by_path[path][0],
                    "row_groups": by_path[path][1],
                    "bytes": by_path[path][2]}
            for table, path in catalog[ds].items()
        }
    return stats


def print_feature_lists(schemas):
    print("=" * 72)
    print("FEATURES PER DATASET")
    print("=" * 72)
    for ds, tables in schemas.items():
        print(f"\n## {ds} -- {len(tables)} tables")
        for table, cols in tables.items():
            print(f"\n{ds}/{table} ({len(cols)} columns)")
            for name, typ in cols:
                print(f"    {name:<24} {typ}")


def print_size_analysis(schemas, stats):
    print("\n" + "=" * 72)
    print("DATASET SIZE ANALYSIS")
    print("=" * 72)
    grand_rows = grand_bytes = 0
    for ds, tables in stats.items():
        total_rows = sum(t["rows"] for t in tables.values())
        total_bytes = sum(t["bytes"] for t in tables.values())
        total_cols = sum(len(schemas[ds][t]) for t in tables)
        grand_rows += total_rows
        grand_bytes += total_bytes
        print(f"\n## {ds} -- {len(tables)} tables, {total_rows:,} rows, "
              f"{total_cols} columns across tables, {total_bytes / 1048576:,.1f} MB")
        print(f"{'table':<20}{'rows':>14}{'cols':>6}{'rowgroups':>11}"
              f"{'size MB':>10}  dimensions")
        for table in sorted(tables, key=lambda t: -tables[t]["rows"]):
            s = tables[table]
            ncols = len(schemas[ds][table])
            print(f"{table:<20}{s['rows']:>14,}{ncols:>6}{s['row_groups']:>11}"
                  f"{s['bytes'] / 1048576:>10,.1f}  {s['rows']:,} x {ncols}")
        print(f"{'TOTAL':<20}{total_rows:>14,}{total_cols:>6}{'':>11}"
              f"{total_bytes / 1048576:>10,.1f}")
    if len(stats) > 1:
        print(f"\nGRAND TOTAL: {grand_rows:,} rows, "
              f"{grand_bytes / 1073741824:,.2f} GB across {len(stats)} datasets")


def print_table_matrix(schemas):
    print("\n" + "=" * 72)
    print("TABLE PRESENCE ACROSS DATASETS")
    print("=" * 72)
    datasets = list(schemas)
    width = max(len(ds) for ds in datasets) + 2
    all_tables = sorted(set().union(*(schemas[ds].keys() for ds in datasets)))
    print(f"{'table':<20}" + "".join(f"{ds:>{width}}" for ds in datasets))
    for table in all_tables:
        marks = "".join(
            f"{('yes' if table in schemas[ds] else '-'):>{width}}" for ds in datasets
        )
        print(f"{table:<20}{marks}")


def print_column_comparison(schemas):
    print("\n" + "=" * 72)
    print("COLUMN COMPARISON (tables present in 2+ datasets)")
    print("=" * 72)
    datasets = list(schemas)
    if len(datasets) < 2:
        print("\n(only one dataset selected -- nothing to compare)")
        return
    all_tables = sorted(set().union(*(schemas[ds].keys() for ds in datasets)))
    for table in all_tables:
        present = {ds: dict(schemas[ds][table]) for ds in datasets if table in schemas[ds]}
        orders = {ds: [c for c, _ in schemas[ds][table]] for ds in present}
        if len(present) < 2:
            continue

        issues = []
        union_cols = set().union(*(set(d) for d in present.values()))
        for ds, cols in present.items():
            missing = sorted(union_cols - set(cols))
            if missing:
                issues.append(f"missing in {ds}: {', '.join(missing)}")
        for col in sorted(union_cols):
            types = {ds: cols[col] for ds, cols in present.items() if col in cols}
            if len(set(types.values())) > 1:
                pairs = ", ".join(f"{ds}={t}" for ds, t in types.items())
                issues.append(f"type mismatch on {col}: {pairs}")
        shared = sorted(union_cols.intersection(*(set(d) for d in present.values())))
        shared_orders = {ds: [c for c in orders[ds] if c in shared] for ds in present}
        if len({tuple(o) for o in shared_orders.values()}) > 1:
            issues.append("column ORDER differs (align by name, never by position)")

        if issues:
            print(f"\n{table}  [{', '.join(present)}]")
            for issue in issues:
                print(f"    !! {issue}")
        else:
            print(f"\n{table}  [{', '.join(present)}]  -- identical")


def main():
    args = parse_args()
    con = connect(args.region)
    try:
        catalog = discover(con, args.bucket, args.prefix)
    except duckdb.Error as e:
        if "403" in str(e) or "Forbidden" in str(e):
            sys.exit(
                "S3 access denied. Run this first in the same terminal:\n"
                '  eval "$(aws configure export-credentials --format env)"'
            )
        raise
    if not catalog:
        sys.exit(f"no parquet datasets found under s3://{args.bucket}/{args.prefix}/")

    selected = choose_datasets(catalog, args)
    schemas = load_schemas(con, catalog, selected)
    stats = load_stats(con, catalog, selected)
    print_feature_lists(schemas)
    print_size_analysis(schemas, stats)
    print_table_matrix(schemas)
    print_column_comparison(schemas)


if __name__ == "__main__":
    main()

