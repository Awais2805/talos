#!/usr/bin/env python3
"""Profile the Talos parquet lake -- aggregated by LOG TYPE, not per file.

The parquet zone mirrors the extracted Zeek tree (one parquet per source .log),
so each dataset holds thousands of small files that are really many captures of
the same handful of log types (conn, dns, http, ssl, ...). Listing them one by
one is noise. This tool rolls them up: for each dataset x log type it reports

  1. INVENTORY   -- files, capture folders, total rows, size, column count
  2. SCHEMA      -- the UNION of every column+type seen (the feature universe),
                    with a within-type consistency check (uniform vs varies)
  3. CROSS-DATASET comparison of the log types shared by 2+ datasets
                    (missing columns, type mismatches, column-order drift)

Reads parquet footers only -- no row data is downloaded.

Usage:
    eval "$(aws configure export-credentials --format env)"   # once per terminal
    python lake_feature_discovery.py                          # menu -> stdout
    python lake_feature_discovery.py --non_interactive        # all datasets
    python lake_feature_discovery.py --datasets cic-ids-2017,cic-ids-2018
    optional: --drift-sample N   (files sampled per type for the consistency check)
"""

import argparse
import random
import subprocess
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
    p.add_argument("--drift-sample", type=int, default=200,
                   help="files sampled per log type for the schema-consistency check")
    return p.parse_args()


def _aws_creds():
    """Current S3 creds straight from the AWS CLI (handles session refresh)."""
    d = {}
    try:
        out = subprocess.run(["aws", "configure", "export-credentials", "--format", "env"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("export "):
                k, _, v = line[7:].partition("="); d[k.strip()] = v.strip()
    except Exception:
        pass
    return d


def refresh_secret(con, region):
    """Re-mint the DuckDB S3 secret from the CLI's current (rotating) creds.

    Called before each log type so a long footer-reading run can't die on an
    ExpiredToken partway through. Returns False if the CLI can no longer supply
    credentials (session fully expired -> caller should tell the user to re-auth).
    """
    cr = _aws_creds()
    akid, secret, token = (cr.get("AWS_ACCESS_KEY_ID"), cr.get("AWS_SECRET_ACCESS_KEY"),
                           cr.get("AWS_SESSION_TOKEN"))
    if not (akid and secret):
        return False
    extra = f", SESSION_TOKEN '{token}'" if token else ""
    con.execute(f"CREATE OR REPLACE SECRET aws_lake "
                f"(TYPE S3, KEY_ID '{akid}', SECRET '{secret}'{extra}, REGION '{region}');")
    return True


def connect(region):
    con = duckdb.connect()
    con.execute(f"INSTALL httpfs FROM '{EXT_REPO}'; LOAD httpfs;")
    con.execute(f"INSTALL aws FROM '{EXT_REPO}'; LOAD aws;")
    if not refresh_secret(con, region):
        sys.exit('no AWS credentials from the CLI — re-auth (aws login) or run:\n'
                 '  eval "$(aws configure export-credentials --format env)"')
    return con


def discover(con, bucket, prefix):
    """Return {dataset: {logtype: [paths]}} grouping the mirror by log type.

    dataset = first path segment under the prefix; logtype = file basename
    (conn, dns, ...), so every capture folder's conn.parquet is one group.
    """
    root = f"s3://{bucket}/{prefix}/"
    files = [r[0] for r in con.execute(
        "SELECT file FROM glob(?)", [root + "**/*.parquet"]).fetchall()]
    catalog = defaultdict(lambda: defaultdict(list))
    for path in files:
        rel = path[len(root):]
        segs = rel.split("/")
        dataset = segs[0] if len(segs) > 1 else "(root)"
        logtype = segs[-1].removesuffix(".parquet")
        catalog[dataset][logtype].append(path)
    return {ds: dict(lts) for ds, lts in catalog.items()}


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
        nfiles = sum(len(p) for p in catalog[ds].values())
        print(f"  [{i}] {ds:<28} ({len(catalog[ds])} log types, {nfiles:,} files)", file=err)
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


def captures_of(paths, dataset):
    """Distinct first-level folder under the dataset root -- the capture split."""
    caps = set()
    for p in paths:
        rel = p.split(f"/{dataset}/", 1)[-1]
        caps.add(rel.split("/", 1)[0] if "/" in rel else "(root)")
    return caps


def summarize(con, bucket, prefix, dataset, logtype, paths, drift_sample):
    """Roll up one dataset x log type: counts, size, union schema, consistency."""
    glob = f"s3://{bucket}/{prefix}/{dataset}/**/{logtype}.parquet"

    nfiles, rows, nbytes = con.execute(
        "SELECT count(*), coalesce(sum(num_rows), 0), coalesce(sum(file_size_bytes), 0) "
        "FROM parquet_file_metadata(?)", [glob]).fetchone()

    # union schema (friendly DuckDB types), reading footers only
    schema = [(r[0], r[1]) for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{glob}', union_by_name=true)").fetchall()]

    # schema-consistency check on a bounded sample of files
    sample = paths if len(paths) <= drift_sample else random.sample(paths, drift_sample)
    n_sampled, n_sigs = con.execute(
        "SELECT count(DISTINCT file_name), count(DISTINCT sig) FROM ("
        "  SELECT file_name, string_agg(name, ',' ORDER BY name) sig "
        "  FROM parquet_schema(?) WHERE type IS NOT NULL GROUP BY file_name)",
        [sample]).fetchone()

    optional_cols = []
    if n_sigs and n_sigs > 1:  # only dig into which columns vary when they do
        optional_cols = [r[0] for r in con.execute(
            "SELECT name FROM parquet_schema(?) WHERE type IS NOT NULL "
            "GROUP BY name HAVING count(DISTINCT file_name) < ?",
            [sample, n_sampled]).fetchall()]

    return {"nfiles": nfiles, "rows": rows, "bytes": nbytes, "schema": schema,
            "captures": len(captures_of(paths, dataset)),
            "n_sampled": n_sampled, "n_sigs": n_sigs or 1, "optional": optional_cols}


def collect(con, catalog, selected, bucket, prefix, drift_sample, region):
    """Return {dataset: {logtype: summary}}, printing progress to stderr."""
    out = {}
    for ds in selected:
        out[ds] = {}
        for lt in sorted(catalog[ds]):
            if not refresh_secret(con, region):  # keep creds fresh across a long run
                sys.exit("\nAWS session expired mid-run and the CLI can no longer mint "
                         "credentials — re-auth (aws login) and rerun.")
            print(f"  ... {ds}/{lt}", file=sys.stderr, flush=True)
            try:
                out[ds][lt] = summarize(con, bucket, prefix, ds, lt,
                                        catalog[ds][lt], drift_sample)
            except duckdb.Error as e:
                print(f"      skip {ds}/{lt}: {e}", file=sys.stderr)
    return out


def print_inventory(data):
    print("=" * 78)
    print("INVENTORY  (rolled up by log type)")
    print("=" * 78)
    g_files = g_rows = g_bytes = 0
    for ds, lts in data.items():
        d_files = sum(s["nfiles"] for s in lts.values())
        d_rows = sum(s["rows"] for s in lts.values())
        d_bytes = sum(s["bytes"] for s in lts.values())
        g_files += d_files; g_rows += d_rows; g_bytes += d_bytes
        print(f"\n## {ds} -- {len(lts)} log types, {d_files:,} files, "
              f"{d_rows:,} rows, {d_bytes / 1048576:,.1f} MB")
        print(f"{'log type':<16}{'files':>8}{'captures':>10}{'rows':>16}"
              f"{'cols':>6}{'MB':>10}")
        for lt in sorted(lts, key=lambda t: -lts[t]["rows"]):
            s = lts[lt]
            print(f"{lt:<16}{s['nfiles']:>8,}{s['captures']:>10,}{s['rows']:>16,}"
                  f"{len(s['schema']):>6}{s['bytes'] / 1048576:>10,.1f}")
        print(f"{'TOTAL':<16}{d_files:>8,}{'':>10}{d_rows:>16,}{'':>6}"
              f"{d_bytes / 1048576:>10,.1f}")
    if len(data) > 1:
        print(f"\nGRAND TOTAL: {g_files:,} files, {g_rows:,} rows, "
              f"{g_bytes / 1073741824:,.2f} GB across {len(data)} datasets")


def print_schemas(data):
    print("\n" + "=" * 78)
    print("UNION SCHEMA per log type  (the feature universe)")
    print("=" * 78)
    for ds, lts in data.items():
        print(f"\n## {ds}")
        for lt in sorted(lts):
            s = lts[lt]
            if s["n_sigs"] == 1:
                note = f"uniform across {s['nfiles']:,} files"
                if s["n_sampled"] < s["nfiles"]:
                    note += f" (checked {s['n_sampled']})"
            else:
                note = (f"VARIES across files ({s['n_sigs']} schemas in "
                        f"{s['n_sampled']} sampled)")
                if s["optional"]:
                    note += f"; not-universal cols: {', '.join(sorted(s['optional']))}"
            print(f"\n{lt}  ({len(s['schema'])} columns, {note})")
            for name, typ in s["schema"]:
                print(f"    {name:<26} {typ}")


def print_cross_dataset(data):
    print("\n" + "=" * 78)
    print("CROSS-DATASET COMPARISON  (log types in 2+ datasets)")
    print("=" * 78)
    datasets = list(data)
    if len(datasets) < 2:
        print("\n(only one dataset selected -- nothing to compare)")
        return
    all_lts = sorted(set().union(*(set(lts) for lts in data.values())))
    for lt in all_lts:
        present = {ds: dict(data[ds][lt]["schema"]) for ds in datasets if lt in data[ds]}
        if len(present) < 2:
            continue
        orders = {ds: [c for c, _ in data[ds][lt]["schema"]] for ds in present}
        union_cols = set().union(*(set(d) for d in present.values()))

        issues = []
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

        tag = f"[{', '.join(present)}]"
        if issues:
            print(f"\n{lt}  {tag}")
            for issue in issues:
                print(f"    !! {issue}")
        else:
            print(f"\n{lt}  {tag}  -- identical schema")


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
    print("profiling (footers only) ...", file=sys.stderr)
    data = collect(con, catalog, selected, args.bucket, args.prefix,
                   args.drift_sample, args.region)
    print_inventory(data)
    print_schemas(data)
    print_cross_dataset(data)


if __name__ == "__main__":
    main()
