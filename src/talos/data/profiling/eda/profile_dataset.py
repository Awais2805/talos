#!/usr/bin/env python3
"""Profile ONE dataset into a self-contained statistical summary.

This is the only EDA stage that touches the lake, and it touches it once: a
single grouped aggregate over the labelled conn flows produces every number
both reports will ever need. Everything is keyed by label_class, so "attack vs
benign", "overall", and "per class" are all merges over the same summary rather
than separate scans.

What the profile stores is deliberately not a set of finished answers but a set
of SUFFICIENT STATISTICS -- counts, sums, sums of squares, pairwise sums of
products, and fixed-grid histograms. Those are additive, so compare.py can
build the exact pooled mean, variance and correlation matrix of every OTHER
dataset in the pool from their profiles alone, and never re-reads 230M flows to
answer "how does this dataset differ from the rest". Adding the fourth dataset
costs one scan, not four.

Cost note: this reads every row of one dataset's conn zone (2019 is ~72.5M
flows / 2.4 GB). Run it on the EC2 box, or use --sample for a smoke test.

Usage:
    eval "$(aws configure export-credentials --format env)"
    python -m talos.eda.profile_dataset --dataset cic-ids-2017
    python -m talos.eda.profile_dataset --dataset cic-ddos-2019 --sample 2
    python -m talos.eda.profile_dataset --dataset cic-ids-2018 --no-cascade
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from talos.common import zones
from talos.common.config import Config
from talos.common.duck import DuckEngine
from talos.common.paths import default_config
from talos.data.profiling.eda import compare
from talos.data.profiling.eda.spec import Spec, _q

PROFILE_VERSION = 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="dataset name as it appears in the lake")
    p.add_argument("--source", default=None,
                   help="parquet glob to profile (default: the dataset's labelled conn flows)")
    p.add_argument("--zone", default="labelled", choices=["labelled", "canonical", "parquets"],
                   help="lake zone to read when --source is not given")
    p.add_argument("--sample", type=float, default=None,
                   help="profile a random %% of rows (recorded in the profile)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N rows — a SECONDS-long smoke test of the whole "
                        "pipeline. Biased (it is the head of the file, so the earliest "
                        "flows), so the profile is marked partial and must not be reported")
    p.add_argument("--out", default=None, help="profile JSON path")
    p.add_argument("--no-cascade", action="store_true",
                   help="do not regenerate the comparison reports afterwards")
    p.add_argument("--no-render", action="store_true", help="write JSON only, no HTML")
    p.add_argument("--memory-limit", default="8GB", help="DuckDB memory limit")
    p.add_argument("--threads", type=int, default=4, help="DuckDB threads")
    p.add_argument("--temp-dir", default=None, help="DuckDB spill directory")
    p.add_argument("--max-temp", default="20GB", help="cap on the spill directory")
    p.add_argument("--spec", default=None, help="override the EDA spec file")
    p.add_argument("--config", default=str(default_config()))
    return p.parse_args()


# --------------------------------------------------------------- query build

def scan_clause(src, sample, limit):
    """The source expression both queries read from.

    LIMIT is pushed into the scan, so a smoke test stops after N rows instead of
    reading the whole zone -- the difference between validating the pipeline in
    five seconds and finding out twenty minutes into 2019 that a column is not
    the type the spec assumed. SAMPLE still reads everything; it trades
    aggregation work for statistical validity, which is the opposite trade.
    """
    sql = f"SELECT * FROM read_parquet('{src}', union_by_name = true)"
    if sample:
        sql += f" USING SAMPLE {sample} PERCENT (bernoulli)"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return sql


def build_stats_query(src, spec, num, cat, ident, nulls, group_expr, sample, limit):
    """One pass, one GROUP BY, every heavy statistic.

    The `f` CTE projects each feature exactly once (raw value + transformed
    value); the aggregate then references those aliases, so a 16-feature spec
    costs 16 expression evaluations per row rather than one per bin. Aggregates
    are packed into LIST literals -- `[sum(a), sum(b)]` -- which keeps the
    result to a couple of dozen columns instead of a thousand, and keeps the
    parse order tied to the spec order rather than to generated column names.
    """
    proj = ["%s AS grp" % group_expr]
    proj += [f"{f.raw} AS r{i}" for i, f in enumerate(num)]
    proj += [f"{f.trans} AS t{i}" for i, f in enumerate(num)]
    proj += [f"{f.bin_index(f'r{i}')} AS g{i}" for i, f in enumerate(num)]
    proj += [f"{f.expr} AS k{j}" for j, f in enumerate(cat)]
    proj += [f"{_q(c)} AS id{k}" for k, c in enumerate(ident)]
    proj += [f"{_q(c)} AS nz{k}" for k, c in enumerate(nulls)]

    qs = ", ".join(str(q) for q in spec.quantiles)

    agg = ["grp", "count(*) AS n"]
    for i, f in enumerate(num):
        agg.append(
            f"[count(*) FILTER (r{i} IS NULL)::DOUBLE, count(*) FILTER (r{i} = 0)::DOUBLE, "
            f"min(r{i})::DOUBLE, max(r{i})::DOUBLE, sum(t{i})::DOUBLE, "
            f"sum(t{i} * t{i})::DOUBLE] AS m{i}")
        agg.append("[" + ", ".join(f"({c})::DOUBLE" for c in f.bin_counts(f"r{i}")) + f"] AS h{i}")
        agg.append(f"approx_quantile(r{i}::DOUBLE, [{qs}]) AS q{i}")
    pairs = Spec.pairs(num)
    agg.append("[" + ", ".join(f"sum(t{a} * t{b})::DOUBLE" for a, b in pairs) + "] AS xy"
               if pairs else "[]::DOUBLE[] AS xy")
    # The joint bin histogram of EVERY pair, as one MAP per pair: both bin
    # indices packed into a single integer key. This is what makes a rank
    # correlation possible at all downstream -- true ranks depend on the whole
    # dataset and could never be merged across datasets, but joint bin counts
    # are plain counts, so they add exactly like everything else here. It also
    # means any pair can be drawn as a 2-D density surface, not just a
    # pre-chosen few. Sparse in practice: most cells of most pairs are empty.
    agg += [f"histogram(g{a} * 1000 + g{b}) AS j{a}_{b}" for a, b in pairs]
    agg += [f"histogram(k{j}) AS c{j}" for j in range(len(cat))]
    if ident:
        # Identity columns are counted, never valued: 2019's spoofed sources
        # alone would put millions of addresses in a histogram, and an IP is a
        # fact about the testbed, not a feature.
        agg.append("[" + ", ".join(f"approx_count_distinct(id{k})::DOUBLE"
                                   for k in range(len(ident))) + "] AS dst")
    if nulls:
        agg.append("[" + ", ".join(f"count(*) FILTER (nz{k} IS NULL)::DOUBLE"
                                   for k in range(len(nulls))) + "] AS nul")

    return (f"WITH src AS (\n  {scan_clause(src, sample, limit)}\n"
            f"), f AS (\n  SELECT " + ",\n         ".join(proj) + "\n  FROM src\n)\n"
            "SELECT " + ",\n       ".join(agg) + "\nFROM f GROUP BY grp ORDER BY n DESC")


def build_capture_query(src, group_expr, sample, limit):
    """Domain sub-structure: flows and time span per capture x class.

    Cheap next to the main pass -- parquet column pruning means it reads three
    columns, not twenty -- and it is what turns "this dataset" into "these ten
    capture days", which is the unit domain shift actually happens in.
    """
    return (f"WITH src AS (\n"
            f"  SELECT capture, ts, {group_expr} AS grp "
            f"FROM ({scan_clause(src, sample, limit)})\n)\n"
            f"SELECT capture, grp, count(*)::DOUBLE AS n, "
            f"min(ts)::DOUBLE AS t0, max(ts)::DOUBLE AS t1 "
            f"FROM src GROUP BY 1, 2 ORDER BY 3 DESC")


# ------------------------------------------------------------------ parsing

def parse_row(row, spec, num, cat, ident, nulls):
    """Unpack one group's aggregate row using the spec order it was built from."""
    it = iter(row)
    grp, n = next(it), int(next(it))
    numeric = {}
    for f in num:
        m, h, q = next(it), next(it), next(it)
        numeric[f.name] = {
            "n_undefined": int(m[0] or 0), "n_zero": int(m[1] or 0),
            "min": m[2], "max": m[3], "sum_t": m[4] or 0.0, "sumsq_t": m[5] or 0.0,
            "hist": [int(x) for x in h],
            "q": dict(zip((str(x) for x in spec.quantiles), list(q or []))),
        }
    xy = [float(v) for v in (next(it) or [])]

    # Unpack the packed 2-D keys back into sparse bins x bins surfaces, in the
    # same pair order as pairs_xy.
    joints = []
    for _ in Spec.pairs(num):
        joints.append({f"{int(k) // 1000},{int(k) % 1000}": int(v)
                       for k, v in (next(it) or {}).items()})

    categorical = {}
    for f in cat:
        counts = {str(k): int(v) for k, v in (next(it) or {}).items()}
        top = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:spec.top_k])
        categorical[f.name] = {
            "counts": top,
            "n_other": sum(counts.values()) - sum(top.values()),
            "n_distinct": len(counts),
        }
    distinct = ({c: int(v) for c, v in zip(ident, next(it))} if ident else {})
    nullc = ({c: int(v) for c, v in zip(nulls, next(it))} if nulls else {})
    return grp, {"n": n, "numeric": numeric, "pairs_xy": xy, "pairs_joint": joints,
                 "categorical": categorical, "distinct": distinct, "nulls": nullc}


# ------------------------------------------------------------------ reporting

def summarise(profile):
    """Console summary -- the same numbers the HTML leads with."""
    meta, by = profile["meta"], profile["by_class"]
    total = meta["rows"]
    print(f"\n{'class':<16}{'flows':>16}{'share':>9}")
    for cls, blk in sorted(by.items(), key=lambda kv: -kv[1]["n"]):
        print(f"{cls:<16}{blk['n']:>16,}{100 * blk['n'] / total:>8.3f}%")
    print(f"{'-' * 41}\n{'TOTAL':<16}{total:>16,}")
    atk = total - by.get(profile["benign_class"], {}).get("n", 0)
    print(f"\nattack ratio {100 * atk / total:.3f}%  "
          f"({len(by)} classes, {meta['numeric_features']} numeric features, "
          f"{meta['categorical_features']} categorical)")
    if meta["skipped_features"]:
        print("\nnot measurable on this source (missing columns):")
        for name, cols in meta["skipped_features"].items():
            print(f"  {name:<22} needs {', '.join(cols)}")


def main():
    a = parse_args()
    cfg = Config.load(a.config)
    region, role = cfg.region, cfg.role(a.dataset)
    eda_out = cfg.eda_dir

    src = a.source or cfg.parquet_glob(a.zone, a.dataset)
    spec = Spec(a.spec) if a.spec else Spec()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"dataset   {a.dataset}  (role: {role})")
    print(f"source    {src}")
    partial = ("; ".join(x for x in (
        f"{a.sample}% random sample" if a.sample else "",
        f"first {a.limit:,} rows only" if a.limit else "") if x) or None)

    print(f"spec      v{spec.version} sha {spec.sha}"
          f"{'  PARTIAL: ' + partial if partial else ''}")
    print(f"duckdb    {a.threads} threads, {a.memory_limit} memory"
          f"{f', spill {a.temp_dir} capped {a.max_temp}' if a.temp_dir else ''}")

    lake = DuckEngine(remote=zones.is_remote(src), region=region,
                      memory_limit=a.memory_limit, temp_dir=a.temp_dir,
                      threads=a.threads, max_temp=a.max_temp)
    cols = lake.columns(src)
    num, cat, ident, nulls, skipped = spec.resolve(cols)
    if not num:
        sys.exit(f"none of the spec's numeric features exist in {src}")
    # An unlabelled zone still profiles cleanly -- it just has one class. That
    # keeps this usable on `parquets` for a pre-labelling sanity check.
    group_expr = _q(spec.group_by) if spec.group_by in cols else "'(unlabelled)'"

    idx = {f.name: i for i, f in enumerate(num)}
    featured = [[a, b] for a, b in spec.joint_pairs if a in idx and b in idx]

    print(f"features  {len(num)} numeric, {len(cat)} categorical, "
          f"{len(Spec.pairs(num))} feature pairs, {len(Spec.pairs(num))} joint densities"
          f"{f', {len(skipped)} skipped' if skipped else ''}")
    print("\nscanning (one pass) ...", flush=True)

    t0 = time.time()
    rows = lake.sql(build_stats_query(src, spec, num, cat, ident, nulls, group_expr,
                                      a.sample, a.limit))
    by_class = dict(parse_row(r, spec, num, cat, ident, nulls) for r in rows)
    elapsed = time.time() - t0

    captures = []
    if "capture" in cols and "ts" in cols:
        for cap, grp, n, t_0, t_1 in lake.sql(
                build_capture_query(src, group_expr, a.sample, a.limit)):
            captures.append({"capture": cap, "class": grp, "n": int(n),
                             "ts_min": t_0, "ts_max": t_1})

    profile = {
        "profile_version": PROFILE_VERSION,
        "meta": {
            "dataset": a.dataset, "role": role, "source": src, "zone": a.zone,
            "generated_utc": started, "scan_seconds": round(elapsed, 1),
            "spec_version": spec.version, "spec_sha": spec.sha,
            # A partial profile is still a valid profile -- it just describes a
            # slice. It is carried through to both reports as a banner so a
            # smoke-test run can never be mistaken for the real thing.
            "sample_pct": a.sample, "limit_rows": a.limit, "partial": partial,
            "duckdb": duckdb.__version__,
            "rows": sum(b["n"] for b in by_class.values()),
            "columns": sorted(cols),
            "numeric_features": len(num), "categorical_features": len(cat),
            "skipped_features": skipped,
        },
        "benign_class": spec.benign_class,
        # The ruler, carried WITH the measurements: a reader (or a future
        # comparison) never has to go find the spec to know what bin 7 means.
        "numeric_spec": {f.name: {"transform": f.transform, "edges": f.edges,
                                  "bin_labels": f.bin_labels(), "expr": f.expr}
                         for f in num},
        "categorical_spec": {f.name: {"expr": f.expr} for f in cat},
        "pair_order": [[num[i].name, num[j].name] for i, j in Spec.pairs(num)],
        "featured_pairs": featured,
        "quantiles": [str(q) for q in spec.quantiles],
        "identity_columns": ident,
        "by_class": by_class,
        "captures": captures,
    }

    out = Path(a.out) if a.out else eda_out / f"profile_{a.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=1) + "\n")
    summarise(profile)
    print(f"\nscanned in {elapsed:,.1f}s -> {out}")

    # ---- the inverse dependency -------------------------------------------
    # A new dataset changes what "the rest" means for every dataset already in
    # the pool, so every comparison is stale the moment this file lands. They
    # are regenerated from JSON alone -- no lake access, seconds not hours.
    if not a.no_cascade:
        from talos.data.profiling.eda import render
        made = compare.regenerate(eda_out)
        print(f"comparisons regenerated: {', '.join(m.name for m in made) or '(none)'}")
        if not a.no_render:
            print(f"rendered: {len(render.render_all(eda_out))} page(s) -> "
                  f"{eda_out / 'index.html'}")


if __name__ == "__main__":
    main()
