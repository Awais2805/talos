#!/usr/bin/env python3
"""Profile ONE view of ONE dataset into a self-contained statistical summary.

A VIEW, not just a dataset: the same flows labelled by `schedule` and by `ae`
are two different populations of `label_class`, and a profile that cannot tell
them apart cannot honestly be compared with anything. So a profile is
identified by (dataset, zone, method) -- `prelabel_cic-ids-2017`,
`postlabel-schedule_cic-ids-2017`, `postlabel-ae_cic-ids-2017` -- and that
identity is in the filename, in the metadata, and in what `compare` will
consent to put side by side.

This is the only EDA stage that touches the lake, and it touches it once: a
single grouped aggregate over one view's conn flows produces every number both
reports will ever need. Everything is keyed by label_class, so "attack vs
benign", "overall", and "per class" are all merges over the same summary rather
than separate scans.

What the profile stores is deliberately not a set of finished answers but a set
of SUFFICIENT STATISTICS -- counts, sums, sums of squares, pairwise sums of
products, and fixed-grid histograms. Those are additive, so compare.py can
build the exact pooled mean, variance and correlation matrix of every OTHER
dataset in the pool from their profiles alone, and never re-reads 230M flows to
answer "how does this dataset differ from the rest". Adding the fourth dataset
costs one scan, not four.

Cost note: this reads every row of one view (2019 is ~72.5M flows / 2.4 GB).
Run it on the box, or use --sample for a smoke test.

Usage:
    talos eda --dataset cic-ids-2017                  # post-label, schedule
    talos eda --dataset cic-ids-2017 --method ae      # post-label, ae
    talos eda --dataset cic-ids-2017 --zone parquet   # pre-label
    talos eda --dataset cic-ddos-2019 --sample 2
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
from talos.data.flows import CONN, FlowTable, SourceError, feature_space_of
from talos.data.profiling.eda import compare
from talos.data.profiling.eda.parse import parse_row, summarise
from talos.data.profiling.eda.query import build_capture_query, build_stats_query
from talos.data.profiling.eda.spec import Spec, _q

PROFILE_VERSION = 3


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="dataset name as it appears in the lake")
    p.add_argument("--source", default=None,
                   help="parquet glob to profile instead of resolving one "
                        "(you own the scope; nothing is checked)")
    # `labelled`, because `group_by: label_class` is the whole design and the
    # parquet zone has no such column -- profiling it collapses every flow into
    # one `(unlabelled)` group, so class imbalance is silently not measured.
    # It is still the right zone for a pre-label shape check, hence the choice.
    p.add_argument("--zone", default="labelled", choices=["parquet", "labelled", "canonical"],
                   help="lake zone to read (default: labelled = a post-label profile)")
    p.add_argument("--method", default="schedule",
                   help="which labelling method's table to profile, when "
                        "--zone labelled. Part of the profile's identity: "
                        "schedule and ae are different populations, not two "
                        "readings of one")
    p.add_argument("--feature-space", default=None)
    p.add_argument("--log", default=CONN,
                   help="which Zeek log type to profile (default: conn, the flow backbone)")
    p.add_argument("--captures", nargs="+", default=None,
                   help="restrict --zone parquet to named capture subfolders")
    p.add_argument("--include-invalid", action="store_true",
                   help="profile every row, including those a validity invariant "
                        "rejects (default: exclude them, as the training pools do)")
    p.add_argument("--sample", type=float, default=None,
                   help="profile a random %% of rows (recorded in the profile)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N rows — a SECONDS-long smoke test of the whole "
                        "pipeline. Biased (it is the head of the file, so the earliest "
                        "flows), so the profile is marked partial and must not be reported")
    p.add_argument("--out", default=None, help="profile JSON path")
    # Comparison is opt-in, not a side effect of profiling: a profile alone
    # (e.g. the first-ever run, or the only dataset profiled so far) needs no
    # peer to be useful, and a comparison silently computed against whatever
    # else happens to be sitting in the directory is exactly the "is this even
    # comparing the right things" risk this pair of flags exists to remove.
    cascade = p.add_mutually_exclusive_group()
    cascade.add_argument("--compare-with", default=None, metavar="DATASET",
                         help="after profiling, compare against exactly this "
                              "one other dataset's profile OF THE SAME VIEW")
    cascade.add_argument("--compare-all", action="store_true",
                         help="after profiling, regenerate comparisons across "
                              "every profile that shares this profile's view")
    p.add_argument("--no-render", action="store_true",
                   help="with --compare-all, write JSON only, no HTML")
    # Default to None so the machine's resource profile decides. A flag here
    # is a deliberate per-run override, not a guess baked in at import time.
    p.add_argument("--memory-limit", default=None, help="override DuckDB memory limit")
    p.add_argument("--threads", type=int, default=None, help="override DuckDB threads")
    p.add_argument("--temp-dir", default=None, help="override DuckDB spill directory")
    p.add_argument("--max-temp", default=None, help="override the spill cap")
    p.add_argument("--spec", default=None, help="override the EDA spec file")
    p.add_argument("--config", default=str(default_config()))
    return p.parse_args(argv)


def target_for(cfg, a) -> FlowTable:
    """What this run profiles, as one addressable identity.

    Built before anything is read, because every later decision -- the glob,
    the filename, which profiles this one may be compared against -- is a
    property of it rather than of the flags that produced it.
    """
    return FlowTable(
        dataset=a.dataset, zone=a.zone,
        feature_space=a.feature_space or feature_space_of(cfg),
        method=a.method if a.zone == "labelled" else None,
        captures=tuple(a.captures or ()), log=a.log)


def main(argv=None):
    a = parse_args(argv)
    cfg = Config.load(a.config)
    # `lake_source` = which of the three provenance kinds this dataset is, not
    # to be confused with `a.source`/`src`, which is the parquet glob. A ROLE is
    # deliberately not read here: a role belongs to an experiment, and a profile
    # is a property of the data -- the same profile whichever study reads it.
    region, lake_source = cfg.region, cfg.source_of(a.dataset)
    eda_out = cfg.eda_dir

    target = target_for(cfg, a)
    try:
        src = [a.source] if a.source else target.globs(cfg, cfg.lake())
    except SourceError as exc:
        sys.exit(f"error: {exc}")

    spec = Spec(a.spec) if a.spec else Spec()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"dataset   {a.dataset}  (from: {lake_source})")
    print(f"view      {target.view}")
    for one in src:
        print(f"source    {one}")
    partial = ("; ".join(x for x in (
        f"{a.sample}% random sample" if a.sample else "",
        f"first {a.limit:,} rows only" if a.limit else "") if x) or None)

    print(f"spec      v{spec.version} sha {spec.sha}"
          f"{'  PARTIAL: ' + partial if partial else ''}")
    print(cfg.resources.override(memory_limit=a.memory_limit, threads=a.threads,
                                temp_dir=a.temp_dir, max_temp=a.max_temp).describe())

    lake = DuckEngine(remote=zones.is_remote(src[0]), region=region,
                      profile=cfg.resources,
                      memory_limit=a.memory_limit, temp_dir=a.temp_dir,
                      threads=a.threads, max_temp=a.max_temp)
    cols = lake.columns(src[0])
    num, cat, ident, nulls, skipped = spec.resolve(cols)
    if not num:
        sys.exit(f"none of the spec's numeric features exist in {src[0]}")
    # An unlabelled zone still profiles cleanly -- it just has one class. That
    # keeps this usable on `parquet` for a pre-labelling sanity check.
    group_expr = _q(spec.group_by) if spec.group_by in cols else "'(unlabelled)'"

    idx = {f.name: i for i, f in enumerate(num)}
    featured = [[x, y] for x, y in spec.joint_pairs if x in idx and y in idx]

    print(f"features  {len(num)} numeric, {len(cat)} categorical, "
          f"{len(Spec.pairs(num))} feature pairs, {len(Spec.pairs(num))} joint densities"
          f"{f', {len(skipped)} skipped' if skipped else ''}")
    print("\nscanning (one pass) ...", flush=True)

    # The same rows the training pools draw from: a distribution the report
    # shows should be a distribution of what a model actually sees.
    valid = ""
    if not a.include_invalid:
        from talos.data.preprocess.prelabel.verify import valid_sql
        valid = valid_sql(cols)
        if valid == "TRUE":
            valid = ""
    if valid:
        print("rows      excluding flows that break a validity invariant "
              "(--include-invalid to keep them)")

    t0 = time.time()
    rows = lake.sql(build_stats_query(src, spec, num, cat, ident, nulls, group_expr,
                                      a.sample, a.limit, valid))
    by_class = dict(parse_row(r, spec, num, cat, ident, nulls) for r in rows)
    elapsed = time.time() - t0

    captures = []
    if "capture" in cols and "ts" in cols:
        for cap, grp, n, t_0, t_1 in lake.sql(
                build_capture_query(src, group_expr, a.sample, a.limit, valid)):
            captures.append({"capture": cap, "class": grp, "n": int(n),
                             "ts_min": t_0, "ts_max": t_1})

    profile = {
        "profile_version": PROFILE_VERSION,
        "meta": {
            "dataset": a.dataset, "lake_source": lake_source,
            # The identity, carried WITH the measurements. `view` is what
            # `compare` groups on: two profiles of different views describe
            # different populations, and putting them side by side would
            # measure the labeller rather than the domain.
            "view": target.view, "zone": a.zone, "method": target.method,
            "feature_space": target.feature_space, "log": target.log,
            "source": src,
            "generated_utc": started, "scan_seconds": round(elapsed, 1),
            "spec_version": spec.version, "spec_sha": spec.sha,
            # Which population this describes; a profile over a different one is
            # not comparable with this, however identical the ruler.
            "excludes_invalid": bool(valid),
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

    out = Path(a.out) if a.out else eda_out / f"profile_{target.slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=1) + "\n")
    summarise(profile)
    print(f"\nscanned in {elapsed:,.1f}s -> {out}")

    # ---- comparison, only if asked for -------------------------------------
    # A new dataset changes what "the rest" means for every dataset already in
    # the pool, so every comparison is stale the moment this file lands. They
    # are regenerated from JSON alone -- no lake access, seconds not hours --
    # but only when the caller names which comparison they want, and only ever
    # among profiles that share this one's view.
    if a.compare_with:
        made = compare.compare_pair(a.dataset, a.compare_with, eda_out,
                                    view=target.view)
        print(f"comparison written: {made.name}")
    elif a.compare_all:
        from talos.data.profiling.eda import render
        made = compare.regenerate(eda_out, view=target.view)
        print(f"comparisons regenerated: {', '.join(m.name for m in made) or '(none)'}")
        if not a.no_render:
            print(f"rendered: {len(render.render_all(eda_out))} page(s) -> "
                  f"{eda_out / 'index.html'}")


if __name__ == "__main__":
    main()
