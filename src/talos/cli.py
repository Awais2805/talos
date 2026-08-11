#!/usr/bin/env python3
"""Talos command line.

    talos init [path]     create a lake and a config pointing at it
    talos config          show where everything resolves to
    talos extract         run the configured extractor over the raw zone
    talos convert         convert extracted logs to parquet or csv
    talos label           attach ground truth from the attack schedule
    talos discover        profile the lake by log type
    talos eda             profile one dataset -> reports
    talos compare         rebuild comparisons from existing profiles
    talos render          rebuild HTML from existing JSON

Stage commands forward unrecognised flags to the underlying module, so anything
the module accepts still works: `talos eda --dataset X --threads 8`.
"""

import argparse
import runpy
import sys
import tempfile
from pathlib import Path

from talos.common import zones
from talos.common.config import Config, ConfigError
from talos.data.extraction import get_extractor


CONFIG_TEMPLATE = """\
# Talos — central configuration.
# Everything that decides WHERE data lives is here; no script carries its own
# bucket name or prefix. Point lake.root at a local directory or an s3:// URI
# and the whole pipeline follows.
lake:
  root: "{root}"
  zones:
{zones}
reports:
  dir: "reports"

# Which plugins this project uses. `talos config` shows what is registered.
extractor: zeek
model: xgboost

# Only consulted when lake.root is an s3:// URI.
aws:
  region: eu-north-1

# role governs how a dataset may be used downstream:
#   train    -- may contribute flows to a training pool
#   holdout  -- evaluation only; MUST NOT enter a training pool
datasets: {{}}
"""

# --------------------------------------------------------------------- init

def cmd_init(args) -> int:
    """Create a lake wherever it is asked to live.

    Local directory, S3 bucket, GCS, Azure — one code path. `LakeClient` decides
    what "make this exist" means for the backend behind the URI; nothing here
    knows the difference.
    """
    from talos.common.lake.lake import LakeClient, LakeError

    root = args.path
    try:
        lake = LakeClient(root=root)
        made = lake.init()
    except LakeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"lake  {lake.root}   ({'remote' if lake.remote else 'local'})")
    for source in zones.SOURCES:
        dirs = [d for d in made if d.source == source]
        if not dirs:
            continue
        print(f"\n  sources/{source}/   {zones.SOURCE_DESCRIPTIONS[source]}")
        for d in dirs:
            mark = "+" if d.created else " "
            print(f"    {mark} {d.zone + '/':<12} {zones.DESCRIPTIONS[d.zone]}")
    shared = [d for d in made if d.source is None]
    if shared:
        print("\n  shared across sources")
        for d in shared:
            mark = "+" if d.created else " "
            print(f"    {mark} {d.name + '/':<12} {zones.DESCRIPTIONS[d.zone]}")

    cfg_path = Path(args.config or "config.yml")
    if cfg_path.exists():
        print(f"\nconfig {cfg_path} already exists, left alone")
    else:
        # The zone block is generated rather than written out, so a fresh config
        # cannot name zones that differ from the directories just created.
        block = "".join(f"    {zone + ':':<11}{zones.DEFAULT_TEMPLATES[zone]!r}\n"
                        .replace("'", '"') for zone in zones.ZONE_ORDER)
        cfg_path.write_text(CONFIG_TEMPLATE.format(root=root, zones=block))
        print(f"\nconfig {cfg_path} written")

    print(f"\nNext: put captures in "
          f"{lake.uri('raw', dataset='<dataset>', source='<source>')}/ "
          f"then `talos extract`")
    return 0

# ------------------------------------------------------------------- config

def cmd_config(args) -> int:
    cfg = Config.load(args.config)
    print(cfg.describe())
    if not cfg.is_remote and not Path(cfg.root).exists():
        print(f"\nnote: {cfg.root} does not exist yet — run `talos init {cfg.root}`")
    return 0


# ------------------------------------------------------- stage passthroughs
# Each stage is still a runnable module; the CLI is a front door, not a rewrite.

STAGES = {
    "discover": "talos.data.discovery.lake_feature_discovery",
    "eda":      "talos.data.profiling.eda.profile_dataset",
    "compare":  "talos.data.profiling.eda.compare",
    "render":   "talos.data.profiling.eda.render",
}

def run_stage(module: str, extra: list[str]) -> int:
    argv = sys.argv
    sys.argv = [module.rsplit(".", 1)[-1], *extra]
    try:
        runpy.run_module(module, run_name="__main__")
        return 0
    except SystemExit as exc:                       # argparse / sys.exit from the stage
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        # sys.exit("message") means print and fail. int() on it raises, which
        # buried every stage's own error message under a traceback.
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = argv

# ------------------------------------------------------------------- extract

def cmd_extract(args, extra) -> int:
    """Run the configured extractor over the raw zone into the extracted zone."""
    from talos.data.extraction.base import ExtractionError

    cfg = Config.load(args.config)
    try:
        extractor = get_extractor(cfg.extractor, **cfg.doc.get(cfg.extractor, {}))
        ok, why = extractor.available()
        if not ok:
            # Checked BEFORE any pcap is fetched. Discovering a tool is unusable
            # after downloading several hundred GB is an expensive error message.
            print(f"error: {why}", file=sys.stderr)
            return 2
        feature_space = extractor.feature_space
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    lake = cfg.lake()
    datasets = list(extra) if extra else list(cfg.datasets)
    if not datasets:
        print("no datasets given, and none declared in config", file=sys.stderr)
        return 1

    print(f"extractor     {cfg.extractor}")
    print(f"feature space {feature_space}")

    failed_any = False
    for dataset in datasets:
        source = cfg.source_of(dataset)
        raw_prefix = lake.uri("raw", dataset=dataset, source=source)
        pcaps = [f for f in lake.list(raw_prefix)
                 if f.lower().endswith((".pcap", ".pcapng", ".cap"))]
        if not pcaps:
            print(f"\n{dataset}: no pcaps under {raw_prefix}, skipping")
            continue

        destination = lake.uri("extracted", dataset=dataset, source=source,
                               feature_space=feature_space)
        print(f"\n{dataset}  ({source})  {len(pcaps)} pcap(s) -> {destination}")

        report = _extract_dataset(lake, extractor, dataset, pcaps, destination)
        print(f"  {report.summary()}")
        if not report.complete:
            failed_any = True
            for capture, why in report.failed[:5]:
                print(f"    FAILED {capture}: {why.splitlines()[0][:120]}", file=sys.stderr)

    return 1 if failed_any else 0


def _extract_dataset(lake, extractor, dataset, pcaps, destination):
    """One pcap at a time: fetch, extract, upload, delete.

    Streaming rather than staging the whole dataset first. The previous shape
    downloaded every pcap for a dataset into /tmp before running anything, which
    for CIC-IDS-2018 is hundreds of gigabytes on the root volume -- the disk that
    has already taken the box down once.
    """
    from talos.data.extraction.base import ExtractionReport
    from talos.data.extraction.extractors.zeek import CHECKPOINT

    total = ExtractionReport()
    for uri in pcaps:
        capture = Path(uri).stem
        # Ask the LAKE, not local scratch. Each pcap gets a fresh temp dir, so a
        # local check could never fire -- and checking before the fetch is the
        # point: resuming must not re-download a capture it already has.
        if lake.exists(f"{destination}/{capture}/{CHECKPOINT}"):
            total.skipped.append(capture)
            continue

        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            local = lake.get(uri, scratch / "pcaps" / Path(uri).name)
            out = scratch / "logs"
            report = extractor.extract([str(local)], str(out))
            total.extracted += report.extracted
            total.failed += report.failed
            if report.extracted:
                lake.put_tree(out, destination)

    # The sidecar records whether the run completed, so anything later choosing
    # "the most recent extraction" can tell a clean run from a mostly-failed one.
    with tempfile.TemporaryDirectory() as tmp:
        extractor.write_metadata(tmp, dataset, report=total)
        lake.put_tree(Path(tmp), destination)
    return total


# ------------------------------------------------------------------- convert

def cmd_convert(args, extra) -> int:
    """Condenses format conversion into a single routing command."""
    import argparse
    p = argparse.ArgumentParser(prog="talos convert", description="Convert extracted logs.")
    p.add_argument("--dataset", required=True, help="Target dataset name")
    p.add_argument("--format", choices=["parquet", "csv", "both"], default="parquet", 
                   help="Output format (parquet, csv, both). Defaults to parquet.")
    p.add_argument("--feature-space", default=None, help="Explicit feature space")
    
    # Parse known arguments so downstream flags (e.g., --threads) pass through safely
    c_args, c_extra = p.parse_known_args(extra)
    
    # Build standard downstream arguments
    downstream_args = ["--dataset", c_args.dataset]
    if args.config:
        downstream_args.extend(["--config", args.config])
    if c_args.feature_space:
        downstream_args.extend(["--feature-space", c_args.feature_space])
    downstream_args.extend(c_extra)

    ret = 0
    if c_args.format in ("parquet", "both"):
        ret = run_stage("talos.data.conversion.to_parquet", downstream_args)
        if ret != 0: 
            return ret
            
    if c_args.format in ("csv", "both"):
        ret = run_stage("talos.data.conversion.to_csv", downstream_args)
        
    return ret

# --------------------------------------------------------------------- label

def cmd_label(args, extra) -> int:
    """Attach ground truth to one dataset's conn flows from its attack schedule."""
    import argparse
    from talos.common.validation import GateFailure
    from talos.data.labelling.engine import LabellingEngine

    p = argparse.ArgumentParser(prog="talos label")
    p.add_argument("--dataset", required=True, help="dataset name (must have a manifest)")
    p.add_argument("--feature-space", default=None,
                   help="extractor feature space (defaults to the configured extractor)")
    p.add_argument("--source", default=None, help="conn parquet glob to label instead")
    p.add_argument("--no-write", action="store_true",
                   help="report only; do not write the labelled table")
    a = p.parse_args(extra)

    cfg = Config.load(args.config)
    feature_space = a.feature_space or _feature_space(cfg)
    engine = LabellingEngine(cfg)

    print(f"dataset       {a.dataset}")
    print(f"feature space {feature_space}")
    try:
        report = engine.label(a.dataset, feature_space, a.source, write=not a.no_write)
    except GateFailure as exc:
        # The gate has already been written to the run report; surface it and stop.
        print(f"\nLABELLING ABORTED\n{exc}", file=sys.stderr)
        return 1

    print(f"manifest      {report.manifest_sha}   taxonomy {report.taxonomy_sha}")
    print(f"source        {report.source}\n")
    print(report.table())
    print(f"\n{report.gate.report()}")
    if report.overlaps:
        print(f"note: {report.overlaps:,} flow(s) matched >1 rule; lowest rule id wins")
    print(f"\noutput        {report.output or '(report only — --no-write)'}")
    return 0


def _feature_space(cfg) -> str:
    """The configured extractor's feature space, without running the extractor."""
    from talos.data.extraction import get_extractor
    return get_extractor(cfg.extractor, **cfg.doc.get(cfg.extractor, {})).feature_space


# --------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="talos", description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.yml")

    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    q = sub.add_parser("init", help="create a lake and a config pointing at it")
    q.add_argument("path", nargs="?", default="./lake", help="lake root (default: ./lake)")

    sub.add_parser("config", help="show where everything resolves to")
    sub.add_parser("extract", help="run the configured extractor over the raw zone")
    sub.add_parser("convert", help="convert extracted logs to parquet or csv", add_help=False)
    sub.add_parser("label", help="attach ground truth from the attack schedule", add_help=False)

    for name in ("discover", "eda", "compare", "render"):
        sub.add_parser(name, help=f"run the {name} stage", add_help=False)

    return p

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)

    try:
        if args.cmd == "init":
            return cmd_init(args)
        if args.cmd == "config":
            return cmd_config(args)
        if args.cmd == "extract":
            return cmd_extract(args, extra)
        if args.cmd == "convert":
            return cmd_convert(args, extra)
        if args.cmd == "label":
            return cmd_label(args, extra)
        if args.cmd in STAGES:
            if args.config:
                extra = ["--config", args.config, *extra]
            return run_stage(STAGES[args.cmd], extra)
            
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"unknown command {args.cmd!r}")
    return 2

if __name__ == "__main__":
    sys.exit(main())