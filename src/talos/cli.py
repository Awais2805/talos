#!/usr/bin/env python3
"""Talos command line.

    talos init [path]     create a lake and a config pointing at it
    talos config          show where everything resolves to
    talos ingest          put captures into a source's raw zone
    talos extract         run the configured extractor over the raw zone
    talos convert         convert extracted logs to parquet or csv
    talos label           label flows (--method schedule|ae-v1|...)
    talos audit           emit audit candidates, or score methods against them
    talos pools           show how a partition splits a lake
    talos oracle          independent evidence from Suricata (corroborate|offset)
    talos discover        profile the lake by log type
    talos eda             profile one dataset -> reports
    talos compare         rebuild comparisons from existing profiles
    talos render          rebuild HTML from existing JSON

Stage commands forward unrecognised flags to the underlying module, so anything
the module accepts still works: `talos eda --dataset X --threads 8`.
"""

import argparse
import logging
import runpy
import sys
import tempfile
from pathlib import Path

from talos.common import zones
from talos.common.config import Config, ConfigError
from talos.common.provenance import ProvenanceService
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

# Bring your own, keyed by plug-in point. Yours is searched before the shipped
# one, and every run reports the origin of what it resolved.
#
# plugins:
#   extractor: ["~/talos-plugins/nprobe.py"]
#   manifest:  ["~/talos-plugins/manifests"]

# Only consulted when lake.root is an s3:// URI.
aws:
  region: eu-north-1

# What is in the lake, and where each dataset's traffic came from. `source`
# decides which labelling is applicable, not merely where files sit:
#   datasets  -- public corpora, labelled from a published attack schedule
#   netem     -- emulator runs, labels known by construction
#   honeypot  -- live capture, no schedule; behavioural labelling only
#
# ROLES ARE NOT HERE. Whether a dataset may contribute to a training pool is a
# property of an experiment, so it lives in experiments/<name>/experiment.yaml.
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

# -------------------------------------------------------------------- ingest

def cmd_ingest(args, extra) -> int:
    """Put captures into a source's raw zone, with provenance, and seal it."""
    from talos.data.ingestion.ingest import IngestionError, main as ingest_main

    argv = list(extra)
    if args.config:
        argv = ["--config", args.config, *argv]
    try:
        return ingest_main(argv)
    except IngestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


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

    # The sidecar records whether the TREE is complete, so anything later
    # choosing "the most recent extraction" can tell a clean run from a
    # mostly-failed one. The existing sidecar is read first: without it the
    # merge has nothing to carry forward and a second pass over a subset would
    # report `complete` while earlier failures were still missing.
    import json
    from talos.data.extraction.base import META_FILENAME

    previous = None
    marker = f"{destination}/{META_FILENAME}"
    if lake.exists(marker):
        try:
            previous = json.loads(lake.read_text(marker))
        except (ValueError, OSError):
            previous = None          # unreadable sidecar: start fresh, do not crash

    with tempfile.TemporaryDirectory() as tmp:
        extractor.write_metadata(tmp, dataset, report=total, previous=previous)
        lake.put_tree(Path(tmp), destination)
    return total


# ------------------------------------------------------------------- convert

def cmd_convert(args, extra) -> int:
    """Mirror the extracted zone into parquet, and optionally CSV, in one pass."""
    import argparse
    from talos.data.conversion.convert import Converter, ConversionError

    p = argparse.ArgumentParser(prog="talos convert")
    p.add_argument("--dataset", required=True)
    p.add_argument("--format", choices=["parquet", "csv", "both"], default="parquet")
    p.add_argument("--source", default=None,
                   help="defaults to the dataset's declaration in config.yml")
    p.add_argument("--feature-space", default=None,
                   help="defaults to the most recent COMPLETE extraction")
    p.add_argument("--logtypes", nargs="*",
                   help="only convert these log stems, e.g. conn dns")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="convert an extraction that reported failed captures")
    a = p.parse_args(extra)

    formats = ("parquet", "csv") if a.format == "both" else (a.format,)
    cfg = Config.load(args.config)
    try:
        report = Converter(cfg).convert(
            a.dataset, formats=formats, source=a.source,
            feature_space=a.feature_space, logtypes=a.logtypes,
            allow_incomplete=a.allow_incomplete)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"dataset       {report.dataset}  ({report.source})")
    print(f"feature space {report.feature_space}")
    print(f"source        {report.src}")
    for fmt, uri in report.destinations.items():
        print(f"  -> {fmt:<8} {uri}")
    print(f"\n{report.summary()}")
    for rel, why in report.skipped[:5]:
        print(f"  SKIPPED {rel}: {why}", file=sys.stderr)
    return 1 if report.skipped else 0


# --------------------------------------------------------------------- label

def cmd_label(args, extra) -> int:
    """Label one dataset's conn flows, by whichever method is asked for."""
    import argparse
    from talos.common.validation import GateFailure
    from talos.data.labelling.base import LabellingError
    from talos.data.labelling.method import METHODS, MethodLoader
    import talos.points                             # noqa: F401 -- registers methods

    p = argparse.ArgumentParser(prog="talos label")
    p.add_argument("--dataset", required=True, help="dataset name (must have a manifest)")
    p.add_argument("--method", default="schedule",
                   help=f"labelling method: {', '.join(METHODS.names())}")
    p.add_argument("--method-file", default=None,
                   help="a method.yaml outside the package, e.g. your own ae-v2")
    p.add_argument("--pools", default=None,
                   help="override the method's declared partition (behavioural methods only)")
    p.add_argument("--feature-space", default=None,
                   help="extractor feature space (defaults to the configured extractor)")
    p.add_argument("--source", default=None, help="conn parquet glob to label instead")
    p.add_argument("--no-write", action="store_true",
                   help="report only; do not write the labelled table")
    p.add_argument("--force", action="store_true",
                   help="overwrite a table built from different inputs")
    p.add_argument("--allow-untrained", action="store_true",
                   help="write a table from parts that cannot learn (harness only)")
    p.add_argument("--skip-audit-gate", action="store_true",
                   help="fine-tune on D_s's raw schedule label, skipping the "
                        "human-audit requirement (already-verified labels only)")
    a = p.parse_args(extra)

    cfg = Config.load(args.config)
    feature_space = a.feature_space or _feature_space(cfg)
    try:
        # The DECLARATION is resolved first; it names the implementation. A
        # `--method-file` is just a declaration resolved from somewhere else.
        spec = MethodLoader().load(a.method_file or a.method)
        build_kwargs = {"allow_untrained": a.allow_untrained,
                        "skip_audit_gate": a.skip_audit_gate}
        if a.pools:
            build_kwargs["pools"] = a.pools
        method = spec.build(cfg, **build_kwargs)
    except LabellingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"dataset       {a.dataset}")
    print(f"method        {spec.name}  ({spec.implementation}){_whose(spec.origin)}")
    print(f"label space   {spec.label_space.name}  sha {spec.label_space.sha}  "
          f"({spec.label_space.n_classes} classes, "
          f"{len(spec.label_space.excluded)} excluded)")
    print(f"feature space {feature_space}")
    try:
        report = method.label(a.dataset, feature_space, source=a.source,
                              write=not a.no_write, force=a.force)
    except LabellingError as exc:
        # Nothing to label, or the wrong shape of table. Refused before any scan.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except GateFailure as exc:
        # The gate has already been written to the run report; surface it and stop.
        print(f"\nLABELLING ABORTED\n{exc}", file=sys.stderr)
        return 1

    # Only what every method has. Whatever is particular to one of them is its
    # own report's business -- which is the core-schema argument, for output.
    print(f"method sha    {report.method_sha}")
    print(f"source        {report.source}\n")
    print(report.table())
    print(f"\n{report.summary()}")
    print(f"\noutput        {report.output or '(report only — --no-write)'}")
    return 0


# ---------------------------------------------------------------------- audit

def cmd_audit(args, extra) -> int:
    """Emit candidates for a person, read their decisions back, or score methods."""
    import argparse
    from talos.data.labelling.audit import (
        AdjudicationTable, AuditError, AuditPage, CandidateSelector, LabelBenchmark,
        default_path,
    )
    from talos.data.labelling.behavioural.pool import PartitionLoader
    from talos.data.labelling.method import MethodLoader

    p = argparse.ArgumentParser(prog="talos audit")
    p.add_argument("action", choices=("emit", "render", "status", "benchmark"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--pools", default="xdg-v3", help="partition declaring the audit pool")
    p.add_argument("--method", default="schedule", help="method whose labels are the prior")
    p.add_argument("--space", default="core-5")
    p.add_argument("--feature-space", default=None)
    p.add_argument("--out", default=None, help="audit file (default reports/audit/…)")
    p.add_argument("--floor", type=int, default=200, help="rows per class")
    p.add_argument("--cap", type=int, default=2000, help="total rows a person will read")
    p.add_argument("--compare", nargs="*", default=[], help="methods to score")
    p.add_argument("--alerts", default=None,
                   help="a Suricata eve.json, to put independent evidence beside "
                        "each candidate")
    p.add_argument("--tolerance", type=float, default=2.0,
                   help="seconds of slack when joining an alert to a flow")
    a = p.parse_args(extra)

    cfg = Config.load(args.config)
    feature_space = a.feature_space or _feature_space(cfg)
    lake, duck = cfg.lake(), cfg.lake().duck
    space = MethodLoader().load(a.method).label_space
    partition = PartitionLoader().load(a.pools)
    base = Path(a.out) if a.out else default_path(cfg, a.dataset)
    table = AdjudicationTable(space, prior_method=a.method)

    _print_origins(a, partition)
    try:
        if a.action == "emit":
            sources = partition.sources(lake, feature_space, cfg)
            chosen, selection = CandidateSelector(
                partition, floor=a.floor, cap=a.cap).select(duck, sources)
            alerts = _oracle_join(duck, cfg, lake, a, feature_space)
            # Both formats, every time: a person adjudicates the CSV, and the
            # parquet keeps the types the CSV round-trip would flatten.
            written = [table.emit(duck, chosen, base.with_suffix(suffix), alerts)
                       for suffix in (".csv", ".parquet")]
            print(selection.describe())
            print(f"\npool          {selection.pool}")
            page = AuditPage(space, a.dataset, a.method).render(
                duck, base.with_suffix(".csv"), base.with_suffix(".html"))
            print(f"oracle        {'suricata ' + a.alerts if alerts else 'not consulted'}")
            for path in written + [page]:
                print(f"written       {path}")
            print(f"\nOpen {page} to adjudicate, then feed the downloaded CSV back "
                  f"to `talos audit status`.")
            return 0

        if a.action == "render":
            page = AuditPage(space, a.dataset, a.method).render(
                duck, base, base.with_suffix(".html"))
            print(f"written       {page}")
            return 0

        adjudication = table.read(duck, base)
        if a.action == "status":
            print(table.summarise(duck, adjudication, base).describe())
            return 0

        tables = {name: lake.uri("labelled", dataset=a.dataset, method=name,
                                 feature_space=feature_space,
                                 source=cfg.source_of(a.dataset), rel="conn.parquet")
                  for name in (a.compare or [a.method])}
        report = LabelBenchmark(space).run(duck, adjudication, tables, name=str(base))
        print(report.describe())
        ProvenanceService(cfg.reports).run_report(f"benchmark_{a.dataset}",
                                                  report.to_dict())
        return 0
    except (AuditError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _oracle_join(duck, cfg, lake, a, feature_space) -> str | None:
    """Independent evidence beside each candidate, or None if none was given.

    Reuses the oracle's own join rather than writing a second one, so the
    signature an adjudicator reads is the same event the corroboration rate
    counted.
    """
    if not a.alerts:
        return None
    from talos.data.labelling.oracle import Corroborator, SuricataOracle

    labelled = lake.uri("labelled", dataset=a.dataset, method=a.method,
                        feature_space=feature_space,
                        source=cfg.source_of(a.dataset), rel="conn.parquet")
    relation = SuricataOracle().alerts_relation(duck, a.alerts)
    joined = Corroborator(tolerance=a.tolerance).join_sql(labelled, relation.sql_query())
    # Materialised, not nested: the join carries CTEs that do not survive being
    # used as a derived table, and this way it is evaluated once rather than
    # re-run for each of the two output formats.
    duck.sql(f"CREATE OR REPLACE TEMP TABLE oracle_hits AS {joined}")
    return "SELECT uid, alerted, signature FROM oracle_hits"


def _print_origins(a, partition) -> None:
    """Whose declarations resolved. A run on the user's own must not read
    identically to a run on ours -- D7.4, made visible where someone looks."""
    from talos.data.labelling.behavioural.pool import PARTITIONS
    from talos.data.labelling.method import METHODS

    print(f"dataset       {a.dataset}")
    print(f"method        {a.method}{_whose(METHODS.origin_of(a.method))}")
    print(f"pools         {partition.name} {partition.sha}"
          f"{_whose(PARTITIONS.origin_of(a.pools))}")


# --------------------------------------------------------------------- oracle

def cmd_oracle(args, extra) -> int:
    """Independent evidence from Suricata, aggregated rather than joined per row.

    `corroborate` prints the two rates `_oracle_join` computes but never
    surfaces on its own (D6.5: never combined into one number). `offset`
    prints an `OffsetProbe` verdict on which UTC offset the capture's clocks
    actually used -- the one thing no document can settle (W2.3).
    """
    import argparse
    from talos.data.labelling.oracle import Corroborator, OffsetProbe, SuricataError, SuricataOracle
    from talos.data.labelling.schedule.manifest import MANIFESTS, ManifestError, ManifestLoader

    p = argparse.ArgumentParser(prog="talos oracle")
    p.add_argument("action", choices=("corroborate", "offset"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--alerts", required=True, help="a Suricata eve.json")
    p.add_argument("--method", default="schedule",
                   help="labelled table to corroborate against (corroborate only)")
    p.add_argument("--feature-space", default=None)
    p.add_argument("--tolerance", type=float, default=2.0,
                   help="seconds of slack joining an alert to a flow (corroborate only)")
    p.add_argument("--declared", default=None,
                   help="the manifest's own utc_offset (offset only -- see "
                        "manifests/<dataset>.yaml; there is no single value for a "
                        "dataset whose days: table carries different offsets per day). "
                        "Pass as --declared=-03:00 (the '=' is required -- argparse "
                        "reads a space-separated '-03:00' as another flag)")
    a = p.parse_args(extra)

    # Checked here, not left to DuckDB: a missing eve.json otherwise surfaces as
    # an uncaught duckdb.IOException, which every other error in this command
    # is deliberately turned into a clean `error: ...` line instead.
    if not Path(a.alerts).exists():
        print(f"error: no such alerts file: {a.alerts}", file=sys.stderr)
        return 2

    cfg = Config.load(args.config)
    feature_space = a.feature_space or _feature_space(cfg)
    lake, duck = cfg.lake(), cfg.lake().duck
    oracle = SuricataOracle()

    try:
        if a.action == "corroborate":
            labelled_uri = lake.uri("labelled", dataset=a.dataset, method=a.method,
                                    feature_space=feature_space,
                                    source=cfg.source_of(a.dataset), rel="conn.parquet")
            relation = oracle.alerts_relation(duck, a.alerts)
            alerts_total = duck.one(f"SELECT count(*) FROM ({relation.sql_query()}) t")[0]
            report = Corroborator(tolerance=a.tolerance).build(
                duck, a.dataset, labelled_uri, relation.sql_query(), alerts_total)

            print(f"dataset       {a.dataset}")
            print(f"method        {a.method}")
            print(f"alerts        {a.alerts}\n")
            print(report.summary())
            path = ProvenanceService(cfg.reports).run_report(
                f"oracle_corroborate_{a.dataset}_{a.method}", report.to_dict())
            print(f"\nwritten       {path}")
            return 0

        # action == "offset"
        if not a.declared:
            p.error("--declared is required for `talos oracle offset` -- "
                    "the manifest's own utc_offset, read off manifests/<dataset>.yaml")
        manifest = ManifestLoader().load(a.dataset)
        alerts = oracle.parse_alerts(a.alerts)
        verdict = OffsetProbe().probe(alerts, manifest.rules, a.declared)

        print(f"dataset       {a.dataset}")
        print(f"manifest      {manifest.path}{_whose(MANIFESTS.origin_of(a.dataset))}")
        print(f"alerts        {a.alerts}\n")
        print(verdict.summary())
        path = ProvenanceService(cfg.reports).run_report(
            f"oracle_offset_{a.dataset}", verdict.to_dict())
        print(f"\nwritten       {path}")
        return 0
    except (SuricataError, ManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cmd_pools(args, extra) -> int:
    """Show how a partition splits a lake, before 137M rows are split by it."""
    import argparse
    from talos.data.labelling.behavioural.pool import PARTITIONS, PartitionLoader, PoolError

    p = argparse.ArgumentParser(prog="talos pools")
    p.add_argument("--pools", default="xdg-v3", help="partition declaration")
    p.add_argument("--total", type=int, default=0,
                   help="row count to project expected pool sizes against")
    a = p.parse_args(extra)

    Config.load(args.config)                    # installs plug-ins from config.yml
    try:
        partition = PartitionLoader().load(a.pools)
    except PoolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(partition.describe())
    print(f"  origin      {PARTITIONS.origin_of(a.pools).describe()}")
    if a.total:
        print(f"\n{'pool':<10}{'expected rows':>16}")
        for pool in partition:
            print(f"{pool.name:<10}{pool.expected_flows(a.total):>16,}")
    return 0


def _whose(origin) -> str:
    """Nothing for a built-in, a loud suffix for anything the user substituted.

    A run that used the user's own manifest or method must not read identically
    to one that used ours -- which is the entire reason `Origin` exists.
    """
    return "" if origin.built_in else f"   [{origin.where} {origin.source}]"


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
    sub.add_parser("ingest", help="put captures into a source's raw zone",
                   add_help=False)
    sub.add_parser("extract", help="run the configured extractor over the raw zone")
    sub.add_parser("convert", help="convert extracted logs to parquet or csv", add_help=False)
    sub.add_parser("label", help="label flows by the chosen method", add_help=False)
    sub.add_parser("audit", help="emit audit candidates, or score methods against them",
                   add_help=False)
    sub.add_parser("pools", help="show how a partition splits a lake", add_help=False)
    sub.add_parser("oracle", help="independent evidence from Suricata: corroboration "
                   "rates, or a verdict on which UTC offset a capture used",
                   add_help=False)

    for name in ("discover", "eda", "compare", "render"):
        sub.add_parser(name, help=f"run the {name} stage", add_help=False)

    return p

def main(argv=None) -> int:
    # Every stage's `logger.info(...)` (extraction progress, training epochs)
    # was silently discarded before this -- nothing ever configured a handler,
    # so INFO-level messages had nowhere to go regardless of which subcommand
    # ran. One call here covers all of them.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)

    try:
        if args.cmd == "init":
            return cmd_init(args)
        if args.cmd == "config":
            return cmd_config(args)
        if args.cmd == "ingest":
            return cmd_ingest(args, extra)
        if args.cmd == "extract":
            return cmd_extract(args, extra)
        if args.cmd == "audit":
            return cmd_audit(args, extra)
        if args.cmd == "pools":
            return cmd_pools(args, extra)
        if args.cmd == "oracle":
            return cmd_oracle(args, extra)
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