#!/usr/bin/env python3
"""Talos command line.

    talos init [path]     create a lake and a config pointing at it
    talos config          show where everything resolves to
    talos extract         run the configured extractor over the raw zone
    talos convert         convert extracted logs to parquet or csv
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
import shutil
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
    root = args.path
    if zones.is_remote(root):
        print(f"{root} is a remote lake — nothing to create; object stores make "
              f"prefixes on first write.\nSet lake.root to this value in your config.")
        return 0

    base = Path(zones.normalise_root(root))
    print(f"lake  {base}")
    for zone, (path, created) in zones.init(root).items():
        mark = "+" if created else " "
        print(f"  {mark} {path.name + '/':<12} {zones.DESCRIPTIONS[zone]}")

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

    print(f"\nNext: put captures in {base / 'raw' / '<dataset>'}/ then `talos extract`")
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
    """Run the configured extractor over the raw zone to populate the extracted zone."""
    cfg = Config.load(args.config)
    extractor_name = cfg.extractor
    tool_cfg = cfg.doc.get(extractor_name, {}) 
    
    print(f"Using extractor plugin: {extractor_name}")
    try:
        extractor = get_extractor(extractor_name, **tool_cfg)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
        
    lake = cfg.lake()
    datasets = [d for d in cfg.datasets] if not extra else extra
    
    if not datasets:
        print("No datasets specified in config or arguments.", file=sys.stderr)
        return 1

    for dataset in datasets:
        print(f"\n=== Extracting dataset: {dataset} ===")
        raw_prefix = lake.uri("raw", dataset=dataset)
        raw_files = lake.list(raw_prefix)
        
        if not raw_files:
            print(f"No pcaps found in {raw_prefix}. Skipping.")
            continue
            
        ext_uri = lake.uri("extracted", dataset=dataset, feature_space=extractor.feature_space)
        print(f"Destination: {ext_uri}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pcap_scratch = tmp_path / "pcaps"
            out_scratch = tmp_path / "logs"
            pcap_scratch.mkdir()
            out_scratch.mkdir()
            
            local_pcaps = []
            for file_uri in raw_files:
                if not any(file_uri.endswith(ext) for ext in [".pcap", ".pcapng", ".cap"]):
                    continue
                local_dest = pcap_scratch / Path(file_uri).name
                print(f"Fetching {Path(file_uri).name}...")
                
                if lake.remote:
                    lake.backend.fs.get(file_uri, str(local_dest))
                else:
                    shutil.copy2(file_uri, local_dest)
                local_pcaps.append(str(local_dest))
            
            if not local_pcaps:
                continue

            try:
                extractor.extract(local_pcaps, str(out_scratch))
            except Exception as e:
                print(f"Extraction failed for {dataset}: {e}", file=sys.stderr)
                continue
                
            extractor.write_metadata(str(out_scratch), dataset)
            
            print(f"Uploading extracted features to {ext_uri}...")
            if lake.remote:
                lake.backend.fs.put(str(out_scratch) + "/", ext_uri, recursive=True)
            else:
                shutil.copytree(out_scratch, Path(ext_uri), dirs_exist_ok=True)
                
            print(f"Completed {dataset}.")

    return 0

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