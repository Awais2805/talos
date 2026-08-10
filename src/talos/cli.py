"""Talos command line.

    talos init [path]     create a lake and a config pointing at it
    talos config          show where everything resolves to
    talos extract         run the configured extractor over the raw zone
    talos convert         extractor output -> parquet, 1:1
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
from talos.common.lake.lake import LakeClient
from talos.data.extraction import get_extractor


CONFIG_TEMPLATE = """\
# Talos — central configuration.
# Everything that decides WHERE data lives is here; no script carries its own
# bucket name or prefix. Point lake.root at a local directory or an s3:// URI
# and the whole pipeline follows.
lake:
  root: "{root}"
  zones:
    raw:       "raw/{{dataset}}"
    extracted: "extracted/{{dataset}}"
    parquets:  "parquets/{{dataset}}"
    labelled:  "labelled/{{dataset}}"
    mapped:    "mapped/{{dataset}}"

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
        cfg_path.write_text(CONFIG_TEMPLATE.format(root=root))
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
    "convert": "talos.data.to_parquet",
    "discover": "talos.preprocess.lake_feature_discovery",
    "eda": "talos.eda.profile_dataset",
    "compare": "talos.eda.compare",
    "render": "talos.eda.render",
}

def run_stage(module: str, extra: list[str]) -> int:
    argv = sys.argv
    sys.argv = [module.rsplit(".", 1)[-1], *extra]
    try:
        runpy.run_module(module, run_name="__main__")
        return 0
    except SystemExit as exc:                       # argparse / sys.exit from the stage
        return int(exc.code or 0)
    finally:
        sys.argv = argv

# ------------------------------------------------------------------- extract

def cmd_extract(args, extra) -> int:
    """Run the configured extractor over the raw zone to populate the extracted zone."""
    cfg = Config.load(args.config)
    
    # 1. Dynamically load the extractor based on config.yml
    extractor_name = cfg.extractor
    tool_cfg = cfg.doc.get(extractor_name, {}) # Pass tool-specific config (e.g., Zeek image)
    
    print(f"Using extractor plugin: {extractor_name}")
    try:
        extractor = get_extractor(extractor_name, **tool_cfg)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
        
    lake = LakeClient(root=cfg.root, region=cfg.region)
    
    # Allow scoping to specific datasets via extra args, or default to all configured datasets
    datasets = [d for d in cfg.datasets] if not extra else extra
    
    if not datasets:
        print("No datasets specified in config or arguments.", file=sys.stderr)
        return 1

    for dataset in datasets:
        print(f"\n=== Extracting dataset: {dataset} ===")
        
        # Resolve raw zone prefix
        raw_prefix = lake.uri("raw", dataset=dataset)
        raw_files = lake.list(raw_prefix)
        
        if not raw_files:
            print(f"No pcaps found in {raw_prefix}. Skipping.")
            continue
            
        # Determine the target extracted URI with the strict feature_space scope
        ext_uri = lake.uri("extracted", dataset=dataset, feature_space=extractor.feature_space)
        print(f"Target Feature Space: {extractor.feature_space}")
        print(f"Destination: {ext_uri}")
        
        # Use temp directory for isolated, ephemeral processing
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pcap_scratch = tmp_path / "pcaps"
            out_scratch = tmp_path / "logs"
            pcap_scratch.mkdir()
            out_scratch.mkdir()
            
            local_pcaps = []
            
            # Fetch PCAPs to local scratch space (Handles S3 vs Local automatically)
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
                print("No valid pcaps found after filtering. Skipping.")
                continue

            # 2. Execute agnostic extraction
            print(f"Starting extraction over {len(local_pcaps)} files...")
            try:
                extractor.extract(local_pcaps, str(out_scratch))
            except Exception as e:
                print(f"Extraction failed for {dataset}: {e}", file=sys.stderr)
                continue
                
            # 3. Write standardized metadata
            extractor.write_metadata(str(out_scratch), dataset)
            
            # 4. Upload extracted data + metadata to the exact feature_space zone
            print(f"Uploading extracted features to {ext_uri}...")
            if lake.remote:
                # Append slash to ensure contents of folder are uploaded, not the folder itself
                lake.backend.fs.put(str(out_scratch) + "/", ext_uri, recursive=True)
            else:
                shutil.copytree(out_scratch, Path(ext_uri), dirs_exist_ok=True)
                
            print(f"Completed {dataset}.")

    return 0

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

    for name in ("convert", "discover", "eda", "compare", "render"):
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