"""Talos — Mirror extracted log files to Parquet 1:1 per file.

Converts extracted logs in `extracted/{feature_space}/{dataset}/` to 
`parquet/{feature_space}/{dataset}/` preserving directory structure.

Usage:
    python -m talos.data.conversion.to_parquet --dataset cic-ids-2017
    python -m talos.data.conversion.to_parquet --dataset cic-ids-2017 --feature-space zeek_v8.2.1
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from talos.common.config import Config
from talos.common.duck import DuckEngine
from talos.common.lake.lake import LakeClient

LOG = logging.getLogger("to_parquet")


def setup_logger():
    lf = f"to_parquet_{datetime.now():%Y%m%d-%H%M%S}.log"
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(lf)):
        h.setFormatter(fmt)
        LOG.addHandler(h)
    return lf


def resolve_feature_space(lake: LakeClient, dataset: str, explicit_fs: str = None) -> str:
    if explicit_fs:
        return explicit_fs

    extracted_base = lake.uri("extracted").rstrip("/")
    files = lake.list(extracted_base)
    latest_fs, latest_ts = None, ""

    for item in files:
        if item.endswith(f"/{dataset}/_extractor_meta.json"):
            try:
                content = (
                    lake.backend.fs.cat(item).decode("utf-8")
                    if lake.remote
                    else Path(item).read_text(encoding="utf-8")
                )
                meta = json.loads(content)
                ts = meta.get("extracted_at_utc", "")
                fs = meta.get("feature_space")
                if ts > latest_ts and fs:
                    latest_ts, latest_fs = ts, fs
            except Exception:
                continue

    if not latest_fs:
        raise ValueError(f"No extracted runs found for dataset '{dataset}'. Run `talos extract` first.")

    return latest_fs


def process_file(duck: DuckEngine, src: str, dest: str) -> int:
    """Convert ONE extracted log/json file to ONE parquet file."""
    if not dest.startswith("s3://"):
        os.makedirs(os.path.dirname(dest), exist_ok=True)

    query = (
        f"COPY (SELECT * FROM read_json_auto('{src}', format='newline_delimited', ignore_errors=true)) "
        f"TO '{dest}' (FORMAT PARQUET)"
    )
    duck.sql(query)
    count_res = duck.one(f"SELECT count(*) FROM read_parquet('{dest}')")
    return count_res[0] if count_res else 0


def main():
    ap = argparse.ArgumentParser(description="Convert freshly extracted logs to Parquet.")
    ap.add_argument("--config", default=None, help="Path to config.yml")
    ap.add_argument("--dataset", required=True, help="Target dataset name")
    ap.add_argument("--feature-space", default=None, help="Explicit feature space (defaults to most recently extracted)")
    ap.add_argument("--logtypes", nargs="*", help="Only convert specific log stems (e.g. conn dns)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory-limit", default="8GB")
    a = ap.parse_args()

    lf = setup_logger()
    cfg = Config.load(a.config)
    lake = LakeClient(root=cfg.root, region=cfg.region)
    duck = DuckEngine(
        remote=lake.remote,
        region=cfg.region,
        memory_limit=a.memory_limit,
        threads=a.threads,
    )

    feature_space = resolve_feature_space(lake, a.dataset, a.feature_space)
    src_uri = lake.uri("extracted", dataset=a.dataset, feature_space=feature_space)
    dst_uri = lake.uri("parquet", dataset=a.dataset, feature_space=feature_space)

    LOG.info(f"Converting Extracted -> Parquet")
    LOG.info(f"Dataset:       {a.dataset}")
    LOG.info(f"Feature Space: {feature_space}")
    LOG.info(f"Source:        {src_uri}")
    LOG.info(f"Destination:   {dst_uri}")

    files = lake.list(src_uri)
    log_files = [f for f in files if f.endswith(".log") or f.endswith(".json")]

    if a.logtypes:
        want = set(a.logtypes)
        log_files = [f for f in log_files if Path(f).stem in want]

    if not log_files:
        LOG.warning(f"No extracted log files found under {src_uri}")
        return

    LOG.info(f"Found {len(log_files):,} log file(s) to convert.")
    t0 = time.time()
    done = rows = skipped = 0

    for i, src in enumerate(log_files, 1):
        rel_path = src[len(src_uri) :].lstrip("/")
        stem_rel = str(Path(rel_path).with_suffix(".parquet"))
        dest = f"{dst_uri.rstrip('/')}/{stem_rel}"

        try:
            n = process_file(duck, src, dest)
            done += 1
            rows += n
        except Exception as e:
            skipped += 1
            LOG.warning(f"SKIP {src}: {e}")

        if i % 100 == 0 or i == len(log_files):
            LOG.info(f"  {i:,}/{len(log_files):,} files processed ({done:,} ok, {skipped} skipped, {rows:,} rows)")

    LOG.info(f"DONE: {done:,} files converted ({rows:,} rows, {skipped} skipped) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()