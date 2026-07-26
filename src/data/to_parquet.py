#!/usr/bin/env python3
"""
Talos — mirror one dataset's EXTRACTED Zeek logs to Parquet, 1:1 per file.

Every <path>/<name>.log under --input is converted to <path>/<name>.parquet at
the SAME location under --output, so the extracted directory structure (dated /
per-pcap folders) is preserved exactly -- nothing is flattened or consolidated.
Streams JSON->Parquet per file (memory-safe), re-mints S3 creds periodically.

Zones:  extracted/<dataset>/ (zeek .log) -> parquets/<dataset>/ (parquet), same tree
Input layout:  <input>/**/<logtype>.log   (Zeek NDJSON, a single dataset)

Usage:
  python to_parquet.py --input s3://bkt/extracted/cic-ids-2017 --output s3://bkt/parquets/cic-ids-2017
  optional: --logtypes conn dns | --region eu-north-1 | --threads N
"""
from __future__ import annotations
import argparse, logging, os, shutil, subprocess, sys, time
from datetime import datetime

try:
    import duckdb
except ImportError:
    sys.exit("duckdb missing — run: pip install duckdb")

LOG = logging.getLogger("to_parquet")
TMP = "/tmp/talos_duckdb"

# self-scale to the machine: DuckDB gets 60% of RAM, half the CPUs (floor 4)
RAM_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**30
MEM_LIMIT = f"{max(4, int(RAM_GB * 0.6))}GB"
DEF_THREADS = max(4, (os.cpu_count() or 8) // 2)


def logger():
    lf = f"to_parquet_{datetime.now():%Y%m%d-%H%M%S}.log"
    LOG.setLevel(logging.INFO); LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(lf)):
        h.setFormatter(fmt); LOG.addHandler(h)
    return lf


def _aws_creds():
    d = {}
    try:
        out = subprocess.run(["aws", "configure", "export-credentials", "--format", "env"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("export "):
                k, _, v = line[7:].partition("="); d[k.strip()] = v.strip()
    except Exception:
        pass
    if "AWS_ACCESS_KEY_ID" not in d:
        for env, name in (("AWS_ACCESS_KEY_ID", "aws_access_key_id"),
                          ("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
                          ("AWS_SESSION_TOKEN", "aws_session_token")):
            v = subprocess.run(["aws", "configure", "get", name], capture_output=True, text=True).stdout.strip()
            if v: d[env] = v
    return d


def _tmp_guard(min_free_gb=20):
    free = shutil.disk_usage(TMP).free / 2**30
    if free < min_free_gb:
        sys.exit(f"ABORT: only {free:.0f} GiB free under {TMP} (need {min_free_gb}) — refusing to wedge the disk")


def _tmp_clean():
    for f in os.listdir(TMP):
        try:
            p = os.path.join(TMP, f)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        except OSError:
            pass


def connect(need_s3, region, threads):
    os.makedirs(TMP, exist_ok=True)
    c = duckdb.connect()
    for p in (f"SET threads={threads}", f"SET memory_limit='{MEM_LIMIT}'",
              f"SET temp_directory='{TMP}'", "SET max_temp_directory_size='60GB'",
              "SET preserve_insertion_order=false"):
        try: c.execute(p)
        except Exception: pass
    if need_s3:
        c.execute("INSTALL httpfs; LOAD httpfs;")
        if refresh_secret(c, region):
            LOG.info("using AWS credentials resolved from the AWS CLI")
        else:
            LOG.info("WARNING: could not resolve AWS credentials from the CLI")
        c.execute(f"SET s3_region='{region}';")
    return c


def refresh_secret(c, region):
    """Re-mint the S3 secret from the CLI's current (possibly rotated) session creds."""
    cr = _aws_creds()
    akid, secret, token = cr.get("AWS_ACCESS_KEY_ID"), cr.get("AWS_SECRET_ACCESS_KEY"), cr.get("AWS_SESSION_TOKEN")
    if not (akid and secret):
        return False
    extra = f", SESSION_TOKEN '{token}'" if token else ""
    c.execute(f"CREATE OR REPLACE SECRET s3sec (TYPE S3, KEY_ID '{akid}', SECRET '{secret}'{extra}, REGION '{region}');")
    return True


def discover(c, base, logtypes=None):
    """Return every .log file under base, as full paths, mirroring order.

    Optionally filter to given log-type stems (e.g. --logtypes conn dns).
    """
    rows = c.execute(f"SELECT file FROM glob('{base}/**/*.log') ORDER BY file").fetchall()
    files = [r[0] for r in rows]
    if logtypes:
        want = set(logtypes)
        files = [f for f in files if f.rsplit("/", 1)[-1].removesuffix(".log") in want]
    return files


def process_file(c, base, src, output):
    """Convert ONE .log file to ONE .parquet at the mirrored path under output."""
    rel = src[len(base) + 1:]                       # path of the log within the tree
    out = f"{output.rstrip('/')}/{rel[:-4]}.parquet"  # .log -> .parquet, same location
    if not out.startswith("s3://"):                 # local target: ensure parent dir
        os.makedirs(os.path.dirname(out), exist_ok=True)
    res = c.execute(
        f"COPY (SELECT * FROM read_json_auto('{src}', format='newline_delimited', "
        f"ignore_errors=true)) TO '{out}' (FORMAT PARQUET)"
    ).fetchone()
    return out, (res[0] if res else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="one dataset's extracted root")
    ap.add_argument("--output", required=True, help="parquet output root (tree is mirrored under it)")
    ap.add_argument("--logtypes", nargs="*", help="only these log-type stems (default: all)")
    ap.add_argument("--region", default="eu-north-1")
    ap.add_argument("--threads", type=int, default=DEF_THREADS)
    a = ap.parse_args()

    lf = logger()
    base = a.input.rstrip("/")
    s3 = a.input.startswith("s3://") or a.output.startswith("s3://")
    LOG.info(f"TALOS to_parquet (mirror)  input={base}  output={a.output}  log={lf}")
    LOG.info(f"box: {RAM_GB} GiB RAM -> memory_limit={MEM_LIMIT}, threads={a.threads}")
    c = connect(s3, a.region, a.threads)

    files = discover(c, base, a.logtypes)
    if not files:
        sys.exit(f"no .log files under {base}")
    LOG.info(f"mirroring {len(files):,} log files 1:1 -> parquet (same dir structure)")

    t0 = last_refresh = time.time()
    done = rows = skipped = 0
    for i, src in enumerate(files, 1):
        if s3 and time.time() - last_refresh > 600:   # re-mint creds every 10 min
            if not refresh_secret(c, a.region):
                sys.exit("ABORT: AWS CLI can no longer mint credentials (session expired?) — re-auth and rerun")
            last_refresh = time.time()
        _tmp_guard()
        try:
            _, n = process_file(c, base, src, a.output)
            done += 1; rows += n
        except Exception as e:
            skipped += 1; LOG.info(f"SKIP {src}: {e}")
        finally:
            _tmp_clean()
        if i % 100 == 0 or i == len(files):
            LOG.info(f"  {i:,}/{len(files):,}  ({done:,} ok, {skipped} skipped, {rows:,} rows)")
    LOG.info("=" * 80)
    LOG.info(f"DONE {done:,} files ({rows:,} rows, {skipped} skipped) in {time.time()-t0:.1f}s (log: {lf})")


if __name__ == "__main__":
    main()
