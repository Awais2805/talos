#!/usr/bin/env python3
"""
Talos — convert one dataset's EXTRACTED Zeek logs to Parquet (1:1 per log type),
with a profiling report. Streams JSON->Parquet (memory-safe at any size), then
profiles the written Parquet.

Zones:  extracted/<dataset>/ (zeek .log) -> parquets/<dataset>/ (parquet)
Input layout:  <input>/**/<logtype>.log   (Zeek NDJSON, a single dataset)

Usage:
  python to_parquet.py --input s3://bkt/extracted/cic-ids-2017 --output s3://bkt/parquets/cic-ids-2017
  optional: --logtypes conn dns | --top-k 15 | --region eu-north-1
  optional: --tag v2   -> writes <logtype>.v2.parquet (won't overwrite untagged files)

Provenance: every row gets source_path (log file relative to --input) and
capture (first path segment under --input, i.e. the pcap/split folder).
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


def discover(c, base):
    rows = c.execute(
        f"SELECT DISTINCT regexp_extract(file, '([^/]+)\\.log$', 1) lt FROM glob('{base}/**/*.log')"
    ).fetchall()
    return sorted(r[0] for r in rows if r[0])


def process(c, base, lt, output, top_k, tag=""):
    src = f"{base}/**/{lt}.log"
    fname = f"{lt}.{tag}.parquet" if tag else f"{lt}.parquet"
    out = f"{output.rstrip('/')}/{fname}"
    LOG.info("=" * 80); LOG.info(f"LOG TYPE: {lt}")
    # 1) STREAM convert JSON -> Parquet (pipelined; no in-memory table),
    #    keeping per-row provenance: source_path + capture (split folder)
    t0 = time.time()
    c.execute(
        f"COPY (SELECT * EXCLUDE (filename), "
        f"replace(filename, '{base}/', '') AS source_path, "
        f"split_part(replace(filename, '{base}/', ''), '/', 1) AS capture "
        f"FROM read_json_auto('{src}', format='newline_delimited', "
        f"union_by_name=true, ignore_errors=true, filename=true)) "
        f"TO '{out}' (FORMAT PARQUET)"
    )
    dt = time.time() - t0
    # 2) profile from the written Parquet (columnar; cheap)
    P = f"read_parquet('{out}')"
    n = c.execute(f"SELECT count(*) FROM {P}").fetchone()[0]
    LOG.info(f"  rows={n:,}   (streamed to parquet in {dt:.1f}s)")
    res = c.execute(f"SUMMARIZE SELECT * FROM {P}")
    heads = [d[0] for d in res.description]; ci = {h: i for i, h in enumerate(heads)}
    LOG.info("  " + " | ".join(heads))
    cats = []
    for r in res.fetchall():
        LOG.info("  " + " | ".join("" if v is None else str(v) for v in r))
        try:
            typ = str(r[ci["column_type"]]).upper()
            au = r[ci["approx_unique"]]; au = int(au) if au is not None else 10**9
            if typ.startswith(("VARCHAR", "BOOL")) and 0 < au <= 1000:
                cats.append(r[ci["column_name"]])
        except Exception:
            pass
    for cn in cats:  # value distribution for low-cardinality categoricals
        try:
            top = c.execute(
                f'SELECT CAST("{cn}" AS VARCHAR) v, count(*) k FROM {P} '
                f'WHERE "{cn}" IS NOT NULL GROUP BY 1 ORDER BY k DESC LIMIT {top_k}'
            ).fetchall()
            if top: LOG.info(f"  top {cn}: " + ", ".join(f"{v}={k:,}" for v, k in top))
        except Exception:
            pass
    LOG.info(f"  wrote {n:,} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="one dataset's extracted root")
    ap.add_argument("--output", required=True, help="parquet output prefix for this dataset")
    ap.add_argument("--logtypes", nargs="*")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--tag", default="",
                    help="output name suffix: <logtype>.<tag>.parquet (avoids overwriting)")
    ap.add_argument("--region", default="eu-north-1")
    ap.add_argument("--threads", type=int, default=DEF_THREADS)
    a = ap.parse_args()

    lf = logger()
    base = a.input.rstrip("/")
    LOG.info(f"TALOS to_parquet  input={base}  output={a.output}  log={lf}")
    LOG.info(f"box: {RAM_GB} GiB RAM -> memory_limit={MEM_LIMIT}, threads={a.threads}")
    c = connect(a.input.startswith("s3://") or a.output.startswith("s3://"), a.region, a.threads)
    lts = a.logtypes or discover(c, base)
    if not lts:
        sys.exit(f"no .log files under {base}")
    LOG.info(f"log types: {lts}")
    s3 = a.input.startswith("s3://") or a.output.startswith("s3://")
    t0 = time.time()
    for lt in lts:
        _tmp_guard()  # aborts the run outright if disk is low
        if s3 and not refresh_secret(c, a.region):
            sys.exit("ABORT: AWS CLI can no longer mint credentials (session expired?) — run 'aws login' / re-auth and rerun")
        try:
            process(c, base, lt, a.output, a.top_k, a.tag)
        except Exception as e:
            LOG.info(f"SKIP {lt}: {e}")
        finally:
            _tmp_clean()
    LOG.info("=" * 80); LOG.info(f"DONE {len(lts)} log types in {time.time()-t0:.1f}s (log: {lf})")


if __name__ == "__main__":
    main()
