#!/usr/bin/env python3
"""Shared DuckDB-over-S3 connection for anything that reads the lake.

Lifted out of label_flows.py so the EDA stage does not carry a third copy of
the credential dance. Behaviour is unchanged: a scan of the 2019 conn zone
outlives a one-hour STS session, so the S3 secret is re-minted from the AWS
CLI before every statement rather than once at connect time.
"""

import subprocess
import sys
import time

import duckdb

REFRESH_EVERY = 300          # seconds between re-mints; well inside any session TTL
EXT_REPO = "https://extensions.duckdb.org"


def cli_creds():
    """Current S3 credentials straight from the AWS CLI (handles rotation)."""
    try:
        out = subprocess.run(["aws", "configure", "export-credentials", "--format", "env"],
                             capture_output=True, text=True).stdout
        return dict(l[7:].split("=", 1) for l in out.splitlines() if l.startswith("export "))
    except Exception:
        return {}


class Lake:
    """DuckDB connection that keeps its S3 secret fresh across long scans."""

    def __init__(self, region, memory_limit=None, temp_dir=None, threads=None,
                 max_temp=None, require_s3=True):
        # require_s3=False is for local parquet: the same code path profiles a
        # downloaded extract or a test fixture without an AWS session existing.
        self.region, self._minted, self.require_s3 = region, 0.0, require_s3
        self.con = duckdb.connect()
        if require_s3:
            self.con.execute(f"INSTALL httpfs FROM '{EXT_REPO}'; LOAD httpfs;")
        # Guard rails for the 16 GB box, which has twice frozen on this lake --
        # once by filling the root disk with spill, once by thrashing itself to
        # death with 8 threads against a 12 GB limit. Neither failure produced an
        # OOM kill or a log line; both just hung SSH, so they are capped up front
        # rather than diagnosed again.
        if memory_limit:
            self.con.execute(f"SET memory_limit = '{memory_limit}'")
        if threads:
            self.con.execute(f"SET threads = {int(threads)}")
        if temp_dir:
            self.con.execute(f"SET temp_directory = '{temp_dir}'")
            self.con.execute(f"SET max_temp_directory_size = '{max_temp or '20GB'}'")
        if require_s3 and not self.refresh(force=True):
            sys.exit("no AWS credentials from the CLI — run 'aws login' and retry")

    def refresh(self, force=False):
        if not self.require_s3:
            return True
        if not force and time.time() - self._minted < REFRESH_EVERY:
            return True
        cr = cli_creds()
        akid, sec, tok = (cr.get("AWS_ACCESS_KEY_ID"), cr.get("AWS_SECRET_ACCESS_KEY"),
                          cr.get("AWS_SESSION_TOKEN"))
        if not (akid and sec):
            return False
        extra = f", SESSION_TOKEN '{tok}'" if tok else ""
        self.con.execute(f"CREATE OR REPLACE SECRET lake (TYPE S3, KEY_ID '{akid}', "
                         f"SECRET '{sec}'{extra}, REGION '{self.region}');")
        self._minted = time.time()
        return True

    def sql(self, query):
        self.refresh()
        return self.con.execute(query).fetchall()

    def one(self, query):
        self.refresh()
        return self.con.execute(query).fetchone()

    def columns(self, source):
        """Column name -> type for a parquet glob, from footers only."""
        self.refresh()
        rows = self.con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{source}', union_by_name = true)").fetchall()
        return {r[0]: r[1] for r in rows}
