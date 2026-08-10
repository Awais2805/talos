"""In-process SQL over lake parquet.

DuckDB reads a local directory and an object store through the same syntax, so
nothing above this layer needs to know which it is talking to. A local lake is
the default and costs nothing to open: no extension is loaded, no credential is
resolved, no network call is made.

Remote support is an attachment. When -- and only when -- a URI names a remote
backend, the httpfs extension is loaded and a secret is minted. Everything in
`_remote` exists to serve that case and is inert otherwise.

The one behaviour worth knowing: a scan of a large zone can outlive a one-hour
credential session, so the secret is re-minted periodically rather than once at
connect time.
"""

import subprocess
import time

import duckdb

REFRESH_EVERY = 300          # seconds between re-mints; well inside any session TTL


def _remote_credentials():
    """Credentials from the AWS CLI, if one is installed and configured.

    Only reached for remote backends. Returns {} when the CLI is absent, which
    is the normal state on a machine using a local lake.
    """
    try:
        out = subprocess.run(["aws", "configure", "export-credentials", "--format", "env"],
                             capture_output=True, text=True).stdout
        return dict(l[7:].split("=", 1) for l in out.splitlines() if l.startswith("export "))
    except Exception:
        return {}


class DuckEngine:
    """A DuckDB session scoped to one lake.

    `remote` decides whether the object-store attachment is engaged at all. It
    is derived from the lake root or the source URI, never assumed.
    """

    def __init__(self, remote=False, region=None, memory_limit=None, temp_dir=None,
                 threads=None, max_temp=None):
        self.remote, self.region, self._minted = bool(remote), region, 0.0
        self.con = duckdb.connect()

        if self.remote:
            # LOAD first, INSTALL only if genuinely missing. Naming a repository
            # in INSTALL fails outright when the machine already has the
            # extension from a different origin (http:// vs https://).
            try:
                self.con.execute("LOAD httpfs;")
            except duckdb.Error:
                self.con.execute("INSTALL httpfs; LOAD httpfs;")

        # Resource guard rails. A long scan has twice wedged a 16 GB box -- once
        # by filling the root disk with spill, once by thrashing itself to death
        # with too many threads against too small a memory limit. Neither
        # produced an OOM kill or a log line; both just hung the machine, so
        # they are capped up front rather than diagnosed again.
        if memory_limit:
            self.con.execute(f"SET memory_limit = '{memory_limit}'")
        if threads:
            self.con.execute(f"SET threads = {int(threads)}")
        if temp_dir:
            self.con.execute(f"SET temp_directory = '{temp_dir}'")
            self.con.execute(f"SET max_temp_directory_size = '{max_temp or '20GB'}'")

        if self.remote and not self.refresh(force=True):
            raise RuntimeError(
                "this lake is remote but no credentials could be resolved.\n"
                "Either authenticate, or point lake.root at a local directory.")

    # ------------------------------------------------------------ credentials

    def refresh(self, force=False):
        """Re-mint the secret for a remote backend. A no-op for a local lake."""
        if not self.remote:
            return True
        if not force and time.time() - self._minted < REFRESH_EVERY:
            return True
        cr = _remote_credentials()
        key, secret, token = (cr.get("AWS_ACCESS_KEY_ID"), cr.get("AWS_SECRET_ACCESS_KEY"),
                              cr.get("AWS_SESSION_TOKEN"))
        if not (key and secret):
            return False
        extra = f", SESSION_TOKEN '{token}'" if token else ""
        self.con.execute(f"CREATE OR REPLACE SECRET lake (TYPE S3, KEY_ID '{key}', "
                         f"SECRET '{secret}'{extra}, REGION '{self.region}');")
        self._minted = time.time()
        return True

    # ------------------------------------------------------------------- sql

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
