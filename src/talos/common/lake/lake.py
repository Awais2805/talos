"""The only class that touches storage.

A lake is a set of zones under a root. The root is a plain directory by default;
it can be an object-store prefix instead, and nothing above this module changes
when it is. Callers address data logically -- zone, feature space, dataset,
relative path -- and `LakeClient` resolves that to a physical location.

Two backends:

  LocalBackend   a directory. The default. No third-party dependency.
  RemoteBackend  an object store, via fsspec. Requires the `talos[s3]` extra
                 and is imported only when a remote root is actually used.

The base install has no cloud dependency at all. Asking for a remote lake
without the extra installed raises with the command needed to fix it, rather
than failing somewhere deeper with an import error.

Write-once is an obligation, not a property. On an object store it came free;
on a filesystem it does not, so `seal()` makes the raw zone read-only after an
ingest. Provenance downstream assumes immutability, so a backend that cannot
enforce it is not a conforming backend.
"""

from __future__ import annotations   # this class defines `list`, which would otherwise
                                     # shadow the builtin inside its own annotations

import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path

from talos.common import zones


class LakeError(Exception):
    pass


# Written into every zone prefix by `init`. On a filesystem a directory can be
# empty; on an object store it cannot exist at all without something in it.
ZONE_MARKER = ".talos-zone"

# The lake's identity, at its root. This is what makes a lake discoverable --
# commands walk up for it the way git walks up for .git -- and what tells you a
# bucket is a Talos lake rather than somebody's backups.
LAKE_FILE = "lake.yaml"


# ------------------------------------------------------------------ backends

class LocalBackend:
    """A lake that is a directory. The default, and the one with no dependencies."""

    remote = False

    def exists(self, uri: str) -> bool:
        return Path(uri).exists()

    def put(self, local: Path, uri: str) -> str:
        dst = Path(uri)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, dst)
        return uri

    def list(self, prefix: str) -> list[str]:
        base = Path(prefix)
        if not base.exists():
            return []
        return sorted(str(p) for p in base.rglob("*") if p.is_file())

    def mkdir(self, uri: str) -> bool:
        """Make a zone location exist. True if it did not already."""
        path = Path(uri)
        created = not path.exists()
        path.mkdir(parents=True, exist_ok=True)
        return created

    def write_text(self, uri: str, text: str) -> str:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return uri

    def read_text(self, uri: str) -> str:
        return Path(uri).read_text(encoding="utf-8")

    def get(self, uri: str, local: Path) -> Path:
        """Fetch one object to a local path."""
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(uri, local)
        return local

    def put_tree(self, local: Path, uri: str) -> str:
        """Upload a whole directory, preserving its shape."""
        shutil.copytree(Path(local), Path(uri), dirs_exist_ok=True)
        return uri

    def seal(self, uri: str) -> None:
        """Make a zone read-only."""
        base = Path(uri)
        if not base.exists():
            return
        for path in [base, *base.rglob("*")]:
            mode = path.stat().st_mode
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    def unseal(self, uri: str) -> None:
        # Guarded like `seal`: unsealing a zone that does not exist yet is what
        # the first ingest into a dataset does, and it must be a no-op rather
        # than a crash.
        base = Path(uri)
        if not base.exists():
            return
        for path in [base, *base.rglob("*")]:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


class RemoteBackend:
    """An object store, via fsspec. Only constructed for a remote root."""

    remote = True

    def __init__(self, root: str, region: str | None = None):
        try:
            import fsspec
        except ImportError as exc:
            raise LakeError(
                f"lake.root is {root}, which needs the remote backend.\n"
                f"Install it with:  pip install -e '.[s3]'") from exc
        self.protocol = root.split("://", 1)[0]
        self.fs = fsspec.filesystem(self.protocol,
                                    client_kwargs={"region_name": region} if region else None)

    def exists(self, uri: str) -> bool:
        return self.fs.exists(uri)

    def put(self, local: Path, uri: str) -> str:
        self.fs.put(str(local), uri)
        return uri

    def list(self, prefix: str) -> list[str]:
        try:
            found = self.fs.find(prefix)
        except FileNotFoundError:
            return []
        # fsspec answers in bucket/key form. Callers slice these against the
        # scheme-carrying URI they asked for and hand them to DuckDB, both of
        # which are silently wrong without the scheme, so it goes back on here.
        return sorted(f if "://" in f else f"{self.protocol}://{f.lstrip('/')}"
                      for f in found)

    def mkdir(self, uri: str) -> bool:
        """Make a zone prefix exist, by putting a marker object in it.

        An object store has no directories — a prefix springs into existence
        when something is written under it and vanishes when the last object
        goes. That is fine for machines and useless for a person, who opens the
        console after `talos init` and wants to see where their pcaps go. A
        zero-byte marker makes the zone real, listable and visible, and costs
        nothing.
        """
        marker = f"{uri.rstrip('/')}/{ZONE_MARKER}"
        if self.fs.exists(marker):
            return False
        with self.fs.open(marker, "wb") as fh:
            fh.write(b"")
        return True

    def write_text(self, uri: str, text: str) -> str:
        with self.fs.open(uri, "wb") as fh:
            fh.write(text.encode("utf-8"))
        return uri

    def read_text(self, uri: str) -> str:
        return self.fs.cat(uri).decode("utf-8")

    def get(self, uri: str, local: Path) -> Path:
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        self.fs.get(uri, str(local))
        return local

    def put_tree(self, local: Path, uri: str) -> str:
        # The trailing slash is load-bearing: without it fsspec puts the
        # directory *inside* the destination instead of merging into it.
        self.fs.put(str(Path(local)) + "/", uri, recursive=True)
        return uri

    def seal(self, uri: str) -> None:
        """No-op: object stores are already write-once by convention here."""

    def unseal(self, uri: str) -> None:
        """No-op."""


def backend_for(root: str, region: str | None = None):
    return RemoteBackend(root, region) if zones.is_remote(root) else LocalBackend()


# ---------------------------------------------------------------------- lake

class LakeClient:
    """Resolves logical zone addresses against a backend."""

    def __init__(self, root: str, templates: dict[str, str] | None = None,
                 region: str | None = None, backend=None, engine=None,
                 default_source: str = zones.DEFAULT_SOURCE, profile=None):
        self.root = zones.normalise_root(root)
        self.templates = {**zones.DEFAULT_TEMPLATES, **(templates or {})}
        self.region = region
        self.default_source = default_source
        self.profile = profile
        self.backend = backend or backend_for(self.root, region)
        self._engine = engine

    @property
    def remote(self) -> bool:
        return self.backend.remote

    @property
    def duck(self):
        """The SQL engine, built on first use.

        Lazy because constructing one against a remote lake resolves
        credentials and raises without them: a local lake, and any command that
        only lists or hashes, must not pay that. Injectable because a stage that
        has already sized its own memory limit and spill directory should not
        get a second, unbounded connection behind its back.
        """
        if self._engine is None:
            from talos.common.duck import DuckEngine
            self._engine = DuckEngine(remote=self.remote, region=self.region,
                                      profile=self.profile)
        return self._engine

    def uri(self, zone: str, dataset: str | None = None, rel: str | None = None,
            feature_space: str | None = None, schema_version: str | None = None,
            source: str | None = None, method: str | None = None) -> str:
        """Logical address -> physical location.

        `feature_space` scopes the extracted, parquet and labelled zones, because
        an extractor defines what a flow *is*: the same dataset extracted two
        ways is two different tables and must never share a path. `schema_version`
        scopes the canonical zone, which pins the feature space transitively.
        """
        if zone not in self.templates:
            raise LakeError(f"unknown zone {zone!r}; known: {', '.join(sorted(self.templates))}")
        # Truncation at the first unfilled placeholder is `zones.fill`'s job, so
        # that a zone resolves to the same location however it is addressed.
        # `source` defaults rather than truncating: every source-scoped zone
        # needs one, and making every caller pass "datasets" would be noise.
        template = zones.fill(self.templates[zone], dataset=dataset,
                              feature_space=feature_space,
                              schema_version=schema_version,
                              method=method,
                              source=source or self.default_source)
        return zones.join(self.root, template, rel or "")

    # ---------------------------------------------------------------- init

    def init(self, sources: tuple[str, ...] = zones.SOURCES) -> list[zones.ZoneDir]:
        """Materialise the lake wherever it lives.

        Same call for a directory, an S3 bucket, GCS, Azure or anything else
        fsspec reaches — `zones.plan` computes locations, the backend decides
        what "make this exist" means there. That is the whole point of the
        split: pointing Talos somewhere new is a config line, not a code change.
        """
        made = []
        for entry in zones.plan(self.root, self.templates, sources):
            created = self.backend.mkdir(entry.uri)
            made.append(replace(entry, created=created))
        self.write_lake_file()
        return made

    def write_lake_file(self) -> str | None:
        """Stamp the lake's identity at its root, if it is not already there.

        Never overwritten: the id is what downstream artefacts refer back to,
        and re-initialising an existing lake must not silently give it a new
        one.
        """
        uri = zones.join(self.root, LAKE_FILE)
        if self.backend.exists(uri):
            return None
        import uuid
        from datetime import datetime, timezone
        body = (f"lake_id: {uuid.uuid4()}\n"
                f"format_version: 1\n"
                f"created_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
        return self.backend.write_text(uri, body)

    def parquet_glob(self, zone: str, dataset: str, **scope) -> str:
        return f"{self.uri(zone, dataset, **scope)}/**/*.parquet"

    # ----------------------------------------------------------- delegation

    def exists(self, uri: str) -> bool:
        return self.backend.exists(uri)

    def put(self, local: Path, uri: str) -> str:
        return self.backend.put(Path(local), uri)

    def list(self, prefix: str) -> list[str]:
        return self.backend.list(prefix)

    def read_text(self, uri: str) -> str:
        return self.backend.read_text(uri)

    def write_text(self, uri: str, text: str) -> str:
        return self.backend.write_text(uri, text)

    def get(self, uri: str, local: Path) -> Path:
        return self.backend.get(uri, Path(local))

    def put_tree(self, local: Path, uri: str) -> str:
        return self.backend.put_tree(Path(local), uri)

    # ---------------------------------------------------------- parquet io

    def read_parquet(self, uri: str, columns: list[str] | None = None):
        """A relation over one parquet file, a glob, or a whole zone.

        `union_by_name` because the lake mirrors Zeek's tree one file per log,
        and a capture that never saw a given field simply has no column for it;
        positional union would silently misalign those.
        """
        cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
        return self.duck.relation(
            f"SELECT {cols} FROM read_parquet('{uri}', union_by_name = true)")

    def write_parquet(self, rel, uri: str) -> str:
        """Write a relation out as one parquet file.

        Goes through COPY rather than fetching rows into Python: the whole point
        of handing relations around is that a 2.1M-row table never materialises
        outside DuckDB.
        """
        if not zones.is_remote(uri):
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
        # Written from the relation's OWN connection. That is where the data
        # already is, and it is the connection whose memory limit, spill
        # directory and credentials the caller deliberately set; routing the
        # write through a second connection would both copy rows between them
        # and write under settings nobody chose.
        rel.write_parquet(uri)
        return uri

    def seal(self, uri: str) -> None:
        self.backend.seal(uri)

    def unseal(self, uri: str) -> None:
        self.backend.unseal(uri)

    def __repr__(self) -> str:
        kind = "remote" if self.remote else "local"
        return f"<LakeClient {self.root} ({kind})>"
