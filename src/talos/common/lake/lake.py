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

import os
import shutil
import stat
from pathlib import Path

from talos.common import zones


class LakeError(Exception):
    pass


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

    def seal(self, uri: str) -> None:
        """Drop the write bit on everything under `uri`.

        This is the cheap half of the write-once obligation: it catches the
        accident. The content-hash sidecar catches the deliberate.
        """
        base = Path(uri)
        if not base.exists():
            return
        for path in [base, *base.rglob("*")]:
            mode = path.stat().st_mode
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    def unseal(self, uri: str) -> None:
        base = Path(uri)
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
        self.fs = fsspec.filesystem(root.split("://", 1)[0],
                                    client_kwargs={"region_name": region} if region else None)

    def exists(self, uri: str) -> bool:
        return self.fs.exists(uri)

    def put(self, local: Path, uri: str) -> str:
        self.fs.put(str(local), uri)
        return uri

    def list(self, prefix: str) -> list[str]:
        try:
            return sorted(self.fs.find(prefix))
        except FileNotFoundError:
            return []

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
                 region: str | None = None, backend=None):
        self.root = zones.normalise_root(root)
        self.templates = {**zones.DEFAULT_TEMPLATES, **(templates or {})}
        self.region = region
        self.backend = backend or backend_for(self.root, region)

    @property
    def remote(self) -> bool:
        return self.backend.remote

    def uri(self, zone: str, dataset: str | None = None, rel: str | None = None,
            feature_space: str | None = None, schema_version: str | None = None) -> str:
        """Logical address -> physical location.

        `feature_space` scopes the extracted, parquet and labelled zones, because
        an extractor defines what a flow *is*: the same dataset extracted two
        ways is two different tables and must never share a path. `schema_version`
        scopes the canonical zone, which pins the feature space transitively.
        """
        if zone not in self.templates:
            raise LakeError(f"unknown zone {zone!r}; known: {', '.join(sorted(self.templates))}")
        template = self.templates[zone]
        fields = {"dataset": dataset, "feature_space": feature_space,
                  "schema_version": schema_version}
        for name, value in fields.items():
            token = "{" + name + "}"
            if token not in template:
                continue
            if value is None:
                # Truncate at the first unfilled placeholder: that is the static
                # prefix, which is what listing a whole zone needs.
                template = template.split(token, 1)[0]
                break
            template = template.replace(token, str(value))
        return zones.join(self.root, template, rel or "")

    def parquet_glob(self, zone: str, dataset: str, **scope) -> str:
        return f"{self.uri(zone, dataset, **scope)}/**/*.parquet"

    # ----------------------------------------------------------- delegation

    def exists(self, uri: str) -> bool:
        return self.backend.exists(uri)

    def put(self, local: Path, uri: str) -> str:
        return self.backend.put(Path(local), uri)

    def list(self, prefix: str) -> list[str]:
        return self.backend.list(prefix)

    def seal(self, uri: str) -> None:
        self.backend.seal(uri)

    def unseal(self, uri: str) -> None:
        self.backend.unseal(uri)

    def __repr__(self) -> str:
        kind = "remote" if self.remote else "local"
        return f"<LakeClient {self.root} ({kind})>"
