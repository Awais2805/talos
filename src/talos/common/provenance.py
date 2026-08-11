"""Content-addressed lineage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from talos.common.paths import reports_dir

# First 12 hex chars of a SHA-256 digest. A str, but named where it is meant.
Sha12 = str

_CHUNK = 1 << 20        # 1 MiB: manifests are small, but pcaps and parquet are not

# A stamped column name goes into generated SQL, so it may only be an identifier.
_IDENT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class ProvenanceError(Exception):
    pass


def digest(data: bytes) -> Sha12:
    """Content hash of bytes already in hand."""
    return hashlib.sha256(data).hexdigest()[:12]


def content_sha(path: str | Path) -> Sha12:
    """Content hash of a file, read in chunks.

    Streamed rather than slurped because the same function is used for a 4 KB
    taxonomy and a 40 GB capture, and the small case must not dictate a shape
    that breaks the large one.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()[:12]


def _literal(value: Any) -> str:
    """A Python value as a SQL literal.

    Strings are single-quote escaped. This is generated SQL over values that
    include free-text reasons from a manifest, so an unescaped quote is a syntax
    error at best and a rewritten query at worst.
    """
    if value is None:
        return "CAST(NULL AS VARCHAR)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


@dataclass(frozen=True)
class ProvMeta:
    """The provenance columns stamped onto every row a stage writes.

    Column names are the caller's, not this module's: labelling writes
    `manifest_sha` and `labelled_at`, mapping will write something else, and a
    fixed schema here would force every stage to rename its own lineage. Held as
    ordered pairs so the stamped column order is deterministic -- a parquet file
    whose column order moves between runs diffs as changed when it is not.
    """

    columns: tuple[tuple[str, Any], ...]

    @classmethod
    def of(cls, **columns: Any) -> "ProvMeta":
        for name in columns:
            if not name or set(name) - _IDENT or name[0].isdigit():
                raise ProvenanceError(
                    f"{name!r} is not a usable column name; provenance columns are "
                    f"emitted as SQL identifiers")
        return cls(tuple(columns.items()))

    @classmethod
    def now(cls) -> str:
        """UTC, second precision -- the timestamp every stage stamps."""
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.columns)


def stamp_sql(meta: ProvMeta) -> str:
    """The provenance columns as a SQL select-list fragment.

    Separated from `stamp` so the escaping is testable without a database.
    """
    return ", ".join(f"{_literal(v)} AS {name}" for name, v in meta.columns)


class ProvenanceService:
    """Content hashes, row stamping, and run reports."""

    def __init__(self, reports: Path | None = None):
        self.reports = Path(reports) if reports is not None else reports_dir()

    # ------------------------------------------------------------- hashing

    digest = staticmethod(digest)
    content_sha = staticmethod(content_sha)

    @staticmethod
    def verify(path: str | Path, expected: Sha12) -> bool:
        """True when a file still hashes to what an artefact claims it did."""
        return content_sha(path) == expected

    # ------------------------------------------------------------ stamping

    @staticmethod
    def stamp(rel, meta: ProvMeta):
        """Append the provenance columns to every row of a relation.

        Stays on the relation's own connection -- this never opens one, so a
        caller's memory limits and credentials are the ones that apply.
        """
        return rel.project(f"*, {stamp_sql(meta)}")

    # ------------------------------------------------------------- reports

    def run_report(self, name: str, payload: Mapping[str, Any]) -> Path:
        """Write a stage's machine-readable run report and return its path.

        Every run emits one whether it succeeded or not: the report of a run
        that aborted is the most useful one there is.
        """
        path = self.reports / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n")
        return path
