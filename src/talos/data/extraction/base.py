"""The plugin contract for extraction, and the registry that resolves one.

An extractor defines what a flow *is*. The same pcap through Zeek and through
CICFlowMeter yields different row counts, different columns and different flow
boundaries; they are not comparable and must never be pooled. `feature_space`
is the identifier that makes that structural rather than advisory -- it scopes
zones 2-4, so two extractions cannot share a directory unless they genuinely
produced the same kind of table.

Which means the identifier has to change whenever the OUTPUT SCHEMA changes, not
merely when the tool does. Loading an extra Zeek script adds columns; running an
unpinned `latest` tag six months apart is two different builds. Both are covered
here, deliberately over-conservatively: a spurious fork wastes some disk, a
missed fork silently pools incompatible data into one training set.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from talos.common.plugins import CodePlugins

# The provenance sidecar every extraction run drops beside its logs. Named here
# because downstream stages have to recognise it to skip it: it sits in the same
# tree as the logs and would otherwise be converted as if it were one of them.
META_FILENAME = "_extractor_meta.json"

# A feature space becomes a directory name, so it may not smuggle a path
# separator or anything else that would change where data lands.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")

# A tag that looks like a real version can be trusted as one. `latest`, `dev` or
# an absent tag cannot, and the extractor must resolve them properly.
_VERSION_LIKE = re.compile(r"^v?\d+(\.\d+)*[A-Za-z0-9.\-_]*$")


class ExtractionError(Exception):
    pass


# What downstream stages need an extractor to supply. Declared per plugin so a
# stage can refuse, or degrade explicitly, instead of assuming Zeek's data model
# and producing something wrong-but-plausible.
FLOW_ID = "flow_id"              # a stable per-flow identifier (Zeek: uid)
CONN_BACKBONE = "conn_backbone"  # one record per connection, other logs hang off it
BYTE_COUNTS = "byte_counts"      # directional payload bytes (Zeek: orig_bytes)

CAPABILITY_MEANING = {
    FLOW_ID: "a stable per-flow id; labels are attached to it and inherited by "
             "every enrichment log",
    CONN_BACKBONE: "one record per connection; the schedule join is written "
                   "against connections, not packets or sub-events",
    BYTE_COUNTS: "directional payload byte counts; without them an attack cannot "
                 "be told from a connection attempt",
}


@dataclass
class ExtractionReport:
    """What a run actually managed. Returned rather than logged and forgotten."""

    extracted: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)   # (capture, why)
    skipped: list[str] = field(default_factory=list)              # already present

    @property
    def attempted(self) -> int:
        return len(self.extracted) + len(self.failed)

    @property
    def complete(self) -> bool:
        return not self.failed

    def merged_with(self, previous: dict | None) -> dict:
        """This run's outcome, folded into what the zone already recorded.

        A capture that failed before and succeeded (or was skipped, meaning its
        output is present) now counts as resolved. Everything else is still
        outstanding, so completeness describes the tree rather than the last
        invocation.
        """
        resolved = set(self.extracted) | set(self.skipped)
        carried = [(c, why) for c, why in
                   ((f["capture"], f["error"]) for f in (previous or {}).get("failures", []))
                   if c not in resolved and c not in dict(self.failed)]
        outstanding = carried + self.failed
        return {
            "pcaps_extracted": len(self.extracted),
            "pcaps_skipped": len(self.skipped),
            "pcaps_failed": len(outstanding),
            "failures": [{"capture": c, "error": why} for c, why in outstanding],
            # An incomplete extraction must be visible to anything that later
            # picks "the most recent feature space" -- otherwise a run where
            # most pcaps failed is indistinguishable from a clean one.
            "complete": not outstanding,
        }

    def as_meta(self) -> dict:
        return self.merged_with(None)

    def summary(self) -> str:
        parts = [f"{len(self.extracted)} extracted"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} already present")
        if self.failed:
            parts.append(f"{len(self.failed)} FAILED")
        return ", ".join(parts)


#: The extractor plug-in point. `EXTRACTORS.add(path)` imports a user's own `.py`
#: so they can bring their own tool without editing this package; `plugins:
#: extractor:` in config.yml does the same, once, for every command.
EXTRACTORS = CodePlugins("extractor", error=ExtractionError)

register_extractor = EXTRACTORS.register
get_extractor = EXTRACTORS.get
registered = EXTRACTORS.names


class BaseExtractor(ABC):
    """The contract every extraction tool must satisfy."""

    #: The tool's name, e.g. "zeek". A CLASS attribute, not a property: the
    #: registry reads it at import time without constructing anything, and a
    #: plugin's name does not vary between configured instances of it.
    name: ClassVar[str] = ""

    @property
    @abstractmethod
    def version(self) -> str:
        """The tool's resolved version, e.g. '8.2.1'. Never a moving tag."""

    def capabilities(self) -> frozenset[str]:
        """What this extractor's output can support downstream.

        Declared rather than assumed. A stage that needs something absent then
        refuses, or degrades in a recorded way, instead of running against a
        schema it silently mis-reads.
        """
        return frozenset()

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities()

    def require(self, *capabilities: str) -> None:
        """Refuse early when a stage cannot work against this extractor at all."""
        missing = [c for c in capabilities if not self.supports(c)]
        if missing:
            detail = "; ".join(f"{c} ({CAPABILITY_MEANING.get(c, '')})" for c in missing)
            raise ExtractionError(
                f"{self.name} does not provide: {detail}. This stage cannot run "
                f"against its output.")

    def config_fingerprint(self) -> dict:
        """Settings that change the SHAPE of the output, not merely the run.

        Anything returned here forks the feature space when it changes. Zeek's
        loaded scripts belong in it -- `protocols/conn/mac-logging` adds two
        columns to conn -- while a thread count would not.

        Recorded in the sidecar but NOT currently part of the feature space,
        which is tool and version only. So two runs of the same version with
        different settings land in the same directory today; give the second one
        a distinct version, or add the write-time fingerprint check.
        """
        return {}

    @property
    def config_sha(self) -> str:
        fingerprint = self.config_fingerprint()
        if not fingerprint:
            return ""
        canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:6]

    @property
    def feature_space(self) -> str:
        """The directory that scopes zones 2-4, e.g. `zeek_v8.2.1`.

        Tool and version only, for readability. `config_fingerprint()` is still
        recorded in the sidecar, so a write-time check ("this tree was extracted
        with different settings") can be added later without re-extracting
        anything. Until it is, changing a schema-affecting setting WITHOUT also
        changing the version will write into the existing tree -- see the
        warning on config_fingerprint.
        """
        space = f"{self.name}_v{self.version}"
        if not _SAFE_SEGMENT.match(space):
            raise ExtractionError(
                f"{space!r} is not usable as a directory name. A feature space "
                f"becomes a path segment; check the tool's name and version.")
        return space

    def available(self) -> tuple[bool, str]:
        """(can run here, why not). Reported so a caller can refuse early.

        Checked BEFORE any pcap is fetched -- discovering that a tool is
        unusable after downloading several hundred GB is an expensive way to
        read an error message.
        """
        return True, ""

    def write_metadata(self, output_dir: str, dataset: str,
                       report: ExtractionReport | None = None,
                       extra_meta: dict | None = None,
                       previous: dict | None = None) -> Path:
        """The provenance sidecar: the state of the TREE, not of one invocation.

        `previous` is the sidecar already in the zone, if any. Failures from an
        earlier run are carried forward unless this run resolved them, so a
        second pass over a subset cannot quietly report `complete` while five
        captures from the first pass are still missing.
        """
        meta = {
            "dataset": dataset,
            "extractor_tool": self.name,
            "extractor_version": self.version,
            "feature_space": self.feature_space,
            "capabilities": sorted(self.capabilities()),
            "config_fingerprint": self.config_fingerprint(),
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(report or ExtractionReport()).merged_with(previous),
            **(extra_meta or {}),
        }
        if previous and previous.get("first_extracted_at"):
            meta["first_extracted_at"] = previous["first_extracted_at"]
        else:
            meta["first_extracted_at"] = meta["extracted_at_utc"]
        meta["runs"] = int((previous or {}).get("runs", 0)) + 1

        out_path = Path(output_dir) / META_FILENAME
        out_path.write_text(json.dumps(meta, indent=2))
        return out_path

    @abstractmethod
    def extract(self, pcap_paths: list[str], output_dir: str) -> ExtractionReport:
        """Run the tool over each pcap, reporting what succeeded and what did not.

        Returns rather than raises on a per-pcap failure, so one unreadable
        capture does not discard the work already done -- but the caller is
        given the failures and must decide, rather than being told nothing.
        """


def version_from_tag(image: str) -> str | None:
    """A concrete version out of a Docker tag, or None if the tag is a moving one.

    `zeek/zeek:8.2.1` gives 8.2.1. `zeek/zeek:latest` gives None, because
    "latest" is not a version -- two runs months apart would share a feature
    space while being different builds, which is exactly the contamination the
    feature space exists to prevent.
    """
    tail = image.rsplit("/", 1)[-1]           # drop registry host, which may hold a port
    if ":" not in tail:
        return None
    tag = tail.rsplit(":", 1)[-1]
    return tag if _VERSION_LIKE.match(tag) else None
