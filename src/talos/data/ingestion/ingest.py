"""The front door: captures in, raw zone out.

This is where a capture stops being a file on someone's disk and becomes a thing
the platform can reason about. Three obligations, and the first two are cheap
only if they happen here rather than later:

**Provenance.** Every capture is recorded with its size, its modification time
and (by default) a content hash. Nothing downstream can establish this
retrospectively -- once a pcap is in the lake there is no way to tell whether it
is the file that was ingested.

**Immutability.** The raw zone is write-once by contract, and on a filesystem
that is an obligation rather than a property. Ingest seals the dataset when it
finishes. A later ingest into the same dataset unseals, writes and reseals, so
the window in which the zone is writable is as small as it can be.

**The declared UTC offset.** The single most expensive open question in this
project is what timezone CIC-IDS-2017's published schedule is in -- no primary
source states one, so every label rests on an inference. That is unfixable for a
public dataset, but it is entirely preventable for netem and honeypot captures,
which we control. Declaring the offset at ingest is the whole cure, and it costs
one flag.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from talos.common import zones
from talos.common.config import Config
from talos.common.provenance import content_sha

logger = logging.getLogger(__name__)

MANIFEST = "_capture_manifest.json"
PCAP_SUFFIXES = (".pcap", ".pcapng", ".cap")


class IngestionError(Exception):
    pass


@dataclass
class IngestReport:
    dataset: str
    source: str
    destination: str
    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)      # already present, same size
    replaced: list[str] = field(default_factory=list)     # present, different size
    sealed: bool = False

    def summary(self) -> str:
        parts = [f"{len(self.ingested)} ingested"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} already present")
        if self.replaced:
            parts.append(f"{len(self.replaced)} REPLACED")
        parts.append("sealed" if self.sealed else "not sealed")
        return ", ".join(parts)


def describe(path: Path, hash_contents: bool = True) -> dict:
    """What we record about one capture.

    Size and mtime are free and catch the accident -- a truncated copy, a file
    swapped for a different one. The content hash catches the deliberate, and is
    the only thing that makes a later `verify` possible at all, so it is on by
    default: a one-time cost at ingest against a permanent inability to check.
    """
    stat = path.stat()
    record = {
        "file": path.name,
        "bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                            .isoformat(timespec="seconds"),
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if hash_contents:
        record["sha256_12"] = content_sha(path)
    return record


class IngestionService:
    """Puts captures into the raw zone, with their provenance, and seals it."""

    def __init__(self, cfg: Config, lake=None):
        self.cfg = cfg
        self.lake = lake if lake is not None else cfg.lake()

    # ------------------------------------------------------------- manifest

    def manifest_uri(self, dataset: str, source: str) -> str:
        return self.lake.uri("raw", dataset=dataset, source=source, rel=MANIFEST)

    def read_manifest(self, dataset: str, source: str) -> dict:
        uri = self.manifest_uri(dataset, source)
        if not self.lake.exists(uri):
            return {}
        try:
            return json.loads(self.lake.read_text(uri))
        except (ValueError, OSError):
            # An unreadable manifest must not block an ingest, but it must not
            # be silently replaced either -- the captures it described are still
            # in the zone.
            logger.warning("existing capture manifest at %s is unreadable; "
                           "starting a new one", uri)
            return {}

    def write_manifest(self, dataset: str, source: str, captures: list[dict],
                       utc_offset: str | None, previous: dict) -> str:
        """Merge this ingest into whatever the zone already recorded.

        Captures accumulate across ingests, keyed by filename, because the zone
        is the sum of every ingest into it rather than the last one.
        """
        merged = {c["file"]: c for c in previous.get("captures", [])}
        merged.update({c["file"]: c for c in captures})

        # A declared offset is never silently dropped by a later ingest that
        # omits it, and a conflicting one is refused rather than overwritten.
        existing = previous.get("utc_offset")
        if utc_offset and existing and utc_offset != existing:
            raise IngestionError(
                f"{dataset} was ingested with utc_offset {existing!r} and this run "
                f"declares {utc_offset!r}. One of them is wrong, and every label "
                f"derived from this capture depends on which.")

        body = {
            "dataset": dataset,
            "source": source,
            # None means "not established", which is materially different from
            # an offset somebody assumed. CIC-IDS-2017 is the cautionary case.
            "utc_offset": utc_offset or existing,
            "captures": sorted(merged.values(), key=lambda c: c["file"]),
            "last_ingest_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ingests": int(previous.get("ingests", 0)) + 1,
        }
        return self.lake.write_text(self.manifest_uri(dataset, source),
                                    json.dumps(body, indent=2) + "\n")

    # --------------------------------------------------------------- ingest

    def ingest(self, dataset: str, files, source: str | None = None,
               utc_offset: str | None = None, hash_contents: bool = True,
               seal: bool = True) -> IngestReport:
        source = source or self.cfg.source_of(dataset)
        if source not in zones.SOURCES:
            raise IngestionError(
                f"unknown source {source!r}; known: {', '.join(zones.SOURCES)}")
        if utc_offset:
            zones_offset_check(utc_offset)

        destination = self.lake.uri("raw", dataset=dataset, source=source)
        report = IngestReport(dataset=dataset, source=source, destination=destination)
        previous = self.read_manifest(dataset, source)
        known = {c["file"]: c for c in previous.get("captures", [])}

        paths = [Path(f) for f in files]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise IngestionError(
                "not found: " + ", ".join(str(p) for p in missing))

        # The zone is sealed after every ingest, so it has to be opened before
        # another one can write. The window stays as short as possible.
        self.lake.unseal(destination)

        captures = []
        for path in paths:
            if path.suffix.lower() not in PCAP_SUFFIXES:
                logger.warning("%s is not a pcap, skipping", path.name)
                continue

            record = describe(path, hash_contents)
            prior = known.get(path.name)
            if prior and prior.get("bytes") == record["bytes"] and \
                    prior.get("sha256_12", record.get("sha256_12")) == \
                    record.get("sha256_12"):
                report.skipped.append(path.name)
                continue
            if prior:
                # Overwriting an immutable zone. Allowed, because the alternative
                # is a stuck ingest, but never silent -- provenance downstream
                # assumed the old bytes.
                logger.warning("%s already exists with different contents and is "
                               "being REPLACED in an immutable zone", path.name)
                report.replaced.append(path.name)

            self.lake.put(path, f"{destination}/{path.name}")
            captures.append(record)
            report.ingested.append(path.name)

        self.write_manifest(dataset, source, captures, utc_offset, previous)
        if seal:
            self.lake.seal(destination)
            report.sealed = True
        return report


def zones_offset_check(offset: str) -> None:
    """Reject an unparseable offset here rather than at labelling time."""
    from talos.data.labelling.windows import WindowError, parse_offset
    try:
        parse_offset(offset)
    except WindowError as exc:
        raise IngestionError(str(exc)) from exc


# ------------------------------------------------------------------- module

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(prog="talos ingest",
                                description=__doc__.splitlines()[0])
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--source", default=None, choices=zones.SOURCES,
                   help="defaults to the dataset's declaration in config.yml")
    p.add_argument("--utc-offset", default=None,
                   metavar="OFFSET",
                   help="the capture host's UTC offset. Must use the = form, "
                        "since a leading minus reads as a flag: "
                        "--utc-offset=-03:00. Declare it for netem and honeypot "
                        "captures -- not knowing it is what makes CIC-IDS-2017's "
                        "labels rest on an inference")
    p.add_argument("--no-hash", action="store_true",
                   help="skip content hashing (faster; makes `verify` impossible)")
    p.add_argument("--no-seal", action="store_true",
                   help="leave the raw zone writable after ingest")
    p.add_argument("files", nargs="+")
    a = p.parse_args(argv)

    cfg = Config.load(a.config)
    report = IngestionService(cfg).ingest(
        a.dataset, a.files, source=a.source, utc_offset=a.utc_offset,
        hash_contents=not a.no_hash, seal=not a.no_seal)

    print(f"dataset     {report.dataset}  ({report.source})")
    print(f"destination {report.destination}")
    print(f"            {report.summary()}")
    if not a.utc_offset and report.ingested:
        print("\nnote: no --utc-offset declared. For a capture you control, "
              "declaring it now is what stops its labels resting on a guess.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
