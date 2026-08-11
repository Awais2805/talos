"""Zeek, in a container, one capture at a time.

Zeek always writes `conn.log` to its working directory, so every pcap gets its
own output subdirectory named after the capture. That naming is more
load-bearing than it looks: the parquet tree mirrors it, labelling recovers
`capture` from the path, and `capture` is what makes a leakage-safe train/test
split possible -- attacks are contiguous bursts, so a random flow-level split
puts the same flood on both sides of the boundary and reports a fictional score.
A directory name here decides whether an evaluation is honest.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from talos.data.extraction.base import (
    BYTE_COUNTS, CONN_BACKBONE, FLOW_ID, BaseExtractor, ExtractionError,
    ExtractionReport, register_extractor, version_from_tag,
)

logger = logging.getLogger(__name__)

# Loaded on every run. These CHANGE THE CONN SCHEMA -- mac-logging adds
# orig_l2_addr and resp_l2_addr -- so they are part of the config fingerprint,
# and changing them forks the feature space.
DEFAULT_SCRIPTS = ("protocols/conn/mac-logging",)

# Zeek writes conn.log for any capture it processed, so the presence of one is
# the resumption checkpoint. No separate state store to fall out of sync.
CHECKPOINT = "conn.log"


@register_extractor
class ZeekExtractor(BaseExtractor):

    name = "zeek"

    def __init__(self, image: str = "zeek/zeek:8.2.1", flags: str = "-C",
                 json_logs: bool = True, scripts=DEFAULT_SCRIPTS,
                 version: str | None = None):
        self.image = image
        self.flags = tuple(flags.split() if isinstance(flags, str) else flags)
        self.json_logs = bool(json_logs)
        self.scripts = tuple(scripts or ())
        self._declared_version = version
        self._resolved_version: str | None = None

    @property
    def version(self) -> str:
        """The real Zeek version, never a moving tag.

        Resolution order, chosen so the common case costs nothing: an explicit
        `version:` in config wins; a tag that already looks like a version is
        taken as one; and only a moving tag such as `latest` actually starts a
        container to ask. That last case must not be guessed -- recording
        `zeek_vlatest` would let two different builds share a feature space,
        which is the single thing it exists to prevent.
        """
        if self._declared_version:
            return self._declared_version
        if self._resolved_version is None:
            self._resolved_version = version_from_tag(self.image) or self._ask_container()
        return self._resolved_version

    def _ask_container(self) -> str:
        if not shutil.which("docker"):
            raise ExtractionError(
                f"image {self.image!r} carries no usable version tag, and docker is "
                f"not available to ask the container. Pin the tag (zeek/zeek:8.2.1) "
                f"or declare `version:` under `zeek:` in config.yml -- an unpinned "
                f"extractor would let two different builds share a feature space.")
        result = subprocess.run(["docker", "run", "--rm", self.image, "zeek", "--version"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise ExtractionError(
                f"could not read the Zeek version from {self.image}: "
                f"{result.stderr.strip()[:400]}")
        return result.stdout.strip().split()[-1]        # "zeek version 8.2.1"

    def capabilities(self) -> frozenset[str]:
        """Zeek supplies all three: `uid` per connection, a conn backbone every
        other log hangs off, and directional byte counts in orig_bytes."""
        return frozenset({FLOW_ID, CONN_BACKBONE, BYTE_COUNTS})

    def config_fingerprint(self) -> dict:
        """What changes the columns Zeek emits."""
        return {"flags": list(self.flags), "scripts": list(self.scripts),
                "json_logs": self.json_logs}

    def available(self) -> tuple[bool, str]:
        if shutil.which("docker"):
            return True, ""
        return False, "zeek extraction runs in a container, and docker is not installed"

    # ------------------------------------------------------------------- run

    def command(self, pcap: Path, out_dir: Path) -> list[str]:
        """The container invocation. Separated out so a test can read it."""
        cmd = ["docker", "run", "--rm",
               "-v", f"{pcap.parent}:/pcap_in:ro",      # the raw zone is immutable
               "-v", f"{out_dir}:/zeek_out",
               "-w", "/zeek_out",
               self.image, "zeek",
               *self.flags,
               "-r", f"/pcap_in/{pcap.name}"]
        if self.json_logs:
            cmd.append("LogAscii::use_json=T")
        cmd.extend(self.scripts)
        return cmd

    def extract(self, pcap_paths: list[str], output_dir: str) -> ExtractionReport:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        ok, why = self.available()
        if not ok:
            raise ExtractionError(why)

        report = ExtractionReport()
        for raw in pcap_paths:
            pcap = Path(raw).resolve()
            capture = pcap.stem
            if not pcap.exists():
                report.failed.append((capture, "pcap not found"))
                continue

            out_dir = out_root / capture
            if (out_dir / CHECKPOINT).exists():
                # The existence of the output IS the checkpoint, so a run that
                # died 400 pcaps into 2018 resumes rather than restarting.
                report.skipped.append(capture)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)

            logger.info("extracting %s with %s v%s", pcap.name, self.name, self.version)
            result = subprocess.run(self.command(pcap, out_dir),
                                    capture_output=True, text=True)
            if result.returncode != 0:
                # Recorded, not swallowed: the caller decides whether a partial
                # extraction may be written, and the sidecar says it was partial.
                report.failed.append((capture, result.stderr.strip()[:400]))
                logger.error("failed on %s: %s", pcap.name, result.stderr.strip()[:200])
            else:
                report.extracted.append(capture)
        return report
