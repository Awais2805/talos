"""Extracted logs -> parquet (and optionally CSV), one output per source log.

One implementation, two output formats. It used to be two near-identical
modules, which is how the same bug got written twice: both filtered the
provenance sidecar in as if it were data, and both resolved zone paths without a
source. A fix had to be applied in two places or it was not applied at all.

**The tree is mirrored, never reshaped.** One parquet per source `.log`, same
directory structure, no flattening and no joins. Reshaping here would destroy
the `uid` backbone that labelling and the alert enricher both depend on, and the
capture folder that makes a leakage-safe split possible.

**Both formats are written from one read.** A log is parsed once and copied out
twice, because the JSON parse dominates and reading the extracted zone twice to
produce two encodings of the same rows is pure waste.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from talos.common import zones
from talos.common.config import Config
from talos.data.extraction.base import META_FILENAME

logger = logging.getLogger(__name__)

LOG_SUFFIXES = (".log", ".json")

# Output format -> (lake zone, file suffix, COPY options).
FORMATS = {
    "parquet": ("parquet", ".parquet", "FORMAT PARQUET"),
    "csv": ("csv", ".csv", "FORMAT CSV, HEADER true"),
}


class ConversionError(Exception):
    pass


@dataclass
class ConversionReport:
    dataset: str
    source: str
    feature_space: str
    src: str
    destinations: dict[str, str] = field(default_factory=dict)
    converted: int = 0
    rows: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.converted:,} file(s), {self.rows:,} rows"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} SKIPPED")
        return ", ".join(parts)


class Converter:
    """Reads the extracted zone for one dataset and writes the mirror zones."""

    def __init__(self, cfg: Config, lake=None):
        self.cfg = cfg
        self.lake = lake if lake is not None else cfg.lake()

    # ----------------------------------------------------- feature space

    def resolve_feature_space(self, dataset: str, source: str,
                              explicit: str | None = None,
                              allow_incomplete: bool = False) -> str:
        """The extraction to convert: the most recent COMPLETE one.

        An incomplete extraction is skipped rather than preferred for being
        newer. The sidecar records `complete: false` when captures failed, and
        converting that tree produces a parquet zone quietly missing flows --
        which then looks like a clean dataset to everything downstream.
        """
        if explicit:
            return explicit

        base = self.lake.uri("extracted", source=source).rstrip("/")
        candidates, incomplete = [], []
        for item in self.lake.list(base):
            if not item.endswith(f"/{dataset}/{META_FILENAME}"):
                continue
            try:
                meta = json.loads(self.lake.read_text(item))
            except (ValueError, OSError):
                continue
            space, when = meta.get("feature_space"), meta.get("extracted_at_utc", "")
            if not space:
                continue
            (candidates if meta.get("complete", True) else incomplete).append(
                (when, space, meta))

        if not candidates and incomplete and allow_incomplete:
            candidates = incomplete
        if not candidates:
            if incomplete:
                worst = max(incomplete)[2]
                raise ConversionError(
                    f"the only extraction of {dataset!r} is incomplete: "
                    f"{worst.get('pcaps_failed', '?')} capture(s) failed. Re-run "
                    f"`talos extract` to finish it, or pass --allow-incomplete to "
                    f"convert a partial tree knowingly.")
            raise ConversionError(self._not_found(dataset, source, base))
        return max(candidates)[1]

    def _not_found(self, dataset: str, source: str, base: str) -> str:
        """Say where it ISN'T, and where it is.

        A dataset not declared in config.yml defaults to the `datasets` source,
        so a netem capture ingested with --source netem is looked for in the
        wrong tree. Rather than "not found", name the tree that actually holds
        it -- the answer is one cheap existence check per source.
        """
        elsewhere = [other for other in zones.SOURCES if other != source
                     and self.lake.exists(
                         self.lake.uri("extracted", source=other).rstrip("/")
                         + f"/{dataset}")
                     or self.lake.exists(
                         self.lake.uri("raw", dataset=dataset, source=other))]
        message = f"no extraction of {dataset!r} under {base}."
        if elsewhere:
            found = ", ".join(elsewhere)
            return (f"{message} It exists under: {found}. Either pass "
                    f"--source {elsewhere[0]}, or declare it in config.yml:\n"
                    f"  datasets:\n    {dataset}:\n      source: {elsewhere[0]}")
        return f"{message} Run `talos extract` first."

    # ------------------------------------------------------------ convert

    def convert(self, dataset: str, formats=("parquet",), source: str | None = None,
                feature_space: str | None = None, logtypes=None,
                allow_incomplete: bool = False) -> ConversionReport:
        unknown = [f for f in formats if f not in FORMATS]
        if unknown:
            raise ConversionError(
                f"unknown format(s) {', '.join(unknown)}; "
                f"known: {', '.join(FORMATS)}")

        source = source or self.cfg.source_of(dataset)
        feature_space = self.resolve_feature_space(
            dataset, source, feature_space, allow_incomplete)

        src = self.lake.uri("extracted", dataset=dataset, source=source,
                            feature_space=feature_space)
        report = ConversionReport(dataset=dataset, source=source,
                                  feature_space=feature_space, src=src)
        for fmt in formats:
            zone = FORMATS[fmt][0]
            report.destinations[fmt] = self.lake.uri(
                zone, dataset=dataset, source=source, feature_space=feature_space)

        logs = self._logs(src, logtypes)
        if not logs:
            logger.warning("no extracted logs under %s", src)
            return report

        for log in logs:
            rel = log[len(src):].lstrip("/")
            try:
                report.rows += self._convert_one(log, rel, formats, report.destinations)
                report.converted += 1
            except Exception as exc:                    # noqa: BLE001
                # One malformed log must not abandon the other 3,000. Recorded
                # per file so a partial conversion is visible, not inferred.
                report.skipped.append((rel, str(exc).splitlines()[0][:200]))
        return report

    def _logs(self, src: str, logtypes) -> list[str]:
        found = [f for f in self.lake.list(src)
                 if f.endswith(LOG_SUFFIXES) and Path(f).name != META_FILENAME]
        if logtypes:
            want = set(logtypes)
            found = [f for f in found if Path(f).stem in want]
        return found

    def _convert_one(self, log: str, rel: str, formats, destinations) -> int:
        """Parse once, write once per format.

        `ignore_errors` because Zeek occasionally emits a truncated final line
        when a capture ends mid-record, and losing one row is better than losing
        the file.
        """
        reader = (f"read_parquet('{log}')" if log.endswith(".parquet") else
                  f"read_json_auto('{log}', format='newline_delimited', "
                  f"ignore_errors=true)")
        rows = 0
        for fmt in formats:
            _, suffix, options = FORMATS[fmt]
            target = f"{destinations[fmt].rstrip('/')}/{Path(rel).with_suffix(suffix)}"
            if not target.startswith(("s3://", "gs://", "abfs://")):
                Path(target).parent.mkdir(parents=True, exist_ok=True)
            # COPY reports what it wrote, so the row count costs nothing. The
            # previous version re-read each output with count(*) to learn it --
            # a second full round trip per file, thousands of times.
            written = self.lake.duck.sql(
                f"COPY (SELECT * FROM {reader}) TO '{target}' ({options})")
            rows = written[0][0] if written else 0
        return rows
