"""Attack schedules, transcribed from the datasets' own documentation.

Labels never come from the traffic. A capture records what happened, not whether
it was malicious, so every label has to come from an external record of ground
truth joined onto the flows afterwards. That record is one YAML file per
dataset: a list of attack windows, each with the endpoints involved.

The published CSVs do ship labels, but they are attached to *CICFlowMeter*
flows. We re-extracted with Zeek, the two tools cut flows at different
boundaries, and matching by 5-tuple would reintroduce exactly the fragility we
left CICFlowMeter to escape. Hence the schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from talos.data.labelling.quarantine import QuarantineLoader, QuarantineRegion
from talos.data.labelling.taxonomy import TaxonomyMapper
from talos.data.labelling.windows import (
    Window, WindowError, build_all, extend_windows, ips, resolve_when,
    subnet_prefix, to_utc,
)

DEFAULT_DIR = Path(__file__).with_name("manifests")


class ManifestError(Exception):
    pass


@dataclass(frozen=True)
class AttackRule(Window):
    """A window that asserts an attack, plus what the attack is.

    `canonical` defaults to empty rather than to a class. Inheriting from a
    frozen dataclass forces the new fields to carry defaults, and the obvious
    choice -- "benign" -- would mean an attack rule built without a class
    silently labels its own flows benign. Empty is refused instead.
    """

    canonical: str = ""
    needs_payload: bool = False
    note: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if not self.canonical:
            raise WindowError(
                f"{self.label()}: no canonical class. A rule with no class would "
                f"label its own flows benign.")


@dataclass(frozen=True)
class Manifest:
    """One dataset's schedule, normalised and ready to join."""

    dataset: str
    path: Path
    rules: tuple[AttackRule, ...]
    # Loaded from the same parse as the rules, and already clamped against
    # them. Keeping the two together is what stops a caller handing the region
    # loader clamp starts from a different read of the file.
    regions: tuple[QuarantineRegion, ...] = ()

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(sorted({r.canonical for r in self.rules}))

    def rule(self, rid: int) -> AttackRule:
        return self.rules[rid]


class ManifestLoader:
    """Reads a manifest and normalises it into joinable rules."""

    def __init__(self, directory: str | Path = DEFAULT_DIR):
        self.directory = Path(directory)

    def path_for(self, dataset: str) -> Path:
        path = self.directory / f"{dataset}.yaml"
        if not path.exists():
            available = ", ".join(sorted(p.stem for p in self.directory.glob("*.yaml")))
            raise ManifestError(f"no manifest for {dataset!r} in {self.directory}. "
                                f"Have: {available}")
        return path

    # ------------------------------------------------------------------ load

    def load(self, dataset: str, taxonomy: TaxonomyMapper | None = None) -> Manifest:
        taxonomy = taxonomy or TaxonomyMapper()
        path = self.path_for(dataset)
        doc = yaml.safe_load(path.read_text()) or {}

        entries = doc.get("attacks") or []
        if not entries:
            raise ManifestError(f"{path}: no attacks declared")

        taxonomy.validate_covers(e["name"] for e in entries)
        # Invalid windows raise at construction; build_all collects them so a
        # manifest with four mistakes costs one round trip rather than four.
        rules = tuple(build_all(
            lambda pair: self._rule(pair[0], pair[1], doc, taxonomy),
            list(enumerate(entries))))

        # Attack windows clamp against each other; the same starts are handed to
        # the quarantine loader so regions clamp against the schedule too.
        rules = tuple(extend_windows(rules, [r.t0 for r in rules]))

        return Manifest(
            dataset=doc.get("dataset", dataset),
            path=path,
            rules=rules,
            regions=QuarantineLoader().load(doc, [r.t0 for r in rules]),
        )

    # --------------------------------------------------------------- helpers

    def _rule(self, rid: int, entry: dict, doc: dict, taxonomy: TaxonomyMapper) -> AttackRule:
        """Build one rule. `rid` is the entry's POSITION in the YAML list.

        Which means inserting a rule in the middle renumbers every later one,
        and `rule_id` in already-written parquet then refers to a different
        attack. Nothing detects that except comparing `manifest_sha`, so append
        rather than insert, or accept that older labelled tables are stale.
        """
        name = entry["name"]
        try:
            date, offset = resolve_when(entry, doc)
            t0 = to_utc(date, entry["start"], offset)
            t1 = to_utc(date, entry["end"], offset)
        except (KeyError, ValueError, WindowError) as exc:
            raise ManifestError(f"{name} {date}: {exc}") from exc

        return AttackRule(
            rid=rid,
            name=name,
            t0=t0,
            t1=t1,                      # replaced by extend_windows
            t1_raw=t1,
            attackers=ips(entry.get("attacker_ips")),
            victims=ips(entry.get("victim_ips")),
            subnet_prefix=subnet_prefix(entry.get("victim_subnet")),
            date=date,
            start=entry["start"],
            end=entry["end"],
            canonical=taxonomy.canonical(name),
            needs_payload=taxonomy.requires_payload(name),
            note=entry.get("note"),
        )
