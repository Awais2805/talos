"""Raw attack names -> the nine canonical classes.

`DoS-Hulk`, `DoS-GoldenEye` and `DoS-Slowloris` all have to denote the same thing as 2018's
`DoS-SlowHTTPTest` before a model trained on one can be tested against the
other. The raw name is always retained alongside the canonical class, so
mapping never costs provenance.

`requires_payload` keys on the **raw** name, not the class. `DoS-Hulk` requires payload while
`DoS-Slowloris` does not, and both are `dos`. Slow-DoS holds sockets open by
sending almost nothing, so a class-level rule would mark real attack traffic as
mere attempts. Port scans, SYN floods and UDP reflection are excluded for the
same reason — for those, an empty connection *is* the mechanism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

DEFAULT_PATH = Path(__file__).with_name("manifests") / "taxonomy.yaml"

# The class every unmatched flow falls to under the closed-world assumption.
# Defined here, beside the vocabulary it must belong to, so the labeller cannot
# emit a class the taxonomy does not declare.
BENIGN = "benign"


class TaxonomyError(Exception):
    pass


class TaxonomyMapper:
    """The canonical class vocabulary and the mapping onto it."""

    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)
        doc = yaml.safe_load(self.path.read_text()) or {}
        self._classes = tuple(doc.get("canonical_classes") or {})
        self._mapping = dict(doc.get("raw_to_canonical") or {})
        self._needs_payload = frozenset(doc.get("requires_payload") or [])
        self._check_internally_consistent()

    def _check_internally_consistent(self) -> None:
        """A mapping onto a class that does not exist is a typo, caught at load."""
        if not self._classes:
            raise TaxonomyError(f"{self.path}: no canonical_classes declared")
        if BENIGN not in self._classes:
            raise TaxonomyError(
                f"{self.path}: no {BENIGN!r} class. Every flow no rule matched is "
                f"labelled with it under the closed-world assumption, so a taxonomy "
                f"without it would emit a class outside its own vocabulary.")
        unknown = {raw: cls for raw, cls in self._mapping.items()
                   if cls not in self._classes}
        if unknown:
            listed = ", ".join(f"{r} -> {c}" for r, c in sorted(unknown.items()))
            raise TaxonomyError(
                f"{self.path}: mapped to classes that are not declared: {listed}")
        stray = self._needs_payload - set(self._mapping)
        if stray:
            raise TaxonomyError(
                f"{self.path}: requires_payload names nothing in raw_to_canonical: "
                f"{', '.join(sorted(stray))}")

    # ---------------------------------------------------------------- lookup

    @property
    def classes(self) -> tuple[str, ...]:
        return self._classes

    def canonical(self, raw_name: str) -> str:
        """Canonical class for a raw attack name. Unknown names are an error.

        Never a default: a silent fallback to `benign` would delete an attack
        class, and a fallback to anything else would invent one.
        """
        try:
            return self._mapping[raw_name]
        except KeyError:
            raise TaxonomyError(
                f"attack {raw_name!r} is not in {self.path.name}; add it to "
                f"raw_to_canonical rather than letting it fall through") from None

    def requires_payload(self, raw_name: str) -> bool:
        """True when the attacker must send data for the attack to have executed."""
        return raw_name in self._needs_payload

    def raw_names_for(self, canonical_class: str) -> tuple[str, ...]:
        return tuple(sorted(r for r, c in self._mapping.items() if c == canonical_class))

    def validate_covers(self, raw_names: Iterable[str]) -> None:
        """Every name a manifest uses must map. Checked before a run, not during."""
        missing = sorted({n for n in raw_names if n not in self._mapping})
        if missing:
            raise TaxonomyError(
                f"{self.path.name} does not map: {', '.join(missing)}")
