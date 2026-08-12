""" Chooses the set of classes to classify, i.e. 5 classes, 9 classes or binary
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from talos.common.plugins import ConfigPlugins, Declaration
import re

from talos.common.provenance import Sha12
from talos.data.labelling.taxonomy import BENIGN, TaxonomyMapper

SPACES_DIR = Path(__file__).with_name("spaces")
DEFAULT_SPACE = "core-5"

#: The quality flag an out-of-space flow carries. Distinct from `uncertain`,
#: which means "we are unsure this label is right"; this means "this label is
#: fine and the study cannot use it".
OUT_OF_SPACE = "out-of-space"

#: An exclusion reason is written into every out-of-space row, grouped on
#: downstream, and interpolated into generated SQL. Strict, so that all three
#: are safe: "no spaces" alone would still admit an apostrophe.
_REASON = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class LabelSpaceError(Exception):
    pass


def _quote(value: str) -> str:
    """A SQL string literal. Class names come from a user-supplied taxonomy."""
    return "'" + str(value).replace("'", "''") + "'"


#: The label-space plug-in point. A user declares their own set of classes -- a
#: binary space, or one over attack families their own traffic actually contains.
SPACES = ConfigPlugins("label space", built_in=SPACES_DIR, error=LabelSpaceError)


@dataclass(frozen=True)
class Exclusion:
    """A class held out of the space, and the argument for holding it out."""

    label: str
    reason: str
    flows: int | None = None
    domains: int | None = None
    note: str | None = None

    def describe(self) -> str:
        evidence = []
        if self.flows is not None:
            evidence.append(f"{self.flows:,} flows")
        if self.domains is not None:
            evidence.append(f"{self.domains} domain(s)")
        detail = f" ({', '.join(evidence)})" if evidence else ""
        return f"{self.label:<14} {self.reason}{detail}"


@dataclass(frozen=True)
class LabelSpace:
    """One declared set of predictable classes, hashed."""

    name: str
    classes: tuple[str, ...]
    excluded: Mapping[str, Exclusion]
    sha: Sha12
    path: Path
    doc: Mapping[str, Any]

    # ------------------------------------------------------------- membership

    def __contains__(self, label: object) -> bool:
        return label in self.classes

    @property
    def n_classes(self) -> int:
        """What a classifier's output layer has to be wide enough for."""
        return len(self.classes)

    @property
    def attack_classes(self) -> tuple[str, ...]:
        return tuple(c for c in self.classes if c != BENIGN)

    def index(self, label: str) -> int:
        """Position in the declared order -- the noise matrix's row and column."""
        try:
            return self.classes.index(label)
        except ValueError:
            raise LabelSpaceError(
                f"{label!r} is not in label space {self.name!r}: "
                f"{', '.join(self.classes)}") from None

    def reason_for(self, label: str) -> str | None:
        """Why this class is out, or None if it is in (or unknown to the space)."""
        exclusion = self.excluded.get(label)
        return exclusion.reason if exclusion else None

    # ------------------------------------------------------------------- SQL

    def out_of_space_sql(self, column: str = "label_class") -> str:
        """`CASE` mapping a class to its exclusion reason, else NULL.

        Generated from the declaration rather than written by hand, so a class
        added to `excluded:` is flagged without anybody editing the writer.
        """
        if not self.excluded:
            return "CAST(NULL AS VARCHAR)"
        # The class name comes from a user-supplied taxonomy, so it is escaped.
        # The reason is pattern-checked at load and cannot need escaping.
        whens = " ".join(
            f"WHEN {column} = {_quote(label)} THEN '{e.reason}'"
            for label, e in sorted(self.excluded.items()))
        return f"CASE {whens} ELSE CAST(NULL AS VARCHAR) END"

    def describe(self) -> str:
        lines = [f"label space   {self.name}  sha {self.sha}",
                 f"  classes     {', '.join(self.classes)}"]
        for label in sorted(self.excluded):
            lines.append(f"  excluded    {self.excluded[label].describe()}")
        return "\n".join(lines)


class LabelSpaceLoader:
    """Reads a label-space declaration and refuses the ways it can be wrong."""

    def __init__(self, point: ConfigPlugins = SPACES):
        self.point = point

    def load(self, name: str | Path = DEFAULT_SPACE,
             taxonomy: TaxonomyMapper | None = None) -> LabelSpace:
        declaration = self.point.get(name)
        classes = tuple(declaration.doc.get("classes") or ())
        excluded = self._exclusions(declaration)
        self._validate(declaration, classes, excluded, taxonomy)
        return LabelSpace(
            name=declaration.doc.get("name", declaration.name),
            classes=classes, excluded=excluded, sha=declaration.sha,
            path=declaration.path, doc=declaration.doc)

    @staticmethod
    def _exclusions(declaration: Declaration) -> dict[str, Exclusion]:
        out: dict[str, Exclusion] = {}
        for label, entry in (declaration.doc.get("excluded") or {}).items():
            entry = entry or {}
            out[label] = Exclusion(
                label=label, reason=str(entry.get("reason", "")).strip(),
                flows=entry.get("flows"), domains=entry.get("domains"),
                note=entry.get("note"))
        return out

    @staticmethod
    def _validate(declaration: Declaration, classes: tuple[str, ...],
                  excluded: Mapping[str, Exclusion],
                  taxonomy: TaxonomyMapper | None) -> None:
        """Every refusal here is for a failure that would otherwise be silent."""
        errs = []
        if not classes:
            errs.append("declares no classes, so nothing could be predicted")
        if classes and BENIGN not in classes:
            errs.append(
                f"no {BENIGN!r} class. Closed-world labelling falls to it, so a "
                f"space without it would emit a class outside itself")
        if len(set(classes)) != len(classes):
            duplicated = sorted({c for c in classes if classes.count(c) > 1})
            errs.append(f"lists {', '.join(duplicated)} more than once; the order "
                        f"is the noise matrix's index, so it must be a sequence")

        both = sorted(set(classes) & set(excluded))
        if both:
            errs.append(f"{', '.join(both)} appear in classes AND excluded. One "
                        f"of the two is a leftover and nothing can tell which")

        for label, exclusion in sorted(excluded.items()):
            if not exclusion.reason:
                errs.append(f"{label}: excluded with no reason. Holding a class "
                            f"out of a study is an argument and has to be on record")
            elif not _REASON.match(exclusion.reason):
                errs.append(f"{label}: reason {exclusion.reason!r} is not an "
                            f"identifier. It is written into every out-of-space "
                            f"row, grouped on downstream and interpolated into "
                            f"SQL, so it must match [a-z0-9][a-z0-9-]*; put the "
                            f"explanation in `note`")

        if taxonomy is not None:
            named = set(classes) | set(excluded)
            unknown = sorted(named - set(taxonomy.classes))
            if unknown:
                errs.append(
                    f"names classes the taxonomy does not declare: "
                    f"{', '.join(unknown)}. A typo would silently exclude nothing "
                    f"and predict nothing")

        if errs:
            raise LabelSpaceError(f"{declaration.path}:\n  " + "\n  ".join(errs))
