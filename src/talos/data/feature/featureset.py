"""Which columns a model sees, declared and hashed.

Nothing here is fitted: a vocabulary learned per capture would make the same
column index mean a different thing in each domain. Constraints are carried,
not interpreted — a reconstruction loss consumes them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from talos.common.plugins import ConfigPlugins, Declaration
from talos.common.provenance import Sha12

FEATURES_DIR = Path(__file__).with_name("features")
DEFAULT_FEATURES = "conn"

#: A feature name becomes a column identifier and an index into the matrix.
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

IDENTITY, LOG1P = "identity", "log1p"
TRANSFORMS = (IDENTITY, LOG1P)

#: `ratio`: a = b / c. `sum`: a = b + c. Each is a residual a decoder is penalised on.
RATIO, SUM = "ratio", "sum"
FORMS = (RATIO, SUM)

#: How Eq. 2's residual is reduced. `l1` is the equation; `log_square` is an
#: adaptation for byte-scale residuals and invalidates the paper's phi.
L1, LOG_SQUARE = "l1", "log_square"
NORMS = (L1, LOG_SQUARE)

#: Where a value outside the vocabulary goes, and where a NULL goes.
OTHER = "__other"

#: Suffix of the companion column marking that a numeric value was missing.
MISSING = "__missing"


class FeatureError(Exception):
    pass


#: NOT "feature space": that already means the extractor version (`zeek_v8.2.1`).
FEATURE_SETS = ConfigPlugins("feature set", built_in=FEATURES_DIR,
                             error=FeatureError)


@dataclass(frozen=True)
class Numeric:
    """One continuous feature."""

    name: str
    expr: str
    transform: str = IDENTITY
    missing: float = 0.0
    indicator: bool = False
    note: str = ""
    #: Declared, never fitted, and in TRANSFORMED space -- standardisation is
    #: applied after `transform`, so a log1p feature's centre is the median of
    #: log1p(x), not log1p of the median. Both absent means pass through.
    centre: float | None = None
    scale: float | None = None

    @property
    def standardised(self) -> bool:
        return self.centre is not None and self.scale is not None

    def sql(self) -> str:
        """Cast, impute, transform, then standardise.

        Imputing first keeps the sentinel where it was declared; standardising
        last means the constants describe the values a model actually sees.
        """
        value = f"coalesce(CAST({self.expr} AS DOUBLE), {self.missing})"
        if self.transform == LOG1P:
            # Guarded: a negative value makes log1p NaN, which spreads silently.
            value = f"ln(1 + greatest({value}, 0))"
        if self.standardised:
            return f"(({value}) - {self.centre}) / {self.scale}"
        return value

    def clamped_sql(self) -> str:
        """1.0 where the log1p guard fired, so a clamp is counted, never silent.

        After the validity screen a clamp should be impossible; one that fires
        anyway is a defect in the screen, and this is what makes it visible.
        """
        if self.transform != LOG1P:
            return "0.0"
        value = f"coalesce(CAST({self.expr} AS DOUBLE), {self.missing})"
        return f"CASE WHEN ({value}) < 0 THEN 1.0 ELSE 0.0 END"

    def indicator_sql(self) -> str:
        return f"CASE WHEN ({self.expr}) IS NULL THEN 1.0 ELSE 0.0 END"


@dataclass(frozen=True)
class Categorical:
    """One discrete feature with a declared vocabulary."""

    name: str
    expr: str
    values: tuple[str, ...]
    note: str = ""

    @property
    def width(self) -> int:
        return len(self.values) + 1

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(f"{self.name}_{v}" for v in self.values) + (self.name + OTHER,)

    def sql(self) -> tuple[str, ...]:
        """One indicator per value, then a catch-all for unknown AND for NULL."""
        quoted = [f"'{v.replace(chr(39), chr(39) * 2)}'" for v in self.values]
        cols = [f"CASE WHEN {self.expr} = {q} THEN 1.0 ELSE 0.0 END" for q in quoted]
        known = ", ".join(quoted)
        cols.append(f"CASE WHEN {self.expr} IS NULL OR {self.expr} NOT IN ({known}) "
                    f"THEN 1.0 ELSE 0.0 END")
        return tuple(cols)


@dataclass(frozen=True)
class Constraint:
    """A relation that holds in a real record, for a reconstruction to respect."""

    name: str
    form: str
    target: str
    operands: tuple[str, ...]
    #: Eq. 2's phi_m. None means "use the method's global residual_weight".
    weight: float | None = None

    def describe(self) -> str:
        op = " / " if self.form == RATIO else " + "
        return f"{self.name}: {self.target} = {op.join(self.operands)}"


@dataclass(frozen=True)
class FeatureSet:
    """One declared, hashed view of what a row is."""

    name: str
    numeric: tuple[Numeric, ...]
    categorical: tuple[Categorical, ...]
    constraints: tuple[Constraint, ...]
    sha: Sha12
    path: Path
    doc: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Bare column names this set reads, for a pre-flight against a table."""
        refs = [f.expr for f in self.numeric] + [c.expr for c in self.categorical]
        return tuple(sorted({r for r in refs if _NAME.match(r)}))

    def describe(self) -> str:
        lines = [f"feature set   {self.name}  sha {self.sha}",
                 f"  numeric     {len(self.numeric)} "
                 f"(+{sum(1 for n in self.numeric if n.indicator)} missingness)",
                 f"  categorical {len(self.categorical)} -> "
                 f"{sum(c.width for c in self.categorical)} columns"]
        for constraint in self.constraints:
            lines.append(f"  constraint  {constraint.describe()}")
        return "\n".join(lines)


class FeatureSetLoader:
    """Reads a feature declaration and refuses the ways it can be wrong."""

    def __init__(self, point: ConfigPlugins = FEATURE_SETS):
        self.point = point

    def load(self, name: str | Path = DEFAULT_FEATURES) -> FeatureSet:
        declaration = self.point.get(name)
        doc = declaration.doc
        numeric = self._numeric(doc.get("numeric") or {}, doc.get("derived") or {})
        categorical = self._categorical(doc.get("categorical") or {})
        constraints = self._constraints(doc.get("constraints") or [])
        self._validate(declaration, numeric, categorical, constraints)
        return FeatureSet(
            name=doc.get("name", declaration.name), numeric=numeric,
            categorical=categorical, constraints=constraints,
            sha=declaration.sha, path=declaration.path, doc=doc)

    # ------------------------------------------------------------- parsing

    @staticmethod
    def _numeric(declared: Mapping[str, Any],
                 derived: Mapping[str, Any]) -> tuple[Numeric, ...]:
        """One list, so a constraint never spans two blocks of the matrix."""
        out: list[Numeric] = []
        for source in (declared, derived):
            for name, entry in source.items():
                entry = entry or {}
                centre, scale = entry.get("centre"), entry.get("scale")
                out.append(Numeric(
                    name=name, expr=str(entry.get("expr", name)),
                    transform=str(entry.get("transform", IDENTITY)),
                    missing=float(entry.get("missing", 0.0)),
                    indicator=bool(entry.get("indicator", False)),
                    note=str(entry.get("note", "")),
                    centre=None if centre is None else float(centre),
                    scale=None if scale is None else float(scale)))
        return tuple(out)

    @staticmethod
    def _categorical(declared: Mapping[str, Any]) -> tuple[Categorical, ...]:
        return tuple(
            Categorical(name=name, expr=str((entry or {}).get("expr", name)),
                        values=tuple(str(v) for v in (entry or {}).get("values") or ()),
                        note=str((entry or {}).get("note", "")))
            for name, entry in declared.items())

    @staticmethod
    def _constraints(declared: Sequence[Any]) -> tuple[Constraint, ...]:
        return tuple(
            Constraint(name=str(entry.get("name", "")), form=str(entry.get("form", "")),
                       target=str(entry.get("target", "")),
                       operands=tuple(str(o) for o in entry.get("operands") or ()),
                       weight=(None if entry.get("weight") is None
                               else float(entry["weight"])))
            for entry in declared if isinstance(entry, dict))

    # ---------------------------------------------------------- validation

    @staticmethod
    def _validate(declaration: Declaration, numeric: tuple[Numeric, ...],
                  categorical: tuple[Categorical, ...],
                  constraints: tuple[Constraint, ...]) -> None:
        """Every refusal is for a failure that is otherwise silent."""
        errs: list[str] = []
        if not numeric and not categorical:
            errs.append("declares no features, so a model would be fitted to a "
                        "matrix with no columns")

        names = [f.name for f in numeric] + [c.name for c in categorical]
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            errs.append(f"{', '.join(duplicated)} declared more than once. A name "
                        f"is an index into the matrix, so it must be unique")
        for name in names:
            if not _NAME.match(name):
                errs.append(f"feature name {name!r} must match [a-z][a-z0-9_]*")

        # A name is a column wherever this schema is written, and DuckDB folds
        # identifiers case-insensitively -- two names differing only by case
        # would merge into one column and silently lose half the signal.
        folded = [n.lower() for n in names]
        collided = sorted({n for n in folded if folded.count(n) > 1})
        if collided and not duplicated:
            errs.append(f"{', '.join(collided)}: two features differ only by case. "
                        f"Written to a table they would fold into one column")

        for feature in numeric:
            if feature.transform not in TRANSFORMS:
                errs.append(f"{feature.name}: transform {feature.transform!r} is "
                            f"not one of {', '.join(TRANSFORMS)}")
            if (feature.centre is None) != (feature.scale is None):
                errs.append(
                    f"{feature.name}: declares only "
                    f"{'centre' if feature.scale is None else 'scale'}. "
                    f"Standardisation needs both, and half of one is not a "
                    f"transform anybody chose")
            if feature.scale is not None and feature.scale == 0:
                errs.append(f"{feature.name}: scale is 0, which divides every row "
                            f"by zero. A constant column has no scale; drop it "
                            f"instead of standardising it")

        for feature in categorical:
            if not feature.values:
                errs.append(
                    f"{feature.name}: categorical with no declared `values`. The "
                    f"vocabulary must be written down, not learned -- a fitted one "
                    f"differs per capture, so the same column index would mean a "
                    f"different value in each domain")
            if len(set(feature.values)) != len(feature.values):
                repeated = sorted({v for v in feature.values
                                   if feature.values.count(v) > 1})
                errs.append(f"{feature.name}: repeats {', '.join(repeated)}. Each "
                            f"value is one column and the second could never fire")

        known = set(names)
        for constraint in constraints:
            if constraint.form not in FORMS:
                errs.append(f"{constraint.name or '<unnamed>'}: form "
                            f"{constraint.form!r} is not one of {', '.join(FORMS)}")
            unknown = sorted({f for f in (constraint.target, *constraint.operands)
                              if f not in known})
            if unknown:
                errs.append(
                    f"{constraint.name or '<unnamed>'}: names undeclared feature(s) "
                    f"{', '.join(unknown)}. The residual would be computed over a "
                    f"column that does not exist, or silently skipped")
            if len(constraint.operands) != 2:
                errs.append(f"{constraint.name or '<unnamed>'}: {constraint.form} "
                            f"takes exactly two operands, got "
                            f"{len(constraint.operands)}")

        if errs:
            raise FeatureError(f"{declaration.path}:\n  " + "\n  ".join(errs))
