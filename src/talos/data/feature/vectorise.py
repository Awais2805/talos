"""A declared feature set applied to a table: rows -> a numeric matrix.

The transform is SQL, so it runs inside the scan. What comes back is a matrix
AND a `ColumnSpec`: without the one-hot block boundaries a reconstruction loss
squares everything and trains a decoder to emit 0.4 of a protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator, Mapping, Sequence

import numpy as np

from talos.common.plugins import CodePlugins
from talos.data.feature.featureset import (
    IDENTITY, LOG1P, MISSING, Categorical, FeatureError, FeatureSet, Numeric,
    RATIO,
)

DEFAULT_BATCH = 65_536

#: What a prediction joins back on; overridable per extractor.
DEFAULT_KEY = "uid"


class VectoriserError(FeatureError):
    pass


VECTORISERS = CodePlugins("vectoriser", error=VectoriserError)


@dataclass(frozen=True)
class Block:
    """A contiguous run of columns that mean one thing together."""

    name: str
    start: int
    width: int

    @property
    def stop(self) -> int:
        return self.start + self.width

    @property
    def indices(self) -> range:
        return range(self.start, self.stop)


@dataclass(frozen=True)
class ColumnSpec:
    """What each column of the matrix is, and which ones belong together."""

    names: tuple[str, ...]
    numeric: tuple[int, ...]
    categorical: tuple[Block, ...]
    constraints: tuple[tuple[str, str, int, tuple[int, ...]], ...]
    #: Column index -> its transform. A relation holds between ORIGINAL values.
    transforms: Mapping[int, str] = field(default_factory=dict)
    #: Feature column -> its missingness indicator, where one was declared.
    indicators: Mapping[int, int] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise VectoriserError(
                f"{name!r} is not a column of this matrix. Have: "
                f"{', '.join(self.names[:8])}…") from None

    def describe(self) -> str:
        blocks = ", ".join(f"{b.name}[{b.start}:{b.stop}]" for b in self.categorical)
        return (f"{self.width} columns: {len(self.numeric)} numeric, "
                f"{len(self.categorical)} one-hot blocks ({blocks or 'none'})")


class Vectoriser(ABC):
    """Turns a relation into batches of numbers, according to a feature set."""

    name: ClassVar[str] = ""

    def __init__(self, features: FeatureSet, key: str = DEFAULT_KEY):
        self.features = features
        self.key = key

    # ------------------------------------------------------------- contract

    @abstractmethod
    def numeric_sql(self, feature: Numeric) -> str:
        """How one continuous feature becomes a column."""

    @abstractmethod
    def categorical_sql(self, feature: Categorical) -> Sequence[str]:
        """How one discrete feature becomes a block of columns."""

    # ---------------------------------------------------------------- shape

    def columns(self) -> ColumnSpec:
        """Numeric first, then one one-hot block per categorical, in declared order."""
        names: list[str] = []
        transforms: dict[int, str] = {}
        indicators: dict[int, int] = {}
        for feature in self.features.numeric:
            position = len(names)
            transforms[position] = feature.transform
            names.append(feature.name)
            if feature.indicator:
                indicators[position] = len(names)
                transforms[len(names)] = IDENTITY      # an indicator is already 0/1
                names.append(feature.name + MISSING)
        numeric = tuple(range(len(names)))

        blocks: list[Block] = []
        for feature in self.features.categorical:
            blocks.append(Block(feature.name, len(names), feature.width))
            names.extend(feature.columns)

        return ColumnSpec(names=tuple(names), numeric=numeric,
                          categorical=tuple(blocks),
                          constraints=self._constraint_indices(tuple(names)),
                          transforms=transforms, indicators=indicators)

    def _constraint_indices(
            self, names: tuple[str, ...]
    ) -> tuple[tuple[str, str, int, tuple[int, ...]], ...]:
        """Resolve constraints to column positions once, not per batch per epoch."""
        resolved = []
        for constraint in self.features.constraints:
            fields = (constraint.target, *constraint.operands)
            missing = [f for f in fields if f not in names]
            if missing:
                raise VectoriserError(
                    f"constraint {constraint.name!r} names {', '.join(missing)}, "
                    f"which {'is' if len(missing) == 1 else 'are'} not a single "
                    f"column of the matrix. A categorical occupies a block and has "
                    f"no single value.")
            resolved.append((constraint.name, constraint.form,
                             names.index(constraint.target),
                             tuple(names.index(o) for o in constraint.operands)))
        return tuple(resolved)

    # ------------------------------------------------------------------ SQL

    def select_list(self, carry: Sequence[str] = ()) -> str:
        """The projection, key first, then any carried columns."""
        parts = [f'"{self.key}" AS __key']
        parts += [f'"{column}"' for column in carry]
        for feature in self.features.numeric:
            parts.append(f"{self.numeric_sql(feature)} AS \"{feature.name}\"")
            if feature.indicator:
                parts.append(f'{feature.indicator_sql()} AS "{feature.name}{MISSING}"')
        for feature in self.features.categorical:
            for column, expression in zip(feature.columns,
                                          self.categorical_sql(feature)):
                parts.append(f'{expression} AS "{column}"')
        return ",\n       ".join(parts)

    def project(self, rel, carry: Sequence[str] = ()):
        """The relation, reduced to the key plus the declared feature columns."""
        return rel.project(self.select_list(carry))

    # -------------------------------------------------------------- reading

    def batches(self, rel, size: int = DEFAULT_BATCH) -> Iterator[tuple]:
        """`(keys, X)` per batch. Streamed: D_l is tens of millions of rows."""
        projected = self.project(rel)
        while True:
            rows = projected.fetchmany(size)
            if not rows:
                return
            yield self._split(rows)

    def matrix(self, rel) -> tuple:
        """`(keys, X)` for the whole relation. For small pools and for tests."""
        rows = self.project(rel).fetchall()
        return self._split(rows)

    def labelled_matrix(self, rel, column: str) -> tuple:
        """`(keys, X, values)` — one carried column, read in the SAME projection.

        Two separate scans would not be guaranteed to come back in the same
        order, so the targets could silently belong to different rows.
        """
        rows = self.project(rel, carry=(column,)).fetchall()
        keys, X = self._split(rows, carried=1)
        return keys, X, np.array([r[1] for r in rows], dtype=object)

    def _split(self, rows: Sequence[Sequence], carried: int = 0) -> tuple:
        """Key off the front, the rest to float64 whatever backend consumes it."""
        first = 1 + carried
        keys = np.array([r[0] for r in rows], dtype=object)
        values = np.array([r[first:] for r in rows], dtype=np.float64)
        if values.size and values.shape[1] != self.columns().width:
            raise VectoriserError(
                f"projection produced {values.shape[1]} columns, the column spec "
                f"describes {self.columns().width}")
        return keys, values


@VECTORISERS.register
class PassthroughVectoriser(Vectoriser):
    """Applies the declaration and nothing else — nothing is fitted here.

    Standardisation is deliberately absent: it is estimated from data, so it
    belongs to a model's checkpoint. Centre each domain on its own mean and the
    between-domain differences this project measures quietly disappear.
    """

    name: ClassVar[str] = "passthrough"

    def numeric_sql(self, feature: Numeric) -> str:
        return feature.sql()

    def categorical_sql(self, feature: Categorical) -> Sequence[str]:
        return feature.sql()


def untransform(column, transform: str, xp=np):
    """Recover the original quantity. `log1p` is exactly invertible by `expm1`."""
    return xp.expm1(column) if transform == LOG1P else column


@dataclass(frozen=True)
class Residual:
    """How far rows depart from one declared constraint, and where it applies."""

    name: str
    value: Any                   # zero wherever `applicable` is False
    applicable: Any              # bool per row

    @property
    def n_applicable(self) -> int:
        return int(self.applicable.sum())

    def mean_square(self):
        """Averaged over applicable rows, so capture quality does not move the score.

        Deliberately not cast to `float`: on a torch tensor that would sever the
        graph, and this is a term a reconstruction loss adds.
        """
        if not self.n_applicable:
            return 0.0
        return (self.value ** 2).sum() / self.n_applicable

    def mean_log_square(self, xp=np):
        """The training reduction: squared `log1p|residual|`.

        A residual on bytes-per-second spans as many orders of magnitude as the
        feature does, so a plain square optimises entirely for the single worst
        row in the batch — the same argument that puts `log1p` on the features.
        """
        if not self.n_applicable:
            return 0.0
        return (xp.log1p(abs(self.value)) ** 2).sum() / self.n_applicable


def residuals(X, spec: ColumnSpec, xp=np, mask_from=None) -> dict[str, Residual]:
    """Departure from each declared constraint, in ORIGINAL units.

    The transform is undone first: `log1p(a / b)` is not `log1p(a) / log1p(b)`.
    A constraint does not apply where an operand was imputed or a denominator is
    zero — undefined there, not violated. `mask_from` supplies applicability when
    `X` is a reconstruction: whether a value was absent is a fact about the input.
    """
    if xp is np:
        X = np.asarray(X, dtype=np.float64)
    truth = X if mask_from is None else mask_from
    out: dict[str, Residual] = {}
    for name, form, target, operands in spec.constraints:
        fields = (target, *operands)
        want, left, right = (untransform(X[:, i], spec.transforms.get(i, IDENTITY), xp)
                             for i in fields)

        applicable = xp.ones_like(want) > 0
        for column in fields:
            indicator = spec.indicators.get(column)
            if indicator is not None:
                applicable = applicable & (truth[:, indicator] == 0.0)

        if form == RATIO:
            applicable = applicable & (right != 0)
            # Guard the denominator itself: masking afterwards leaves inf * 0 = NaN.
            expected = left / xp.where(right != 0, right, xp.ones_like(right))
        else:
            expected = left + right

        out[name] = Residual(name=name, value=applicable * (want - expected),
                             applicable=applicable)
    return out


def build(name: str, features: FeatureSet, key: str = DEFAULT_KEY) -> Vectoriser:
    return VECTORISERS.get(name, features=features, key=key)
