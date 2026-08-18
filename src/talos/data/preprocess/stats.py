"""Per-column statistics, computed without reading any label.

Everything a label-free screen needs, plus the distribution shape that decides
how a numeric should be scaled: skew, kurtosis and tail ratios before and after
log1p. Aggregates for every column come from one scan; only the dominant-value
share needs its own, because it is a GROUP BY rather than an aggregate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

#: The quantiles the tail ratios are built from.
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

#: Guards a ratio whose denominator is a quantile that can legitimately be zero.
EPSILON = 1e-12


class StatsError(Exception):
    pass


def quoted(column: str) -> str:
    """Zeek column names contain dots (`id.orig_h`), so quoting is not optional."""
    return '"' + column.replace('"', '""') + '"'


@dataclass(frozen=True)
class Shape:
    """How heavy-tailed a numeric is, and therefore what may scale it."""

    skewness: float | None
    kurtosis: float | None
    quantiles: tuple[float, ...]
    minimum: float | None
    maximum: float | None
    mean: float | None
    stddev: float | None

    def quantile(self, q: float) -> float | None:
        try:
            return self.quantiles[QUANTILES.index(q)]
        except (ValueError, IndexError):
            return None

    @property
    def tail_ratio(self) -> float | None:
        """p99 / p50 — how far the tail sits above the typical value."""
        p50, p99 = self.quantile(0.50), self.quantile(0.99)
        if p50 is None or p99 is None:
            return None
        return p99 / max(abs(p50), EPSILON)

    @property
    def extreme_ratio(self) -> float | None:
        """max / p99 — how far the worst row sits beyond the tail itself.

        This is the number that decides D1: a large value means a mean and a
        standard deviation are both set by a handful of rows.
        """
        p99 = self.quantile(0.99)
        if p99 is None or self.maximum is None:
            return None
        return self.maximum / max(abs(p99), EPSILON)

    def to_dict(self) -> dict:
        return {
            "skewness": self.skewness, "kurtosis": self.kurtosis,
            "min": self.minimum, "max": self.maximum,
            "mean": self.mean, "stddev": self.stddev,
            "quantiles": dict(zip((str(q) for q in QUANTILES), self.quantiles)),
            "tail_ratio_p99_p50": self.tail_ratio,
            "extreme_ratio_max_p99": self.extreme_ratio,
        }


@dataclass(frozen=True)
class ColumnStats:
    """One column, as far as it can be described without a label."""

    name: str
    kind: str                       # "numeric" | "categorical"
    rows: int
    nulls: int
    distinct: int
    dominant_value: str | None = None
    dominant_count: int = 0
    raw: Shape | None = None
    logged: Shape | None = None     # the same shape after log1p
    top: tuple = ()                 # categorical only

    @property
    def null_share(self) -> float:
        return self.nulls / self.rows if self.rows else 0.0

    @property
    def dominant_share(self) -> float:
        """NULL counts as a value here: an all-NULL column is as dead as a constant."""
        return self.dominant_count / self.rows if self.rows else 0.0

    def to_dict(self) -> dict:
        out = {
            "column": self.name, "kind": self.kind, "rows": self.rows,
            "nulls": self.nulls, "null_share": self.null_share,
            "distinct": self.distinct, "dominant_value": self.dominant_value,
            "dominant_share": self.dominant_share,
        }
        if self.raw:
            out["raw"] = self.raw.to_dict()
        if self.logged:
            out["log1p"] = self.logged.to_dict()
        if self.top:
            out["top_values"] = [dict(zip(("value", "flows"), row))
                                 for row in self.top]
        return out


class ColumnProfiler:
    """Reads a relation and describes each declared column."""

    def __init__(self, duck, source: str, top: int = 20):
        self.duck = duck
        self.source = source
        self.top = int(top)
        self._rows: int | None = None

    @property
    def rows(self) -> int:
        if self._rows is None:
            self._rows = self.duck.one(f"SELECT count(*) FROM {self.source}")[0]
        return self._rows

    def profile(self, numeric: Sequence[str],
                categorical: Sequence[str]) -> tuple[ColumnStats, ...]:
        stats = [self._numeric(c) for c in numeric]
        stats += [self._categorical(c) for c in categorical]
        return tuple(stats)

    # ------------------------------------------------------------- numerics

    def _numeric(self, column: str) -> ColumnStats:
        col = quoted(column)
        # log1p on the absolute value: a negative here is an integrity failure
        # the validity screen reports, not something this should hide or crash on.
        logged = f"ln(1 + abs(CAST({col} AS DOUBLE)))"
        raw = self._shape(f"CAST({col} AS DOUBLE)")
        nulls = self.duck.one(
            f"SELECT count(*) FROM {self.source} WHERE {col} IS NULL")[0]
        distinct = self.duck.one(
            f"SELECT count(DISTINCT {col}) FROM {self.source}")[0]
        value, count = self._dominant(column)
        return ColumnStats(
            name=column, kind="numeric", rows=self.rows, nulls=nulls,
            distinct=distinct, dominant_value=value, dominant_count=count,
            raw=raw, logged=self._shape(logged))

    def _shape(self, expression: str) -> Shape:
        quantiles = ", ".join(str(q) for q in QUANTILES)
        row = self.duck.one(f"""
            SELECT skewness({expression}), kurtosis({expression}),
                   min({expression}), max({expression}),
                   avg({expression}), stddev_pop({expression}),
                   quantile_cont({expression}, [{quantiles}])
            FROM {self.source}""")
        skew, kurt, low, high, mean, sd, qs = row
        return Shape(skewness=_float(skew), kurtosis=_float(kurt),
                     quantiles=tuple(_float(q) or 0.0 for q in (qs or ())),
                     minimum=_float(low), maximum=_float(high),
                     mean=_float(mean), stddev=_float(sd))

    # --------------------------------------------------------- categoricals

    def _categorical(self, column: str) -> ColumnStats:
        col = quoted(column)
        nulls = self.duck.one(
            f"SELECT count(*) FROM {self.source} WHERE {col} IS NULL")[0]
        distinct = self.duck.one(
            f"SELECT count(DISTINCT {col}) FROM {self.source}")[0]
        top = tuple(self.duck.sql(f"""
            SELECT CAST({col} AS VARCHAR), count(*) FROM {self.source}
            GROUP BY 1 ORDER BY 2 DESC LIMIT {self.top}"""))
        value, count = self._dominant(column)
        return ColumnStats(
            name=column, kind="categorical", rows=self.rows, nulls=nulls,
            distinct=distinct, dominant_value=value, dominant_count=count,
            top=top)

    def _dominant(self, column: str) -> tuple[str | None, int]:
        """The most common value and its count, counting NULL as a value."""
        col = quoted(column)
        row = self.duck.one(f"""
            SELECT CAST({col} AS VARCHAR), count(*) AS n FROM {self.source}
            GROUP BY 1 ORDER BY n DESC LIMIT 1""")
        if row is None:
            return None, 0
        return row[0], int(row[1])


def _float(value) -> float | None:
    """Decimal in, float out -- and an undefined statistic becomes None.

    A constant column has no skew, and DuckDB returns NaN for it. Kept as NaN it
    would reach the run report, where `json.dumps` writes a bare `NaN` that is
    not valid JSON and that no strict parser will read back.
    """
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) or math.isinf(number) else number
