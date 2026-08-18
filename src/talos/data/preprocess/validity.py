"""Row-level validity: values that cannot be true of a real flow.

Flags rows, drops none -- row selection is experiment-scoped. Runs before the
scaling constants are computed, because a median and a MAD taken over rows this
would flag are set by the very values it exists to find.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from talos.data.preprocess.stats import quoted

#: Bytes per second above any real link. 100 Gbps = 1.25e10 B/s; a flow above
#: that is a near-zero duration dividing a real byte count, not a transfer.
LINE_RATE = 1.25e10

#: Floating types are the only ones that can hold NaN or Infinity.
_FLOAT_TYPES = ("DOUBLE", "FLOAT", "REAL")


class ValidityError(Exception):
    pass


@dataclass(frozen=True)
class Invariant:
    """Something true of every real flow, and why it is true.

    `predicate` is SQL that is TRUE when the row is VALID, so a violation is its
    negation. NULL is never a violation: absence is the missingness screen's
    business, and conflating the two is how `orig_bytes IS NULL` and
    `orig_bytes = 0` stopped being distinguishable once before.
    """

    name: str
    predicate: str
    reason: str
    columns: tuple[str, ...]

    def violation_sql(self) -> str:
        return f"NOT ({self.predicate})"


#: Declared, not derived. `conn`'s `constraints:` block cannot serve here:
#: those are DEFINITIONS of derived features (`overhead_orig` is defined as
#: `orig_ip_bytes - orig_bytes`, so the sum identity is a tautology), and a
#: tautology cannot be violated by raw input.
INVARIANTS: tuple[Invariant, ...] = (
    Invariant(
        name="non_negative_orig_bytes",
        predicate='"orig_bytes" >= 0',
        reason="a byte count cannot be negative",
        columns=("orig_bytes",)),
    Invariant(
        name="non_negative_resp_bytes",
        predicate='"resp_bytes" >= 0',
        reason="a byte count cannot be negative",
        columns=("resp_bytes",)),
    Invariant(
        name="non_negative_duration",
        predicate='"duration" >= 0',
        reason="a connection cannot end before it starts",
        columns=("duration",)),
    Invariant(
        name="orig_ip_bytes_cover_payload",
        predicate='"orig_ip_bytes" >= "orig_bytes"',
        reason=("IP bytes include the headers that carry the payload, so they "
                "cannot be fewer than the payload itself; a violation makes "
                "`overhead_orig` negative, which log1p then silently clamps to 0"),
        columns=("orig_ip_bytes", "orig_bytes")),
    Invariant(
        name="resp_ip_bytes_cover_payload",
        predicate='"resp_ip_bytes" >= "resp_bytes"',
        reason="IP bytes include the headers that carry the payload",
        columns=("resp_ip_bytes", "resp_bytes")),
    Invariant(
        name="orig_bytes_need_packets",
        predicate='NOT ("orig_bytes" > 0 AND "orig_pkts" = 0)',
        reason="payload cannot arrive in zero packets",
        columns=("orig_bytes", "orig_pkts")),
    Invariant(
        name="resp_bytes_need_packets",
        predicate='NOT ("resp_bytes" > 0 AND "resp_pkts" = 0)',
        reason="payload cannot arrive in zero packets",
        columns=("resp_bytes", "resp_pkts")),
    Invariant(
        name="orig_rate_below_line_rate",
        predicate=(f'NOT ("duration" > 0 AND "orig_bytes" / "duration" > {LINE_RATE})'),
        reason=(f"above {LINE_RATE:.3g} B/s (100 Gbps) the rate is a near-zero "
                f"duration dividing a real byte count, not a transfer; the "
                f"result is finite, so no NaN or Inf check catches it"),
        columns=("orig_bytes", "duration")),
)


@dataclass(frozen=True)
class Violation:
    """One invariant, and how much of the table breaks it."""

    name: str
    rows: int
    share: float
    reason: str
    checked: bool = True
    skipped_because: str = ""

    def to_dict(self) -> dict:
        out = {"invariant": self.name, "rows": self.rows, "share": self.share,
               "reason": self.reason, "checked": self.checked}
        if self.skipped_because:
            out["skipped_because"] = self.skipped_because
        return out


@dataclass
class ValidityReport:
    """What could not be true, and how often."""

    dataset: str
    source: str = ""
    rows: int = 0
    violations: tuple[Violation, ...] = ()
    non_finite: dict[str, int] = field(default_factory=dict)
    alphabet: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_violating(self) -> int:
        """Upper bound: one row can break several invariants at once."""
        return sum(v.rows for v in self.violations)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset, "source": self.source, "rows": self.rows,
            "violations": [v.to_dict() for v in self.violations],
            "non_finite": self.non_finite,
            "history_alphabet": self.alphabet,
            "policy": ("rows are flagged, never dropped: row selection is "
                       "experiment-scoped, and an invalid value is routed "
                       "through the missing-value path at vectorise time"),
            **self.extra,
        }

    def table(self) -> str:
        lines = [f"{'invariant':<34}{'rows':>12}{'share':>10}  status"]
        for violation in self.violations:
            status = "ok" if violation.checked else \
                f"skipped ({violation.skipped_because})"
            if violation.checked and violation.rows:
                status = "!! VIOLATED"
            lines.append(f"{violation.name:<34}{violation.rows:>12,}"
                         f"{100 * violation.share:>9.4f}%  {status}")
        return "\n".join(lines)

    def summary(self) -> str:
        broken = [v for v in self.violations if v.checked and v.rows]
        lines = [f"rows          {self.rows:,}",
                 f"invariants    {len(self.violations)} declared, "
                 f"{len([v for v in self.violations if v.checked])} checked, "
                 f"{len(broken)} violated"]
        for violation in broken:
            lines.append(f"!! {violation.name}: {violation.rows:,} rows "
                         f"({100 * violation.share:.4f}%) -- {violation.reason}")
        if any(self.non_finite.values()):
            lines.append("!! non-finite values: " + ", ".join(
                f"{c} {n:,}" for c, n in self.non_finite.items() if n))
        if self.alphabet:
            letters = "".join(sorted(self.alphabet))
            lines.append(f"history alphabet ({len(self.alphabet)}): {letters}")
            if "C" in self.alphabet:
                lines.append("!! 'C' present: Zeek is documented as running with "
                             "-C, which should make checksum letters impossible")
        return "\n".join(lines)


class ValidityAuditor:
    """Counts rows that break a declared invariant. Changes nothing."""

    def __init__(self, duck, invariants: Sequence[Invariant] = INVARIANTS,
                 line_rate: float = LINE_RATE):
        self.duck = duck
        self.invariants = tuple(invariants)
        self.line_rate = float(line_rate)

    def audit(self, dataset: str, source: str, columns) -> ValidityReport:
        present = {c.lower(): c for c in dict(columns)}
        rows = self.duck.one(f"SELECT count(*) FROM {source}")[0]
        report = ValidityReport(dataset=dataset, source=source, rows=rows)
        report.violations = tuple(
            self._check(invariant, source, present, rows)
            for invariant in self.invariants)
        report.non_finite = self._non_finite(source, columns)
        return report

    def _check(self, invariant: Invariant, source: str, present: dict,
               rows: int) -> Violation:
        missing = [c for c in invariant.columns if c.lower() not in present]
        if missing:
            # A silently un-run check reads exactly like a passing one.
            return Violation(
                name=invariant.name, rows=0, share=0.0, reason=invariant.reason,
                checked=False,
                skipped_because=f"no column {', '.join(missing)}")
        count = self.duck.one(
            f"SELECT count(*) FROM {source} WHERE {invariant.violation_sql()}")[0]
        return Violation(name=invariant.name, rows=int(count),
                         share=count / rows if rows else 0.0,
                         reason=invariant.reason)

    def _non_finite(self, source: str, columns) -> dict[str, int]:
        """NaN and Infinity, which only a floating column can hold."""
        out: dict[str, int] = {}
        for name, kind in dict(columns).items():
            if not any(t in str(kind).upper() for t in _FLOAT_TYPES):
                continue
            col = quoted(name)
            out[name] = int(self.duck.one(
                f"SELECT count(*) FROM {source} "
                f"WHERE {col} IS NOT NULL AND NOT isfinite({col})")[0])
        return out


def history_alphabet(duck, source: str, column: str = "history") -> dict[str, int]:
    """Every state letter Zeek actually emitted, measured rather than assumed.

    The per-letter decomposition is declared from this, so the alphabet has to
    come from the data: a letter assumed from the manual and never emitted is a
    dead column in every domain.
    """
    col = quoted(column)
    rows = duck.sql(f"""
        SELECT letter, count(*) FROM (
            SELECT unnest(string_split({col}, '')) AS letter
            FROM {source} WHERE {col} IS NOT NULL)
        WHERE letter <> '' GROUP BY 1 ORDER BY 2 DESC""")
    return {str(letter): int(count) for letter, count in rows}
