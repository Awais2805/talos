"""Why Eq. 17 chose what it chose, counted rather than assumed.

Reads the two branch tables and reports which line of the rule decided each row,
plus the confidence distributions that drive it. Measurement only: nothing is
written, no model runs. Exists because `fused == one branch` is a statement
about the branches' confidence scales, not about the fusion code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Exactly Eq. 17's five lines. No sixth category: a category the equation does
#: not have would not describe what `fuse` actually did.
AGREED = "agreed"
FIRST_CONFIDENCE = "first-confidence"
SECOND_CONFIDENCE = "second-confidence"
FIRST_MARGIN = "first-margin"
SECOND_FALLBACK = "second-fallback"

DECISIONS = (AGREED, FIRST_CONFIDENCE, SECOND_CONFIDENCE, FIRST_MARGIN,
             SECOND_FALLBACK)

#: Which input a decision hands the label to. `agreed` hands it to neither.
WINNER = {
    AGREED: "both", FIRST_CONFIDENCE: "first", SECOND_CONFIDENCE: "second",
    FIRST_MARGIN: "first", SECOND_FALLBACK: "second",
}

#: Why a row reached the margin lines at all — counted across them, not a
#: decision of its own. A NULL confidence makes both `>` tests NULL, so Eq. 17
#: skips them and decides on margin alone: silently, and never as a tie.
TIED, NULL_CONFIDENCE = "tied", "null-confidence"

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def decision_sql(first: str = "a", second: str = "b") -> str:
    """Eq. 17's branch order, labelled instead of resolved.

    Deliberately identical in structure to `FusedLabeller.fusion_sql`, including
    its NULL behaviour: a breakdown that special-cases anything would describe a
    rule the pipeline does not run.
    """
    return (f"CASE\n"
            f"        WHEN {first}.label_class = {second}.label_class "
            f"THEN '{AGREED}'\n"
            f"        WHEN {first}.label_confidence > {second}.label_confidence "
            f"THEN '{FIRST_CONFIDENCE}'\n"
            f"        WHEN {second}.label_confidence > {first}.label_confidence "
            f"THEN '{SECOND_CONFIDENCE}'\n"
            f"        WHEN {first}.label_margin >= {second}.label_margin "
            f"THEN '{FIRST_MARGIN}'\n"
            f"        ELSE '{SECOND_FALLBACK}' END")


def margin_cause_sql(first: str = "a", second: str = "b") -> str:
    """Why the confidence lines did not settle it: an exact tie, or a NULL."""
    return (f"CASE\n"
            f"        WHEN {first}.label_confidence IS NULL "
            f"OR {second}.label_confidence IS NULL THEN '{NULL_CONFIDENCE}'\n"
            f"        ELSE '{TIED}' END")


@dataclass
class ExplainReport:
    """The decision breakdown, and the confidences that produced it."""

    dataset: str
    inputs: tuple[str, ...] = ()
    rows_first: int = 0
    rows_second: int = 0
    rows_joined: int = 0
    decisions: dict[str, int] = field(default_factory=dict)
    margin_causes: dict[str, int] = field(default_factory=dict)
    by_class: tuple = ()
    confidence: tuple = ()
    disagreement_by_class: tuple = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def null_confidence(self) -> int:
        """Rows where Eq. 17 skipped both confidence lines without saying so."""
        return self.margin_causes.get(NULL_CONFIDENCE, 0)

    @property
    def disagreements(self) -> int:
        return self.rows_joined - self.decisions.get(AGREED, 0)

    def share_of_disagreements(self, winner: str) -> float:
        """How often one input took a row the branches disagreed on."""
        if not self.disagreements:
            return 0.0
        won = sum(n for decision, n in self.decisions.items()
                  if WINNER[decision] == winner and decision != AGREED)
        return won / self.disagreements

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset, "inputs": list(self.inputs),
            "rows_first": self.rows_first, "rows_second": self.rows_second,
            "rows_joined": self.rows_joined,
            "unjoined_first": self.rows_first - self.rows_joined,
            "unjoined_second": self.rows_second - self.rows_joined,
            "decisions": {d: self.decisions.get(d, 0) for d in DECISIONS},
            "decision_shares": {
                d: (self.decisions.get(d, 0) / self.rows_joined
                    if self.rows_joined else 0.0) for d in DECISIONS},
            "margin_line_causes": {TIED: self.margin_causes.get(TIED, 0),
                                   NULL_CONFIDENCE: self.null_confidence},
            "disagreements": self.disagreements,
            "disagreements_won_by_first": self.share_of_disagreements("first"),
            "disagreements_won_by_second": self.share_of_disagreements("second"),
            "by_class": [dict(zip(("class", "decision", "flows"), row))
                         for row in self.by_class],
            "disagreement_by_class": [
                dict(zip(("first_class", "second_class", "flows"), row))
                for row in self.disagreement_by_class],
            "confidence": [
                dict(zip(("branch", "class", "flows", "mean", "quantiles"), row))
                for row in self.confidence],
            "note": ("decision counts describe which branch Eq. 17 trusted, not "
                     "which branch was right; neither branch is ground truth."),
            **self.extra,
        }

    def table(self) -> str:
        first = self.inputs[0] if self.inputs else "first"
        second = self.inputs[1] if len(self.inputs) > 1 else "second"
        names = {FIRST_CONFIDENCE: f"{first} won on confidence",
                 SECOND_CONFIDENCE: f"{second} won on confidence",
                 FIRST_MARGIN: f"{first} won on margin",
                 SECOND_FALLBACK: f"{second} took the deterministic fallback",
                 AGREED: "branches agreed"}
        lines = [f"{'decision':<44}{'flows':>14}{'share':>9}"]
        for decision in DECISIONS:
            count = self.decisions.get(decision, 0)
            share = count / self.rows_joined if self.rows_joined else 0.0
            lines.append(f"{names[decision]:<44}{count:>14,}{100 * share:>8.2f}%")
        lines += ["-" * 67,
                  f"{'TOTAL':<44}{self.rows_joined:>14,}{100.0:>8.2f}%"]
        return "\n".join(lines)

    def summary(self) -> str:
        first = self.inputs[0] if self.inputs else "first"
        second = self.inputs[1] if len(self.inputs) > 1 else "second"
        lines = [
            f"disagreements {self.disagreements:,} of {self.rows_joined:,} flows",
            f"  won by {first:<14}{100 * self.share_of_disagreements('first'):>7.2f}%",
            f"  won by {second:<14}{100 * self.share_of_disagreements('second'):>7.2f}%",
        ]
        if self.rows_joined < max(self.rows_first, self.rows_second):
            lines.append(f"!! inner join dropped rows: {self.rows_first:,} and "
                         f"{self.rows_second:,} in, {self.rows_joined:,} joined")
        margin_rows = (self.decisions.get(FIRST_MARGIN, 0)
                       + self.decisions.get(SECOND_FALLBACK, 0))
        if margin_rows:
            lines.append(f"margin lines decided {margin_rows:,} rows: "
                         f"{self.margin_causes.get(TIED, 0):,} an exact confidence "
                         f"tie, {self.null_confidence:,} a NULL confidence")
        if self.null_confidence:
            lines.append(f"!! {self.null_confidence:,} rows had a NULL confidence; "
                         f"Eq. 17 skips BOTH confidence lines and decides them on "
                         f"margin alone, without recording that it did")
        if self.confidence:
            lines += ["", f"{'branch':<12}{'class':<14}{'flows':>12}"
                          f"{'mean':>9}{'p10':>8}{'p50':>8}{'p90':>8}"]
            for branch, cls, n, mean, quantiles in self.confidence:
                p10, p50, p90 = quantiles[0], quantiles[4], quantiles[8]
                lines.append(f"{branch:<12}{cls:<14}{n:>12,}{mean:>9.4f}"
                             f"{p10:>8.4f}{p50:>8.4f}{p90:>8.4f}")
        return "\n".join(lines)


class FusionExplainer:
    """Counts Eq. 17's decisions over two already-written branch tables."""

    def __init__(self, inputs: tuple[str, ...]):
        self.inputs = tuple(inputs)

    def run(self, duck, dataset: str, uris) -> ExplainReport:
        """`uris` are the two branch tables, already resolved and checked."""
        report = ExplainReport(dataset=dataset, inputs=self.inputs)
        report.rows_first = self._count(duck, uris[0])
        report.rows_second = self._count(duck, uris[1])

        duck.sql(f"""CREATE OR REPLACE TEMP TABLE decisions AS
            SELECT a.label_class AS first_class, b.label_class AS second_class,
                   a.label_confidence AS first_confidence,
                   b.label_confidence AS second_confidence,
                   {decision_sql()} AS decision,
                   {margin_cause_sql()} AS margin_cause
            FROM read_parquet('{uris[0]}') a
            JOIN read_parquet('{uris[1]}') b ON a.uid = b.uid""")

        report.rows_joined = duck.one("SELECT count(*) FROM decisions")[0]
        report.decisions = dict(duck.sql(
            "SELECT decision, count(*) FROM decisions GROUP BY 1"))
        report.margin_causes = dict(duck.sql(
            f"SELECT margin_cause, count(*) FROM decisions "
            f"WHERE decision IN ('{FIRST_MARGIN}', '{SECOND_FALLBACK}') "
            f"GROUP BY 1"))
        report.by_class = tuple(duck.sql(
            "SELECT first_class, decision, count(*) FROM decisions "
            "GROUP BY 1, 2 ORDER BY 3 DESC"))
        report.disagreement_by_class = tuple(duck.sql(
            f"SELECT first_class, second_class, count(*) FROM decisions "
            f"WHERE decision <> '{AGREED}' GROUP BY 1, 2 ORDER BY 3 DESC"))
        report.confidence = self._confidence(duck)
        return report

    def _count(self, duck, uri: str) -> int:
        return duck.one(f"SELECT count(*) FROM read_parquet('{uri}')")[0]

    def _confidence(self, duck) -> tuple:
        """Per branch, per predicted class: how confident it actually was."""
        quantiles = ", ".join(str(q) for q in QUANTILES)
        rows: list = []
        for branch, name in zip(("first", "second"), self.inputs):
            rows += [(name, *row) for row in duck.sql(f"""
                SELECT {branch}_class, count(*), avg({branch}_confidence),
                       quantile_cont({branch}_confidence, [{quantiles}])
                FROM decisions WHERE {branch}_confidence IS NOT NULL
                GROUP BY 1 ORDER BY 2 DESC""")]
        return tuple(rows)
