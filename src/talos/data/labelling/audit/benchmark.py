"""Scoring methods against the one thing here that is ground truth.

A person decided the adjudicated slice, so this is ACCURACY, not agreement.
Reported per class and never pooled: the slice has a per-class floor, so its
class mix is not the population's and one figure would use the wrong priors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from talos.data.labelling.audit.adjudication import UNKNOWN

#: Rows below this per class are reported but should not be read as a rate.
THIN = 30


@dataclass(frozen=True)
class ClassScore:
    """One method's performance on one class of the audited slice."""

    label: str
    audited: int
    correct: int
    confused_with: tuple[tuple[str, int], ...] = ()

    @property
    def recall(self) -> float | None:
        """None, not 0.0, when nothing was audited — an unmeasured class."""
        return self.correct / self.audited if self.audited else None

    @property
    def thin(self) -> bool:
        return self.audited < THIN

    def describe(self) -> str:
        rate = "n/a" if self.recall is None else f"{100 * self.recall:6.2f}%"
        flag = "  (thin)" if self.thin and self.audited else ""
        worst = (f"  most confused with {self.confused_with[0][0]}"
                 if self.confused_with else "")
        return (f"{self.label:<14}{self.audited:>8,}{self.correct:>9,}"
                f"{rate:>10}{flag}{worst}")


@dataclass(frozen=True)
class MethodScore:
    """One method, over the whole audited slice."""

    method: str
    scored: int
    unmatched: int
    per_class: tuple[ClassScore, ...] = ()

    def describe(self) -> str:
        lines = [f"{self.method}  ({self.scored:,} audited flows scored"
                 + (f", {self.unmatched:,} not present in its table)"
                    if self.unmatched else ")"),
                 f"  {'class':<14}{'audited':>8}{'correct':>9}{'recall':>10}"]
        lines += [f"  {score.describe()}" for score in self.per_class]
        return "\n".join(lines)


@dataclass(frozen=True)
class BenchmarkReport:
    """Every method against the adjudicated slice, and against each other."""

    adjudication: str
    adjudicated: int
    methods: tuple[MethodScore, ...] = ()
    agreement: tuple[tuple[str, str, int, int], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    #: Why there is no single accuracy figure anywhere in this report.
    NOTE = ("Scored per class and not pooled: the audited slice is drawn with a "
            "per-class floor, so its class mix is not the population's.")

    def to_dict(self) -> dict:
        return {
            "adjudication": self.adjudication, "adjudicated": self.adjudicated,
            "methods": [
                {"method": m.method, "scored": m.scored, "unmatched": m.unmatched,
                 "per_class": [
                     {"class": c.label, "audited": c.audited, "correct": c.correct,
                      "recall": c.recall, "thin": c.thin,
                      "confused_with": [dict(zip(("class", "n"), pair))
                                        for pair in c.confused_with]}
                     for c in m.per_class]}
                for m in self.methods],
            "pairwise_agreement": [
                dict(zip(("a", "b", "agree", "of"), row)) for row in self.agreement],
            "note": self.NOTE, **self.extra,
        }

    def describe(self) -> str:
        lines = [f"adjudicated   {self.adjudicated:,} flows  "
                 f"({self.adjudication})", ""]
        lines += [score.describe() + "\n" for score in self.methods]
        if self.agreement:
            lines.append("pairwise agreement (NOT accuracy — neither side is truth)")
            for a, b, agree, total in self.agreement:
                rate = f"{100 * agree / total:.2f}%" if total else "n/a"
                lines.append(f"  {a} vs {b}: {agree:,}/{total:,}  {rate}")
        lines += ["", f"note: {self.NOTE}"]
        return "\n".join(lines)


class LabelBenchmark:
    """Compares labelled tables on the slice a person adjudicated."""

    def __init__(self, space):
        self.space = space

    def adjudicated_sql(self, adjudication) -> str:
        """Only rows with a verdict, and only verdicts that name a class.

        `unknown` is excluded from scoring and counted separately: a flow a
        person could not classify is not evidence that a method got it wrong.
        """
        return (f"SELECT uid, analyst_class FROM ({adjudication.sql_query()}) a "
                f"WHERE analyst_class IS NOT NULL AND analyst_class <> '' "
                f"AND analyst_class <> '{UNKNOWN}'")

    def score(self, duck, adjudication, method: str, uri: str) -> MethodScore:
        """One method against the verdicts."""
        truth = self.adjudicated_sql(adjudication)
        joined = (f"SELECT t.analyst_class, m.label_class FROM ({truth}) t "
                  f"LEFT JOIN read_parquet('{uri}') m ON m.uid = t.uid")

        rows = duck.sql(
            f"SELECT analyst_class, label_class, count(*) FROM ({joined}) j "
            f"GROUP BY 1, 2")
        unmatched = sum(int(n) for _t, predicted, n in rows if predicted is None)

        per_class: list[ClassScore] = []
        for label in self.space.classes:
            hits = [(predicted, int(n)) for truth_class, predicted, n in rows
                    if truth_class == label and predicted is not None]
            audited = sum(n for _p, n in hits)
            correct = sum(n for predicted, n in hits if predicted == label)
            confused = tuple(sorted(((p, n) for p, n in hits if p != label),
                                    key=lambda pair: -pair[1]))
            per_class.append(ClassScore(label, audited, correct, confused))

        scored = sum(score.audited for score in per_class)
        return MethodScore(method=method, scored=scored, unmatched=unmatched,
                           per_class=tuple(per_class))

    def agreement(self, duck, tables: Mapping[str, str]) -> tuple:
        """How often two methods say the same thing. Agreement, and labelled as such.

        Kept separate from the scores above and named differently on purpose:
        the post-mortem's central failure was reporting one of these as the other.
        """
        names = sorted(tables)
        out = []
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                row = duck.one(
                    f"SELECT sum(CASE WHEN a.label_class = b.label_class THEN 1 "
                    f"ELSE 0 END), count(*) "
                    f"FROM read_parquet('{tables[first]}') a "
                    f"JOIN read_parquet('{tables[second]}') b ON a.uid = b.uid")
                out.append((first, second, int(row[0] or 0), int(row[1] or 0)))
        return tuple(out)

    def run(self, duck, adjudication, tables: Mapping[str, str],
            name: str = "") -> BenchmarkReport:
        adjudicated = duck.one(
            f"SELECT count(*) FROM ({self.adjudicated_sql(adjudication)}) t")[0]
        if not adjudicated:
            raise ValueError(
                "no adjudicated rows: every analyst_class is empty or `unknown`. "
                "There is nothing here that could score a method.")
        return BenchmarkReport(
            adjudication=name, adjudicated=int(adjudicated),
            methods=tuple(self.score(duck, adjudication, method, uri)
                          for method, uri in sorted(tables.items())),
            agreement=self.agreement(duck, tables),
            extra={"label_space": f"{self.space.name} {self.space.sha}"})
