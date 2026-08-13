"""Which flows a person is asked to look at.

Drawn from the partition's audit pool with a per-class floor, which a uniform
hash share cannot guarantee. Selection is a deterministic hash order, so one
declaration picks the same rows without storing a list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from talos.data.labelling.base import LabellingError

#: Rows per class a study asks for, when the declaration does not say.
DEFAULT_FLOOR = 200

#: Total cap. Every row here is eventually read by a person.
DEFAULT_CAP = 2000


class AuditError(LabellingError):
    pass


@dataclass(frozen=True)
class Shortfall:
    """A class the candidate pool could not supply enough of."""

    label: str
    wanted: int
    available: int

    def describe(self) -> str:
        return (f"{self.label:<14} wanted {self.wanted}, pool holds "
                f"{self.available}")


@dataclass(frozen=True)
class Selection:
    """The candidate set, and what it could not deliver."""

    rows: int
    per_class: tuple[tuple[str, int], ...]
    shortfalls: tuple[Shortfall, ...] = ()
    pool: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        lines = [f"{'class':<16}{'candidates':>12}"]
        lines += [f"{label:<16}{n:>12,}" for label, n in self.per_class]
        lines += ["-" * 28, f"{'TOTAL':<16}{self.rows:>12,}"]
        for shortfall in self.shortfalls:
            lines.append(f"!! short: {shortfall.describe()}")
        return "\n".join(lines)


class CandidateSelector:
    """Draws an auditable sample from a partition's audit pool."""

    def __init__(self, partition, pool: str = "d_s", floor: int = DEFAULT_FLOOR,
                 cap: int = DEFAULT_CAP):
        self.partition = partition
        self.pool = pool
        self.floor = int(floor)
        self.cap = int(cap)
        if self.pool not in self.partition:
            raise AuditError(
                f"partition {self.partition.name!r} declares no pool {self.pool!r}. "
                f"Have: {', '.join(p.name for p in self.partition)}")

    def order_sql(self, alias: str = "c") -> str:
        """Deterministic order within a class, from the partition's own seed.

        Not `random()`: the same declaration must select the same rows on any
        machine, or an adjudication cannot be matched to what produced it.
        """
        return self.partition[self.pool].bucket_sql(alias)

    def ranked(self, duck, sources, label_column: str = "label_class"):
        """Audit-pool rows, numbered within their class in hash order."""
        rel = self.partition.relation(duck, self.pool, sources, alias="c")
        return duck.relation(
            f"SELECT *, row_number() OVER (PARTITION BY {label_column} "
            f"ORDER BY {self.order_sql('c')}) AS __rank "
            f"FROM ({rel.sql_query()}) c")

    def select(self, duck, sources, label_column: str = "label_class"):
        """`(relation, Selection)` — the floor per class, capped, plus what was short.

        A class the pool cannot fill is REPORTED, never topped up from outside
        the pool: reaching into `d_l` for one more brute_force row would put an
        audited flow in the pretraining set, which is the one thing the
        partition exists to make impossible.
        """
        ranked = self.ranked(duck, sources, label_column)
        chosen = ranked.filter(f"__rank <= {self.floor}")

        available = dict(duck.sql(
            f"SELECT {label_column}, count(*) FROM ({ranked.sql_query()}) r "
            f"GROUP BY 1"))
        taken = dict(duck.sql(
            f"SELECT {label_column}, count(*) FROM ({chosen.sql_query()}) r "
            f"GROUP BY 1"))

        shortfalls = tuple(
            Shortfall(label, self.floor, int(available[label]))
            for label in sorted(available) if int(available[label]) < self.floor)
        total = sum(int(n) for n in taken.values())
        if total > self.cap:
            raise AuditError(
                f"{total:,} candidates exceeds the cap of {self.cap:,}. Every one "
                f"is read by a person: lower `floor`, or raise `cap` deliberately.")

        selection = Selection(
            rows=total,
            per_class=tuple(sorted(((k, int(v)) for k, v in taken.items()),
                                   key=lambda row: -row[1])),
            shortfalls=shortfalls, pool=f"{self.partition.name}/{self.pool}",
            extra={"floor": self.floor, "cap": self.cap})
        return chosen, selection
