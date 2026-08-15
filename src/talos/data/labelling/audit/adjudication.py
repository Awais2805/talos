"""The table a person fills in, and the file they hand back.

Carries what the machine believes and what an independent detector saw, so the
adjudicator has evidence rather than a bare flow. The verdict column starts
empty and stays that way until a person types in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from talos.common.provenance import Sha12, content_sha
from talos.data.labelling.audit.candidates import AuditError

#: What an adjudicator may write. `unknown` is not a cop-out: a flow a person
#: cannot classify from the evidence is a real outcome, and recording it is more
#: useful than an unexamined guess.
UNKNOWN = "unknown"


def default_path(cfg, dataset: str) -> Path:
    """Where `talos audit emit` writes and everything else reads from.

    One definition, used by the CLI and by anything gating on adjudication
    (e.g. behavioural fine-tuning), so the two cannot silently drift apart.
    """
    return cfg.reports / "audit" / f"{dataset}.csv"

#: Rows an adjudicator has not reached yet.
UNADJUDICATED = "unadjudicated"
CONFIRMED = "confirmed"
CORRECTED = "corrected"

#: The emitted column order. Evidence first, verdict last, so the columns a
#: person types into are at the right-hand edge of a spreadsheet.
COLUMNS = (
    "uid", "ts", "capture", "dataset",
    "prior_method", "prior_class", "prior_source", "prior_quality",
    "oracle_alerted", "oracle_signature",
    "analyst_class", "analyst_note",
)


@dataclass(frozen=True)
class Adjudication:
    """A completed (or partly completed) audit file."""

    path: Path
    sha: Sha12
    rows: int
    adjudicated: int
    confirmed: int
    corrected: int
    per_class: tuple[tuple[str, int], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def outstanding(self) -> int:
        return self.rows - self.adjudicated

    def describe(self) -> str:
        lines = [f"adjudication  {self.path.name}  sha {self.sha}",
                 f"  rows        {self.rows:,} "
                 f"({self.adjudicated:,} adjudicated, {self.outstanding:,} outstanding)"]
        if self.adjudicated:
            lines.append(f"  verdicts    {self.confirmed:,} confirmed, "
                         f"{self.corrected:,} corrected")
        return "\n".join(lines)


class AdjudicationTable:
    """Emits candidates for a person, and reads their decisions back."""

    def __init__(self, space, prior_method: str = "schedule"):
        self.space = space
        self.prior_method = prior_method

    # ------------------------------------------------------------------ emit

    def select_list(self, oracle: bool, flow_columns: Sequence[str] = ()) -> str:
        """The flow itself, then the evidence, then the two a person types into.

        The flow's own columns are carried because nobody can adjudicate a `uid`.
        `alerted` comes from the oracle's join rather than being inferred from a
        non-null signature: "no alert" and "alerted, signature not recorded" are
        different facts.
        """
        alerted = "coalesce(o.alerted, false)" if oracle else "CAST(NULL AS BOOLEAN)"
        signature = "o.signature" if oracle else "CAST(NULL AS VARCHAR)"
        flow = "".join(f'       c."{column}",\n' for column in flow_columns)
        return (f"c.uid, c.ts, c.capture, c.dataset,\n{flow}"
                f"       c.label_method AS prior_method,\n"
                f"       c.label_class  AS prior_class,\n"
                f"       c.label_source AS prior_source,\n"
                f"       c.label_quality AS prior_quality,\n"
                f"       {alerted} AS oracle_alerted,\n"
                f"       {signature} AS oracle_signature,\n"
                f"       CAST(NULL AS VARCHAR) AS analyst_class,\n"
                f"       CAST(NULL AS VARCHAR) AS analyst_note")

    def flow_columns(self, candidates) -> tuple[str, ...]:
        """What the candidate table holds that a person could judge a flow by.

        Everything that is not a label, not provenance and not already emitted —
        so a richer extractor's extra columns travel without being named here.
        """
        from talos.data.labelling.behavioural.base import labelling_columns
        skip = labelling_columns() | {"uid", "ts", "capture", "dataset", "__rank"}
        return tuple(c for c in candidates.columns if c not in skip)

    def emit(self, duck, candidates, path: Path, alerts: str | None = None) -> Path:
        """Write the audit file. `alerts` is SQL yielding (uid, alerted, signature).

        That shape is `Corroborator.join_sql`'s output, deliberately: one
        definition of "did an alert land on this flow", not a second one here
        that could drift from the rate the oracle reports.
        """
        join = ""
        if alerts:
            # LEFT JOIN: a flow with no alert is evidence too, and dropping it
            # would quietly restrict the audit to what Suricata already found.
            join = (f"\n      LEFT JOIN ({alerts}) o ON o.uid = c.uid")
        query = (f"SELECT {self.select_list(bool(alerts), self.flow_columns(candidates))}\n"
                 f"      FROM ({candidates.sql_query()}) c{join}")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = "(HEADER, DELIMITER ',')" if path.suffix == ".csv" else "(FORMAT PARQUET)"
        duck.sql(f"COPY ({query}) TO '{path}' {writer}")
        return path

    # ---------------------------------------------------------------- ingest

    def read(self, duck, path: Path):
        """The adjudication file as a relation, refusing the ways it can be wrong."""
        path = Path(path)
        if not path.exists():
            raise AuditError(f"no adjudication file at {path}")
        reader = ("read_csv_auto('{}', header = true, all_varchar = true)"
                  if path.suffix == ".csv" else "read_parquet('{}')")
        rel = duck.relation(f"SELECT * FROM {reader.format(path)}")

        missing = [c for c in ("uid", "prior_class", "analyst_class")
                   if c not in rel.columns]
        if missing:
            raise AuditError(
                f"{path} has no {', '.join(missing)}. It does not look like an "
                f"audit file emitted by `talos audit emit`.")
        self._validate(duck, rel, path)
        return rel

    def _validate(self, duck, rel, path: Path) -> None:
        """Three refusals, each for a failure that would otherwise be silent."""
        duplicates = duck.sql(
            f"SELECT uid, count(*) FROM ({rel.sql_query()}) a GROUP BY 1 "
            f"HAVING count(*) > 1 LIMIT 5")
        if duplicates:
            raise AuditError(
                f"{path}: uid(s) appear more than once "
                f"({', '.join(str(row[0]) for row in duplicates)}). Two verdicts "
                f"for one flow cannot both be the truth.")

        allowed = set(self.space.classes) | {UNKNOWN}
        unknown = duck.sql(
            f"SELECT DISTINCT analyst_class FROM ({rel.sql_query()}) a "
            f"WHERE analyst_class IS NOT NULL AND analyst_class <> '' "
            f"AND analyst_class NOT IN "
            f"({', '.join(chr(39) + c + chr(39) for c in sorted(allowed))})")
        if unknown:
            raise AuditError(
                f"{path}: analyst_class values outside label space "
                f"{self.space.name!r}: {', '.join(str(r[0]) for r in unknown)}. "
                f"Allowed: {', '.join(sorted(allowed))}. A typo would create a "
                f"class nothing downstream can score.")

    # ----------------------------------------------------------------- state

    def state_sql(self, alias: str = "a") -> str:
        """Unadjudicated, or the verdict against the prior. Never against a model.

        The state is deliberately blind to the oracle: Suricata's silence is not
        disagreement (it has the signatures it has), so folding it in would
        manufacture a verdict nobody gave.
        """
        return (f"CASE WHEN {alias}.analyst_class IS NULL "
                f"OR {alias}.analyst_class = '' THEN '{UNADJUDICATED}' "
                f"WHEN {alias}.analyst_class = {alias}.prior_class THEN '{CONFIRMED}' "
                f"ELSE '{CORRECTED}' END")

    def summarise(self, duck, rel, path: Path) -> Adjudication:
        counts = dict(duck.sql(
            f"SELECT {self.state_sql('a')} AS state, count(*) "
            f"FROM ({rel.sql_query()}) a GROUP BY 1"))
        per_class = tuple(duck.sql(
            f"SELECT analyst_class, count(*) FROM ({rel.sql_query()}) a "
            f"WHERE analyst_class IS NOT NULL AND analyst_class <> '' "
            f"GROUP BY 1 ORDER BY 2 DESC"))
        confirmed = int(counts.get(CONFIRMED, 0))
        corrected = int(counts.get(CORRECTED, 0))
        return Adjudication(
            path=Path(path), sha=content_sha(path),
            rows=sum(int(v) for v in counts.values()),
            adjudicated=confirmed + corrected, confirmed=confirmed,
            corrected=corrected, per_class=per_class)
