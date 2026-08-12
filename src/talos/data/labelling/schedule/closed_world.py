"""Benign by absence — and saying so on every row.

Anything unmatched inside a capture that *has* a schedule is benign. That
assumption is what makes the approach tractable at all: one enumerates the small
set of attacks rather than the vast set of normal behaviour.
"""

from __future__ import annotations

from talos.data.labelling.base import CLOSED_WORLD, MATCHED as MATCHED_SOURCE
from talos.data.labelling.schedule.join import MATCHED, QUARANTINED
from talos.data.labelling.space import OUT_OF_SPACE
from talos.data.labelling.taxonomy import BENIGN

CERTAIN = "certain"
UNCERTAIN = "uncertain"


class ClosedWorldResolver:
    """Emits the label columns that exist because of the closed-world reading."""

    def __init__(self, matched: str = MATCHED, quarantined: str = QUARANTINED):
        self.matched = matched
        self.quarantined = quarantined

    def label_binary_sql(self) -> str:
        return f"{self.matched}.rid IS NOT NULL"

    def label_class_sql(self, benign: str = BENIGN) -> str:
        return f"coalesce({self.matched}.rclass, '{benign}')"

    def label_source_sql(self) -> str:
        """`matched` or `closed-world` -- the shared vocabulary, not a private one.

        These used to be `schedule` and `schedule-closed-world`, which repeated
        the method name in a column that sits beside `label_method`. The
        benchmark groups on this, so the values have to mean the same thing
        across methods: `matched` is a positive identification, `closed-world` is
        the absence of one.
        """
        return (f"CASE WHEN {self.matched}.rid IS NOT NULL THEN '{MATCHED_SOURCE}'\n"
                f"                        ELSE '{CLOSED_WORLD}' END")

    def rules_matched_sql(self) -> str:
        return f"coalesce({self.matched}.nrules, 0)"

    def quality_sql(self, has_regions: bool, out_of_space: str | None = None) -> str:
        """`label_quality` and `quarantine_reason`.

        `out_of_space` is a SQL expression yielding an exclusion reason or NULL,
        generated from the label space. The two flags mean different things --
        `uncertain` says the label may be wrong, `out-of-space` says the label is
        fine and this study cannot use the class -- and a flow can be both.

        `label_quality` gives out-of-space PRECEDENCE, because it excludes the
        flow from every pool while `uncertain` only marks it.

        `quarantine_reason` resolves the tie the OTHER way, and that asymmetry is
        deliberate. An out-of-space reason is recoverable from the row: take
        `label_class`, look it up in the label space, get the reason back. A
        region's reason is not recoverable from anything on the row. So the
        column keeps the irrecoverable one and nothing is lost either way.

        A quarantined flow KEEPS the label the schedule gave it and is flagged
        beside it, rather than getting a third `label_class` value. Every
        invariant tying label_binary, label_class, rule_id and label_source
        together still holds; code that has never heard of quarantine reads the
        closed-world label unchanged, and code that has can filter on one column.

        Without declared regions the columns are still emitted, as constants, so
        the output schema does not depend on whether a manifest happens to
        declare any — a downstream reader must not have to branch on that.
        """
        region = f"{self.quarantined}.qreason" if has_regions else None
        if out_of_space is None and region is None:
            return (f"'{CERTAIN}' AS label_quality,\n"
                    f"             CAST(NULL AS VARCHAR) AS quarantine_reason")

        cases = []                       # most-excluding first
        if out_of_space is not None:
            cases.append(f"WHEN ({out_of_space}) IS NOT NULL THEN '{OUT_OF_SPACE}'")
        if region is not None:
            cases.append(f"WHEN {region} IS NOT NULL THEN '{UNCERTAIN}'")
        quality = f"CASE {' '.join(cases)} ELSE '{CERTAIN}' END"

        # Irrecoverable first: the region reason, then the derivable one.
        reasons = [r for r in (region, f"({out_of_space})"
                               if out_of_space is not None else None) if r]
        reason = reasons[0] if len(reasons) == 1 else f"coalesce({', '.join(reasons)})"
        return (f"{quality} AS label_quality,\n"
                f"             {reason} AS quarantine_reason")
