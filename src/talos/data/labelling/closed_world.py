"""Benign by absence — and saying so on every row.

Anything unmatched inside a capture that *has* a schedule is benign. That
assumption is what makes the approach tractable at all: one enumerates the small
set of attacks rather than the vast set of normal behaviour.
"""

from __future__ import annotations

from talos.data.labelling.join import MATCHED, QUARANTINED
from talos.data.labelling.taxonomy import BENIGN

SCHEDULE = "schedule"
CLOSED_WORLD = "schedule-closed-world"

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
        return (f"CASE WHEN {self.matched}.rid IS NOT NULL THEN '{SCHEDULE}'\n"
                f"                        ELSE '{CLOSED_WORLD}' END")

    def rules_matched_sql(self) -> str:
        return f"coalesce({self.matched}.nrules, 0)"

    def quality_sql(self, has_regions: bool) -> str:
        """`label_quality` and `quarantine_reason`.

        A quarantined flow KEEPS the label the schedule gave it and is flagged
        beside it, rather than getting a third `label_class` value. Every
        invariant tying label_binary, label_class, rule_id and label_source
        together still holds; code that has never heard of quarantine reads the
        closed-world label unchanged, and code that has can filter on one column.

        Without declared regions the columns are still emitted, as constants, so
        the output schema does not depend on whether a manifest happens to
        declare any — a downstream reader must not have to branch on that.
        """
        if not has_regions:
            return (f"'{CERTAIN}' AS label_quality,\n"
                    f"             CAST(NULL AS VARCHAR) AS quarantine_reason")
        q = self.quarantined
        return (f"CASE WHEN {q}.qreason IS NULL THEN '{CERTAIN}'\n"
                f"                  ELSE '{UNCERTAIN}' END AS label_quality,\n"
                f"             {q}.qreason AS quarantine_reason")
