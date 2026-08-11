"""Executed attacks versus mere attempts."""

from __future__ import annotations

from talos.data.labelling.join import MATCHED

# NULL means "not applicable, or not measurable" — never "no". A three-valued
# column is the honest shape: true, false, and we-cannot-say.
#
# CAST, not a bare NULL: an untyped NULL leaves DuckDB to infer the column type
# from nothing, so an extractor without byte counts would write label_executed
# as some other type than the BOOLEAN every other extractor produces. The output
# schema must not depend on which tool made the flows.
NOT_APPLICABLE = "CAST(NULL AS BOOLEAN)"


class ExecutionClassifier:
    """Emits the executed/attempted expression. Holds no state and no policy.

    `measurable` is an argument rather than an attribute for that second reason:
    whether byte counts exist is a property of the RUN, not of the expression
    generator, and it was briefly a mutable field the engine reached in and set.
    """

    def __init__(self, orig_bytes: str = "orig_bytes", matched: str = MATCHED):
        self.orig_bytes = orig_bytes
        self.matched = matched

    def classify_sql(self, conn: str = "c", measurable: bool = True) -> str:
        """`label_executed` as a SQL CASE expression.

        Order matters: not-an-attack and not-payload-bearing are ruled out
        first, then unmeasurable, and only then is zero read as a failed attempt.

        `orig_bytes` is the ORIGINATOR's payload, and the originator is whoever
        opened the connection -- normally the attacking party, including for
        botnet C2 where the infected host calls out. A flow where the victim
        opened the connection therefore measures the victim's bytes. Left as is
        because widening it to orig+resp would count a victim's reply as proof
        the attack executed, which is a weaker claim than the column makes.
        """
        if not measurable:
            return NOT_APPLICABLE
        m, ob = self.matched, f'{conn}."{self.orig_bytes}"'
        return (
            f"CASE WHEN {m}.rid IS NULL OR NOT {m}.needs_payload THEN {NOT_APPLICABLE}\n"
            f"                   WHEN {ob} IS NULL THEN {NOT_APPLICABLE}\n"
            f"                   WHEN {ob} > 0 THEN true\n"
            f"                   ELSE false END"
        )

    def executed_count_sql(self, conn: str = "c", measurable: bool = True) -> str:
        return f"sum(CASE WHEN {self.classify_sql(conn, measurable)} THEN 1 ELSE 0 END)"

    def attempted_count_sql(self, conn: str = "c", measurable: bool = True) -> str:
        """Counts only explicit false. A NULL is not an attempt."""
        return (f"sum(CASE WHEN ({self.classify_sql(conn, measurable)}) = false "
                f"THEN 1 ELSE 0 END)")
