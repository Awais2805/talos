"""Executed attacks versus mere attempts.

Matching on time and endpoints alone means a flow is called an attack *because
of when it happened*, without anything verifying that an attack occurred. Empty
and failed TCP connections between attacker and victim get counted as executed
attacks. Published audits of CIC-IDS-2017 identify this and propose the
distinction; we adopt it.

The distinction is only meaningful where the attacker must send data for the
attack to happen — brute force, web attacks, botnet C2, HTTP floods. That set is
declared per raw attack name in `taxonomy.yaml` and arrives here as the
`needs_payload` column of the rules table, so this module never decides which
attacks qualify. It only decides, for those that do, whether payload was sent.

Two things it must not get wrong:

**Zero is not unknown.** `orig_bytes = 0` is a confirmed empty connection.
`orig_bytes IS NULL` means Zeek could not determine the byte count. Conflating
them marks flows as attempts on the basis of missing measurement.

**Some attacks are legitimately empty.** Port scans, slow-DoS (which hold sockets
open by sending almost nothing), SYN floods and UDP reflection are deliberately
outside `requires_payload`. Marking those "attempted" would discard real attack
traffic — which is why the exempt set is a declared list in the taxonomy rather
than a condition in this file.

The immediate consequence of this test on real data: all 192,300 of 2018's FTP
brute-force flows have conn_state `REJ` — a SYN answered by a reset, one packet
each way, zero bytes — meaning the service refused every connection and the
dataset contains no executed FTP brute force at all.
"""

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
