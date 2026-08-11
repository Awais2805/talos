"""The matching semantics. Everything about *which* flows an attack window claims.

A flow is an attack when its start time falls inside a window **and** its
endpoints match that window's attacker/victim pair. The endpoint half matters as
much as the temporal half: victims keep serving legitimate traffic throughout an
attack, so a pure time filter poisons the benign class with ordinary traffic, and
the resulting label noise is not random — it is structured and concentrated on
precisely the hosts under study.

Four refinements the data forced, each of which was a bug before it was a rule:

  bidirectional   a flow counts attacker->victim or victim->attacker, since
                  response traffic is part of the attack. (The datasets' own
                  labelling is direction-dependent, which published audits
                  identify as a source of mislabelled flows.)
  alias sets      one host appears under several addresses depending on capture
                  point. 2017's attacker sits behind NAT and appears internally
                  as 172.16.0.1; without that alias one rule matched zero flows
                  while 94,614 sat inside its window.
  victim-only     2019's reflection attacks spoof their sources, so the attacker
                  address carries no information and matching is on the victim.
  subnet victims  2017's infiltrated host scans "all other clients", a /24.

This module generates SQL and holds no state. It is the single place matching is
expressed, so quarantine regions run through the identical predicate rather than
a second implementation that could drift from it.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from talos.data.labelling.windows import Window

# Zeek's connection backbone. Named here rather than inline so an extractor with
# a different flow schema has one place to look. (Capability-guarding these for
# a swappable extractor is wave 2; today they are Zeek's.)
TS = "ts"
ORIG = "id.orig_h"
RESP = "id.resp_h"
UID = "uid"

# CTE aliases the generated SQL binds. Shared, because `closed_world` writes
# expressions against them and `writer` writes the FROM clause that creates
# them -- two hardcoded strings that must agree, and would otherwise fail only
# at execute time.
MATCHED = "m"
QUARANTINED = "qn"

RULE_COLUMNS = ("rid", "rname", "rclass", "t0", "t1", "atk", "vic", "vpfx", "needs_payload")


def _text(value) -> str:
    """A string as a SQL literal, or a typed NULL.

    Typed, because a column that is NULL for *every* row of an inline VALUES
    table binds as INTEGER otherwise — which is exactly what happens to the
    attacker column on a reflection-only dataset like 2019, and it fails at
    `list_contains` with a type error rather than anywhere informative.
    """
    if value is None:
        return "CAST(NULL AS VARCHAR)"
    return "'" + str(value).replace("'", "''") + "'"


def _text_list(values: Sequence[str] | None) -> str:
    if not values:
        return "CAST(NULL AS VARCHAR[])"
    return "[" + ", ".join(_text(v) for v in values) + "]"


class IntervalJoinBuilder:
    """Builds the SQL that decides which flows each window claims."""

    def __init__(self, ts: str = TS, orig: str = ORIG, resp: str = RESP, uid: str = UID):
        self.ts, self.orig, self.resp, self.uid = ts, orig, resp, uid

    # ------------------------------------------------------------ rules table

    def rules_table(self, windows: Iterable[Window]) -> str:
        """Windows as an inline VALUES table.

        Attack rules and quarantine regions emit the *same* column set — a
        region simply carries no class and never needs payload — so one
        predicate serves both. `rclass` doubles as the quarantine reason,
        because that is what a quarantined row groups on downstream.
        """
        rows = []
        for w in windows:
            rows.append(
                f"({w.rid}, {_text(w.name)}, {_text(getattr(w, 'canonical', w.name))}, "
                f"{w.t0!r}, {w.t1!r}, {_text_list(w.attackers)}, {_text_list(w.victims)}, "
                f"{_text(w.subnet_prefix)}, "
                f"{'true' if getattr(w, 'needs_payload', False) else 'false'})")
        if not rows:
            raise ValueError("no windows: an empty rules table would match nothing "
                             "and silently label the whole capture benign")
        return ("SELECT * FROM (VALUES\n    " + ",\n    ".join(rows) +
                f"\n  ) AS r({', '.join(RULE_COLUMNS)})")

    # ------------------------------------------------------------- predicates

    def victim_predicate(self, col: str) -> str:
        """Does `col` satisfy this rule's victim constraint?

        Three cases, and the first is the one that bites. A victim constraint
        may be an IP list, a subnet prefix, or *genuinely absent* — absent
        meaning "any peer". Absence must be established from both fields: a
        subnet rule has no `vic` list, and reading that alone as "no victim
        constraint" once made a rule match every flow in its window.
        """
        return (f'(\n        (r.vic IS NULL AND r.vpfx IS NULL)\n'
                f'        OR coalesce(list_contains(r.vic, c."{col}"), false)\n'
                f'        OR coalesce(starts_with(c."{col}", r.vpfx), false)\n'
                f'      )')

    def match_predicate(self) -> str:
        """The join condition: inside the window AND endpoints match.

        The end is exclusive (`ts < t1`) because windows are extended to the end
        of their final minute and clamped at the next window's start; a closed
        end would let two back-to-back attacks both claim the boundary instant.
        """
        return f"""
  c."{self.ts}" >= r.t0 AND c."{self.ts}" < r.t1
  AND (
    -- attacker known: the flow must run between attacker and victim, either way
    (r.atk IS NOT NULL AND (
        (list_contains(r.atk, c."{self.orig}") AND {self.victim_predicate(self.resp)})
     OR (list_contains(r.atk, c."{self.resp}") AND {self.victim_predicate(self.orig)})
    ))
    -- reflection: sources are spoofed, so the victim alone carries the signal
    OR (r.atk IS NULL AND r.vic IS NOT NULL
        AND (list_contains(r.vic, c."{self.orig}") OR list_contains(r.vic, c."{self.resp}")))
  )
"""

    # ------------------------------------------------------------------ CTEs

    def conn_cte(self, src: str, base: str) -> str:
        """Every conn flow, with its originating capture recovered from the path.

        The parquet zone mirrors the extracted tree one file per log, so the
        capture folder survives in `filename`. `capture` is what makes
        leakage-safe splitting possible later: attacks are contiguous bursts, so
        a random flow-level split puts the same flood on both sides of a
        train/test boundary and reports a fictional score.
        """
        return f"""conn AS (
        SELECT * EXCLUDE (filename),
               split_part(replace(filename, {_text(base)}, ''), '/', 1) AS capture
        FROM read_parquet({_text(src)}, filename = true, union_by_name = true)
      )"""

    def hits_cte(self, name: str, rules_cte: str, id_alias: str = "rid") -> str:
        """One winning window per flow, plus how many claimed it.

        `GROUP BY uid` with `min(rid)`, never `row_number() OVER (PARTITION BY
        uid)`. The window function gives the same answer but forces a sort over
        every matched row, which spilled 5+ GB on 2019 where ~70M of 72.5M flows
        match. This is a hash aggregate.

        Lowest rule id wins. That makes overlap resolution deterministic and
        independent of scan order -- a re-run cannot silently reassign a class --
        but it resolves by MANIFEST ORDER, not by specificity: a broad rule
        written earlier outranks a precise one written later. Determinism is the
        requirement; specificity would need a ranking nobody has defined (is
        "two attackers, one victim" narrower than "one attacker, a /24"?).

        Currently theoretical. Across all three datasets exactly one pair of
        windows overlaps -- 2017's Infiltration and Infiltration-PortScan, both
        15:04-15:45 -- and their endpoint sets are disjoint, so no flow is
        claimed by both. `rules_matched` counts any that ever are, so the day
        this stops being theoretical is visible in the run report.
        """
        return f"""{name} AS (
        SELECT c."{self.uid}" AS {self.uid}, min(r.rid) AS {id_alias}, count(*) AS nrules
        FROM conn c JOIN {rules_cte} r ON {self.match_predicate()}
        GROUP BY c."{self.uid}"
      )"""

    # -------------------------------------------------------------- assembly

    def base_ctes(self, rules: Sequence[Window], src: str, base: str,
                  regions: Sequence[Window] = ()) -> str:
        """rules + conn + matched (+ quarantined), the shared head of every query.

        Emitting the quarantine CTEs only when regions are declared means a
        dataset without them produces byte-identical SQL to a build that predates
        the feature — which keeps that comparison available as a check.
        """
        parts = [
            f"rules AS ({self.rules_table(rules)})",
            self.conn_cte(src, base),
            self.hits_cte("hits", "rules"),
            """matched AS (
        SELECT h.uid, h.rid, h.nrules, r.rname, r.rclass, r.needs_payload
        FROM hits h JOIN rules r ON r.rid = h.rid
      )""",
        ]
        if regions:
            parts += [
                f"qrules AS ({self.rules_table(regions)})",
                self.hits_cte("qhits", "qrules", id_alias="qid"),
                """quarantined AS (
        SELECT h.uid, h.qid, r.rname AS qreason
        FROM qhits h JOIN qrules r ON r.rid = h.qid
      )""",
            ]
        return "WITH " + ",\n      ".join(parts)
