"""Two rates, never one number.

Agreement is not available here even in principle, because the two sources
cannot disagree symmetrically. Suricata firing on a flow is positive evidence.
Suricata *not* firing is nothing at all — it has the signatures it has, and CIC's
captures contain attacks (slow-DoS, some infiltration behaviour) that no
signature set reliably catches. So this module publishes:

    corroboration rate   of schedule-attack flows, how many Suricata also flagged
    dissent rate         of schedule-benign flows, how many Suricata flagged anyway

The first is a floor on how much of the attack class is independently supported.
The second is a list of candidate missed attacks — the same thing the benign-burst
audit hunts for, arrived at by a completely different route.

Neither is an accuracy, and `CorroborationReport` deliberately exposes no
property that combines them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# An alert timestamp falls somewhere inside a flow's lifetime, but Zeek's flow
# start and Suricata's packet clock need not agree to the microsecond. A couple
# of seconds of slack absorbs that without letting an alert claim a neighbouring
# flow -- these captures are dense, so wider would start matching the wrong ones.
DEFAULT_TOLERANCE = 2.0


@dataclass(frozen=True)
class CorroborationReport:
    """What an independent detector said about each side of the schedule's call."""

    dataset: str
    attack_flows: int
    attack_corroborated: int
    benign_flows: int
    benign_alerted: int
    alerts_total: int
    alerts_joined: int
    top_signatures: tuple[tuple[str, int], ...] = ()
    dissent_examples: tuple[dict[str, Any], ...] = ()
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def corroboration_rate(self) -> float:
        """Of flows the schedule called attack, the share Suricata also flagged.

        A FLOOR on independent support, not a measure of correctness: the
        remainder is unexplained, not wrong.
        """
        return self.attack_corroborated / self.attack_flows if self.attack_flows else 0.0

    @property
    def dissent_rate(self) -> float:
        """Of flows the schedule called benign, the share Suricata flagged.

        Candidate missed attacks. Expected to be small and non-zero; a large
        value means a window is misplaced.
        """
        return self.benign_alerted / self.benign_flows if self.benign_flows else 0.0

    @property
    def unjoined_alerts(self) -> int:
        """Alerts that matched no flow at all -- a join-quality signal.

        Large means the 5-tuple or the clock does not line up, and neither rate
        above can be trusted until that is understood.
        """
        return self.alerts_total - self.alerts_joined

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "tolerance_seconds": self.tolerance,
            "attack": {"flows": self.attack_flows,
                       "corroborated": self.attack_corroborated,
                       "corroboration_rate": self.corroboration_rate},
            "benign": {"flows": self.benign_flows,
                       "alerted": self.benign_alerted,
                       "dissent_rate": self.dissent_rate},
            "alerts": {"total": self.alerts_total, "joined": self.alerts_joined,
                       "unjoined": self.unjoined_alerts},
            "top_signatures": [{"signature": s, "flows": n}
                               for s, n in self.top_signatures],
            "dissent_examples": list(self.dissent_examples),
            # Named so nobody adds one later and calls it accuracy.
            "note": ("no combined accuracy is reported: an alert corroborates an "
                     "attack label, but absence of an alert is not evidence of benign"),
        }

    def summary(self) -> str:
        return "\n".join([
            f"corroboration  {self.attack_corroborated:,} of {self.attack_flows:,} "
            f"attack flows ({100 * self.corroboration_rate:.1f}%) also alerted",
            f"dissent        {self.benign_alerted:,} of {self.benign_flows:,} "
            f"benign flows ({100 * self.dissent_rate:.2f}%) alerted — candidate misses",
            f"alerts         {self.alerts_joined:,} of {self.alerts_total:,} joined to a flow"
            + (f"  ({self.unjoined_alerts:,} unmatched)" if self.unjoined_alerts else ""),
        ])


class Corroborator:
    """Joins alerts onto a labelled table and counts the two rates."""

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE, top_k: int = 10,
                 examples: int = 5):
        self.tolerance = tolerance
        self.top_k = top_k
        self.examples = examples

    def join_sql(self, labelled_uri: str, alerts: str) -> str:
        """One row per flow, with whether any alert landed on it.

        Endpoints match either way round because Suricata reports a packet's
        direction and Zeek reports the connection's originator; for a mid-stream
        alert those need not agree.
        """
        tol = self.tolerance
        return f"""
      WITH alerts AS ({alerts}),
      flows AS (SELECT * FROM read_parquet('{labelled_uri}', union_by_name = true)),
      hit AS (
        SELECT f.uid,
               count(a.ts) AS alerts,
               any_value(a.signature) AS signature
        FROM flows f LEFT JOIN alerts a
          ON a.ts >= f.ts - {tol}
         AND a.ts <= f.ts + coalesce(f.duration, 0) + {tol}
         AND ((a.src_ip = f."id.orig_h" AND a.dest_ip = f."id.resp_h")
           OR (a.src_ip = f."id.resp_h" AND a.dest_ip = f."id.orig_h"))
        GROUP BY f.uid
      )
      SELECT f.uid, f.label_binary, f.label_class, f.ts,
             f."id.orig_h" AS orig, f."id.resp_h" AS resp,
             h.alerts > 0 AS alerted, h.signature
      FROM flows f JOIN hit h USING (uid)"""

    def build(self, duck, dataset: str, labelled_uri: str, alerts: str,
              alerts_total: int) -> CorroborationReport:
        """`alerts` is a SQL expression yielding (ts, src_ip, dest_ip, signature).

        A `read_json_auto(...)` or `read_parquet(...)` for anything real; an
        inline VALUES table is fine for a handful of rows in a test. It used to
        be VALUES unconditionally, which at 2019 scale meant a ~300 MB query
        string.
        """
        joined = f"({self.join_sql(labelled_uri, alerts)})"

        counts = duck.one(f"""
            SELECT count(*) FILTER (label_binary),
                   count(*) FILTER (label_binary AND alerted),
                   count(*) FILTER (NOT label_binary),
                   count(*) FILTER (NOT label_binary AND alerted),
                   count(*) FILTER (alerted)
            FROM {joined} t""")

        signatures = duck.sql(f"""
            SELECT signature, count(*) AS n FROM {joined} t
            WHERE alerted GROUP BY 1 ORDER BY 2 DESC LIMIT {self.top_k}""")

        dissent = duck.sql(f"""
            SELECT uid, ts, orig, resp, signature FROM {joined} t
            WHERE alerted AND NOT label_binary
            ORDER BY ts LIMIT {self.examples}""")

        return CorroborationReport(
            dataset=dataset,
            attack_flows=counts[0], attack_corroborated=counts[1],
            benign_flows=counts[2], benign_alerted=counts[3],
            alerts_total=alerts_total, alerts_joined=counts[4],
            top_signatures=tuple((s, n) for s, n in signatures),
            dissent_examples=tuple(
                dict(zip(("uid", "ts", "orig", "resp", "signature"), row))
                for row in dissent),
            tolerance=self.tolerance)
