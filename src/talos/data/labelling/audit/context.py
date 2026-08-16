"""Neighbouring flows for a candidate -- the campaign a single row can't show.

A stratified one-row-per-class sample answers "what did the schedule/method say
about this flow" but not "is this one of two hundred near-identical connections
in a burst, or a genuine one-off" -- exactly the distinction a brute-force or
DoS verdict often turns on. This fetches, per candidate, the other flows between
the SAME two hosts (either direction, same capture) within a time window around
it -- the same host-pair-plus-time-range shape `Corroborator` already uses to
match an alert to a flow.

Written as two directional equi-joins UNIONed together, not one join with an
OR across the two directions -- `Corroborator.join_sql` hit exactly this
shape on the real 2017 join (2.1M flows) and it turned an O(n) hash join into
an unfinishable nested loop. Candidates here are a couple hundred rows, so the
mistake would be cheap this time, but there is no reason to make it twice.
"""

from __future__ import annotations

#: Seconds either side of a candidate's own timestamp. Ten minutes comfortably
#: spans the shortest CIC windows (PortScan sub-windows, DoS bursts) without
#: reaching into an unrelated part of the day for two hosts that talk rarely.
DEFAULT_WINDOW = 600.0

#: Neighbours shown per candidate, nearest in time first. A chatty pair (the
#: FTP-Patator brute force is ~700 connections in an hour) would otherwise
#: make one candidate's context as large as the whole audit.
DEFAULT_LIMIT = 20


class ContextFetcher:
    """Finds the flows around a candidate, without re-running the whole join
    once per row -- everything below is one query over every candidate."""

    def __init__(self, window: float = DEFAULT_WINDOW, limit: int = DEFAULT_LIMIT):
        self.window = float(window)
        self.limit = int(limit)

    def sql(self, sources: list[str], candidates_sql: str) -> str:
        listed = ", ".join(f"'{uri}'" for uri in sources)
        w = self.window
        return f"""
      WITH flows AS (
        SELECT uid, ts, capture, "id.orig_h" AS orig, "id.resp_h" AS resp,
               "id.orig_p" AS orig_p, "id.resp_p" AS resp_p, proto, service,
               duration, orig_bytes, resp_bytes, conn_state
        FROM read_parquet([{listed}], union_by_name = true)
      ),
      candidates AS (
        SELECT uid AS candidate_uid, ts AS candidate_ts, capture,
               "id.orig_h" AS orig, "id.resp_h" AS resp
        FROM ({candidates_sql}) c
      ),
      matched AS (
        SELECT c.candidate_uid, c.candidate_ts, f.*
        FROM candidates c JOIN flows f
          ON f.capture = c.capture AND f.orig = c.orig AND f.resp = c.resp
         AND f.ts BETWEEN c.candidate_ts - {w} AND c.candidate_ts + {w}
         AND f.uid != c.candidate_uid
        UNION ALL
        SELECT c.candidate_uid, c.candidate_ts, f.*
        FROM candidates c JOIN flows f
          ON f.capture = c.capture AND f.orig = c.resp AND f.resp = c.orig
         AND f.ts BETWEEN c.candidate_ts - {w} AND c.candidate_ts + {w}
         AND f.uid != c.candidate_uid
         -- same guard as Corroborator: a candidate whose own orig == resp
         -- would otherwise match twice, once per branch, for one real flow.
         AND c.orig != c.resp
      ),
      ranked AS (
        SELECT *, row_number() OVER (
          PARTITION BY candidate_uid ORDER BY abs(ts - candidate_ts)
        ) AS __rank
        FROM matched
      )
      SELECT candidate_uid, uid, CAST(ts AS DOUBLE) AS ts, orig, resp,
             CAST(orig_p AS BIGINT) AS orig_p, CAST(resp_p AS BIGINT) AS resp_p,
             proto, service, CAST(duration AS DOUBLE) AS duration,
             CAST(orig_bytes AS BIGINT) AS orig_bytes,
             CAST(resp_bytes AS BIGINT) AS resp_bytes, conn_state
      FROM ranked WHERE __rank <= {self.limit}
      ORDER BY candidate_uid, __rank"""

    def fetch(self, duck, sources: list[str], candidates) -> dict[str, list[dict]]:
        """`candidates` is a relation with at least uid/ts/capture/orig/resp.

        Returns candidate uid -> its neighbours, nearest in time first. A
        candidate with none is present with an empty list, not absent --
        "no other flows in this window" is itself something the page should
        say, not silently omit.
        """
        rows = duck.sql(self.sql(sources, candidates.sql_query()))
        cols = ("candidate_uid", "uid", "ts", "orig", "resp", "orig_p", "resp_p",
                "proto", "service", "duration", "orig_bytes", "resp_bytes",
                "conn_state")
        context: dict[str, list[dict]] = {uid: [] for uid in
                                          _candidate_uids(duck, candidates)}
        for row in rows:
            record = dict(zip(cols, row))
            context.setdefault(record["candidate_uid"], []).append(
                {k: v for k, v in record.items() if k != "candidate_uid"})
        return context


def _candidate_uids(duck, candidates) -> list[str]:
    return [row[0] for row in
            duck.sql(f"SELECT uid FROM ({candidates.sql_query()}) c")]
