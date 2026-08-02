#!/usr/bin/env python3
"""Tier 4 -- contradictions the labels contain regardless of the schedule.

Tiers 0-3 judge the labels against the document that produced them and against
outside evidence. This tier asks something narrower and harder to argue with:
taking the labels entirely at face value, do any of them contradict each other?

There is exactly one check here, and that is deliberate. An earlier version of
this module also estimated label noise by cross-validated modelling -- k-nearest
neighbour disagreement, confident learning, margin ranking. Those were removed.
They are estimates whose validity rests on assumptions this data violates
(confident learning assumes class-conditional noise; schedule-derived noise
depends on a flow's own features, such as whether it straddled a window edge),
and for 2017 and 2018 the external oracle answers the same question with
evidence rather than inference. Keeping a thousand lines of uncalibrated
estimator alongside a measurement would have been worse than not having it.

What survives proves rather than estimates, needs no model, no sample and no
seed, and returns the same answer every time it runs.
"""

from src.label_validation.core.finding import Finding, Severity
from src.label_validation.core.registry import check

# Identity, timing and provenance: the columns that say WHICH flow this is
# rather than what it looked like on the wire. A key built from them would make
# every flow unique and the check would find nothing by construction.
NON_FEATURE = {
    "uid", "id.orig_h", "id.resp_h", "ts", "capture", "dataset",
    "label_binary", "label_class", "label_raw", "rule_id", "label_executed",
    "label_source", "rules_matched", "manifest_sha", "taxonomy_sha",
    "labelled_at", "label_quality", "quarantine_reason",
}

TMP_CONFLICT = "con_duplicate_conflict"


@check(id="con.duplicate_conflict", tier=4,
       title="identical flows never carry different classes",
       needs=("labelled",), max_severity=Severity.MAJOR)
def duplicate_conflict(ctx):
    """Two flows Zeek recorded identically, labelled two different ways.

    The grouping key is every observable conn column -- ports, protocol,
    service, state, history, byte and packet counters -- with identity, timing
    and the label columns removed, so a group is a set of flows that no
    feature-based model could ever tell apart. If such a group carries two
    classes then at least the smaller side is wrong, whatever the schedule says.

    That count is a hard floor under the label error rate, and a floor no
    modelling can be repaired around: the rows are identical inputs with
    different targets, so the loss is irreducible.

    The key deliberately keeps id.orig_p, which varies per connection. That
    makes the check conservative -- flows differing only in an ephemeral source
    port count as distinct and are never flagged -- so every group it reports is
    a real contradiction rather than a plausible one.
    """
    cols = [c for c in ctx.columns if c not in NON_FEATURE]
    if not cols:
        return []
    src = ctx.q("labelled")
    keys = ", ".join(f'"{c}" AS k{i}' for i, c in enumerate(cols))
    names = ", ".join(f"k{i}" for i in range(len(cols)))
    build = f"""CREATE OR REPLACE TEMP TABLE {TMP_CONFLICT} AS
WITH vec AS (
    SELECT {keys}, label_class, count(*) AS n
    FROM read_parquet('{src}', union_by_name=true)
    GROUP BY ALL
), grp AS (
    SELECT *,
           sum(n)   OVER w AS grp_flows,
           count(*) OVER w AS grp_classes,
           row_number() OVER (PARTITION BY {names}
                              ORDER BY n DESC, label_class) AS rk
    FROM vec WINDOW w AS (PARTITION BY {names})
)
SELECT * FROM grp WHERE grp_classes > 1"""

    total = ctx.lake.one(f"SELECT count(*) FROM read_parquet('{src}', "
                         f"union_by_name=true)")[0]
    # Materialised once: the group-by over every observable column is the whole
    # cost of the check, and the three readings below are cheap against the
    # conflicting groups alone.
    ctx.lake.sql(build)
    try:
        groups, flows, minority, classes = ctx.lake.one(f"""
            SELECT count(*) FILTER (rk = 1), coalesce(sum(n), 0),
                   coalesce(sum(n) FILTER (rk > 1), 0),
                   count(DISTINCT label_class)
            FROM {TMP_CONFLICT}""")
        if not groups:
            return []
        per_class = ctx.lake.sql(f"""
            SELECT label_class, sum(n) AS flows,
                   coalesce(sum(n) FILTER (rk > 1), 0) AS minority
            FROM {TMP_CONFLICT} GROUP BY 1 ORDER BY minority DESC, flows DESC""")
        worst = ctx.lake.sql(f"""
            SELECT {names}, any_value(grp_flows) AS flows,
                   any_value(grp_flows) - max(n) AS minority,
                   string_agg(label_class || ' x' || n::VARCHAR, ', '
                              ORDER BY n DESC) AS split
            FROM {TMP_CONFLICT} GROUP BY {names}
            ORDER BY minority DESC, flows DESC LIMIT 20""")
    finally:
        ctx.lake.sql(f"DROP TABLE IF EXISTS {TMP_CONFLICT}")

    rate = minority / total if total else 0.0
    limit = ctx.threshold("duplicate_conflict_rate_max", 0.001)
    evidence = []
    for r in worst:
        row = {c: r[i] for i, c in enumerate(cols)}
        row.update(flows=int(r[-3]), minority_flows=int(r[-2]), split=r[-1])
        evidence.append(row)
    return [Finding(
        check_id="con.duplicate_conflict", dataset=ctx.dataset,
        severity=Severity.MAJOR if rate > limit else Severity.MINOR,
        title=f"{int(minority):,} flow(s) are provably mislabelled "
              f"({rate:.4%}): identical flows carrying different classes",
        detail=f"{int(groups):,} distinct feature vectors appear in the labelled zone "
               f"under more than one class. The vectors are built from every "
               f"observable conn column with identity and timing removed, so the "
               f"flows in a group are indistinguishable to any model. Taking the "
               f"majority label of each group as correct, {int(minority):,} flows are "
               f"labelled wrongly -- a lower bound, since a whole group could be "
               f"wrong together. Empty and near-empty flows dominate these groups "
               f"in practice, which is the signature of a time-window label "
               f"catching connection attempts that carry no attack.",
        metrics={"rows": int(total), "conflicting_groups": int(groups),
                 "flows_in_conflicting_groups": int(flows),
                 "minority_flows": int(minority),
                 "minority_rate": round(rate, 9),
                 "threshold": limit, "classes_involved": int(classes),
                 "key_columns": len(cols),
                 **{f"minority_{r[0]}": int(r[2]) for r in per_class},
                 **{f"in_conflict_{r[0]}": int(r[1]) for r in per_class}},
        evidence=evidence,
        scope={"key_columns": cols},
        repro=build.replace(f"CREATE OR REPLACE TEMP TABLE {TMP_CONFLICT} AS\n", ""),
    )]
