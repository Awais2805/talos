#!/usr/bin/env python3
"""Tier 3 -- the external oracle: a second opinion produced outside this project.

Everything else in this module judges Talos against Talos. The integrity tier
asks whether the labelled zone is consistent with itself; the schedule tier asks
whether our transcription of a web page is coherent; even the corroboration
checks, which read logs the labels were never attached to, are reading one
tool's account of the same packets our labels describe. None of them can say a
label is WRONG, only that nothing supports it.

Engelen, Lanvin, Rimmer and Joosen (CNS 2022) re-labelled CIC-IDS-2017 and
CSE-CIC-IDS-2018 from the original PCAPs by hand -- reading attack tooling
output, following streams, and separating attacks that executed from attacks
that merely connected. They had no access to our manifests, our Zeek extraction
or our taxonomy. Where they disagree with us, the disagreement is evidence.

There is no such re-labelling of CIC-DDoS-2019. Nobody has published one. These
checks are restricted to the two datasets that have an oracle and are skipped on
2019 with that recorded, because "no external ground truth exists" and "the
external ground truth agrees" must never be reported as the same state.

WHAT A DISAGREEMENT INDICTS
---------------------------
The two sides construct flows differently and always will, so most differences
between them are not label errors and must not be reported as any. The
adjudication is baked into the checks rather than left to the reader:

  flow COUNTS differ                     -> NEITHER. CICFlowMeter ran with a hard
                                            120s active timeout and chops long
                                            connections; Zeek does not. This is
                                            never reported as a label error, and
                                            no check here divides one side's row
                                            count by the other's.
  one Zeek flow, many oracle rows        -> NEITHER. Same timeout semantics. The
                                            unit of comparison is the uid, and
                                            the overlapping rows are aggregated
                                            into one verdict with a purity score.
  oracle row is an 8.0.6.4 / protocol-0  -> THEIRS. A CICFlowMeter ARP-as-IPv4
    artefact                                artefact. Excluded from the join by
                                            src/label_validation/oracle/stage.py and counted.
  CLASS disagreement on an overlapping,  -> INVESTIGATE. The only genuinely
    direction-agreeing match                diagnostic case, and what
                                            orc.class_agreement exists to count.
  we say benign, oracle says attack, in  -> OURS. We inherited CIC's own
    a window CIC originally got wrong       scheduling error. The windows are
                                            named below and raise the severity.
  we say attack, oracle says `Attempted` -> NEITHER. Our label model has no
                                            second axis for "attempted but did
                                            not execute", so we cannot be wrong
                                            about a distinction we do not make.
                                            Reported as a gap, not an error.

Everything here reads reports/validation/oracle_<dataset>_uid.parquet, built
offline by src/label_validation/oracle/stage.py. The runner marks `oracle` available only when
that artefact exists, so these checks are skipped -- with a reason -- rather than
returning nothing on a lake where the join has never been run.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.label_validation.core.finding import Finding, Severity
from src.label_validation.oracle.stage import (ATTEMPTED_CATEGORIES, ORACLE_SOURCES, UNMAPPED,
                                 oracle_paths)
from src.label_validation.core.registry import check

ORACLE_DATASETS = tuple(sorted(ORACLE_SOURCES))

ATT_CATS = (0, 1, 2, 3, 4, 5, 6)

# Loaded once per dataset per process: the join report is a small JSON and every
# check below wants its provenance block.
_CACHE = {}

# Windows the published CIC schedule GOT WRONG, established by the CNS2022
# forensics. Each is a named PREDICTION about where the oracle will disagree
# with us and in which direction, not a generic "here be dragons" region -- so a
# check can report whether the prediction held, and raise severity when a
# disagreement lands inside one, because there the fault is demonstrably ours:
# we transcribed CIC's schedule faithfully and CIC's schedule was wrong.
#
#   expect="missed"  we call it benign; the oracle should call it an attack
#   expect="false"   we call it an attack; the oracle should call it benign
#   expect="label"   both call it an attack, but not the same attack
#
# Absolute windows are UTC, quoted from the forensics. Rule-relative ones are
# resolved against our own manifest at run time, so they follow a window if the
# manifest is corrected instead of silently pointing at the old one.
KNOWN_WRONG = (
    {"dataset": "cic-ids-2017", "name": "Infiltration Cool Disk - MAC",
     "utc": ("2017-07-06 17:53:36", "2017-07-06 20:02:19"),
     "hosts": ("205.174.165.73", "192.168.10.25"), "expect": "missed",
     "why": "CIC's published schedule misses this phase entirely. Our manifest "
            "covers its first seven minutes; the remaining two hours are benign "
            "in our labels by the closed-world assumption."},
    {"dataset": "cic-ids-2017", "name": "Infiltration NMAP scan, early start",
     "utc": ("2017-07-06 17:33:30", "2017-07-06 17:34:04"),
     "hosts": ("192.168.10.8", "192.168.10.5"), "expect": "missed",
     "why": "the internal scan begins at 17:33, before the published start."},
    {"dataset": "cic-ids-2017", "name": "Infiltration NMAP scan, second sweep",
     "utc": ("2017-07-06 18:05:14", "2017-07-06 18:46:04"),
     "hosts": ("192.168.10.8",), "expect": "missed",
     "why": "a second sweep across ten hosts, outside the published window."},
    {"dataset": "cic-ids-2017", "name": "Botnet Ares failed-C2 tail",
     "utc": ("2017-07-07 14:02:02", "2017-07-07 20:01:24"), "expect": "missed",
     "why": "active C2 ends at 14:02:02, but failed callbacks continue for six "
            "hours past it -- a tail no published window mentions."},
    {"dataset": "cic-ids-2017", "name": "SQL Injection early start",
     "rule": "WebAttack-SQLInjection", "before_s": 300, "expect": "missed",
     "why": "the first malicious packet precedes the published start by five "
            "minutes."},
    {"dataset": "cic-ids-2018", "name": "Infiltration second stage 28-02",
     "rule": "Infiltration", "date": "2018-02-28", "after_end_s": 14400,
     "expect": "missed",
     "why": "the infiltration runs hours past CIC's published finish; 44 flows "
            "across 28-02 and 01-03 are genuine infiltration CIC called benign."},
    {"dataset": "cic-ids-2018", "name": "Infiltration second stage 01-03",
     "rule": "Infiltration", "date": "2018-03-01", "after_end_s": 14400,
     "expect": "missed",
     "why": "as 28-02: the second stage continues past the published finish."},
    {"dataset": "cic-ids-2018", "name": "DDoS-HOIC late start 21-02",
     "rule": "DDoS-HOIC", "date": "2018-02-21", "after_start_s": 360,
     "expect": "false",
     "why": "the attack starts six minutes later than documented, so the first "
            "six minutes of our window are ordinary traffic labelled ddos."},
    {"dataset": "cic-ids-2018", "name": "DoS Slowhttptest absent 16-02",
     "rule": "DoS-SlowHTTPTest", "date": "2018-02-16", "expect": "false",
     "why": "the attack is not in the capture at all; the window holds failed "
            "FTP-Patator connections, which our labels call an executed DoS."},
    {"dataset": "cic-ids-2018", "name": "DDoS-LOIC-HTTP mislabelled 20-02",
     "rule": "DDoS-LOIC-HTTP", "date": "2018-02-20", "expect": "label",
     "why": "every flow here is mislabelled at the raw level: the traffic is a "
            "UDP flood recorded as an HTTP one. Both map to ddos, so the class "
            "agrees while the mechanism does not."},
)


# ------------------------------------------------------------------ plumbing

def _lit(v):
    return "'" + str(v).replace("'", "''") + "'"


def _share(n, d):
    return round(n / d, 9) if d else None


# Evidence rows kept per group list. The runner caps again on serialisation; this
# cap is earlier and coarser, so it is the one that has to be declared.
EVIDENCE_ROWS = 40


def _capped(rows, metrics, name, limit=EVIDENCE_ROWS):
    """The head of a row list, with the truncation stated in the metrics.

    A check that matches millions of flows must not carry millions of rows into
    a JSON report, so these lists are heads. Silently returning a head is the
    same quiet miscount this module exists to catch -- a reader who cannot see
    the cap will read `40 groups` as the whole population -- so the cap is
    recorded next to the number it bounds rather than left implicit.
    """
    if len(rows) > limit:
        metrics[f"{name}_total"] = len(rows)
        metrics[f"{name}_shown"] = limit
        metrics[f"{name}_truncated"] = True
    return rows[:limit]


def _src(ctx):
    """The per-uid join artefact, as a read_parquet expression."""
    path = ctx.q("oracle") or str(oracle_paths(ctx.dataset)[1])
    return f"read_parquet({_lit(path)})"


def _doc(ctx):
    """The join report, or {} if only the parquet survived.

    The report carries what the parquet cannot: provenance, the licence status,
    the artefact counts, and the oracle rows that matched nothing on our side.
    Its absence degrades the findings rather than failing them.
    """
    key = (ctx.dataset, "doc")
    if key not in _CACHE:
        path = Path(ctx.q("oracle_report") or oracle_paths(ctx.dataset)[0])
        try:
            _CACHE[key] = json.loads(path.read_text())
        except Exception:                       # noqa: BLE001 -- absence, not failure
            _CACHE[key] = {}
    return _CACHE[key]


def _provenance(ctx):
    """Provenance carried into every finding's metrics, licence included.

    The source states no licence anywhere -- not a file, not a header, not a
    clause in the paper. That is recorded on every claim derived from it, because
    "we were able to download it" and "we may redistribute it" are different
    facts and only the first has been established.
    """
    p = _doc(ctx).get("provenance", {})
    return {"oracle_source": p.get("source"), "oracle_url": p.get("url"),
            "oracle_licence": p.get("licence"),
            "oracle_retrieved_utc": p.get("retrieved_utc"),
            "oracle_bytes_match": p.get("bytes_match")}


def _utc(text):
    return (datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc).timestamp())


def _regions(ctx, expect=None):
    """KNOWN_WRONG resolved against this dataset's manifest.

    Returns (resolved, unresolved). A region naming a rule the manifest no longer
    contains is returned as unresolved rather than dropped: a prediction that
    cannot be tested is a different result from one that was tested and held.
    """
    want = set(expect) if expect else None
    out, missing = [], []
    for spec in KNOWN_WRONG:
        if spec["dataset"] != ctx.dataset:
            continue
        if want and spec["expect"] not in want:
            continue
        if "utc" in spec:
            t0, t1 = _utc(spec["utc"][0]), _utc(spec["utc"][1])
        else:
            rules = [r for r in ctx.rules if r["name"] == spec["rule"]
                     and (not spec.get("date") or str(r["date"]) == spec["date"])]
            if not rules:
                missing.append({"region": spec["name"], "rule": spec["rule"],
                                "date": spec.get("date"),
                                "reason": "no rule of that name and date in the "
                                          "manifest"})
                continue
            a, b = min(r["t0"] for r in rules), max(r["t1"] for r in rules)
            if "before_s" in spec:
                t0, t1 = a - spec["before_s"], a
            elif "after_start_s" in spec:
                t0, t1 = a, a + spec["after_start_s"]
            elif "after_end_s" in spec:
                t0, t1 = b, b + spec["after_end_s"]
            else:
                t0, t1 = a, b
        out.append({"name": spec["name"], "expect": spec["expect"],
                    "why": spec["why"], "hosts": spec.get("hosts"),
                    "t0": t0, "t1": t1})
    return out, missing


def _region_case(regions):
    """A CASE naming the known-wrong region a flow falls in, or NULL."""
    if not regions:
        return "CAST(NULL AS VARCHAR)"
    arms = []
    for r in regions:
        cond = f"ts >= {r['t0']} AND ts < {r['t1']}"
        if r["hosts"]:
            hosts = ", ".join(_lit(h) for h in r["hosts"])
            cond += f' AND ("id.orig_h" IN ({hosts}) OR "id.resp_h" IN ({hosts}))'
        arms.append(f"WHEN {cond} THEN {_lit(r['name'])}")
    return "CASE " + " ".join(arms) + " END"


def _window(lo, hi):
    def fmt(t):
        return (datetime.fromtimestamp(float(t), timezone.utc)
                .strftime("%Y-%m-%d %H:%M:%SZ") if t is not None else None)
    return f"{fmt(lo)} - {fmt(hi)}"


def _no_artefact(check_id, ctx):
    return Finding(
        check_id=check_id, dataset=ctx.dataset, severity=Severity.INFO,
        title="check did not run: the oracle join artefact is empty or unreadable",
        detail="The per-uid join exists but yielded no rows. Re-run "
               "`python src/label_validation/oracle/stage.py --dataset <ds> --join`. Recorded as "
               "a finding so an unbuilt artefact cannot be read as agreement.",
        metrics={"artefact": ctx.q("oracle")},
        repro=f"SELECT count(*) FROM {_src(ctx)}",
    )


# -------------------------------------------------------------------- checks

@check(id="orc.coverage", tier=3,
       title="share of our flows the external oracle has anything to say about",
       datasets=ORACLE_DATASETS, needs=("labelled", "oracle"),
       max_severity=Severity.CRITICAL)
def coverage(ctx):
    """The gate on every other number in this tier.

    A class agreement rate of 98% means nothing if the oracle matched 3% of the
    class: the 98% is then a statement about whichever flows happened to join,
    which is not a random sample of anything. So this runs first and its result
    conditions the reading of the rest.

    Low coverage is a fault in the JOIN, not in the labels, and the three ways it
    happens are separated here rather than averaged: a clock offset between the
    two sides (which puts every interval past every other interval and drives
    coverage to zero without raising an error), a tuple mismatch, and oracle rows
    whose label we could not map onto our taxonomy at all. That last one is its
    own finding, because an unmapped label is the oracle saying something we did
    not read -- silently binning it would let this tier report perfect agreement
    on a dataset it had mostly failed to understand.

    The fan-out figures are reported here too, and they are the reason no check
    in this tier compares flow counts: one Zeek connection routinely spans many
    oracle rows because CICFlowMeter cut it at a 120-second active timeout.
    """
    rows = ctx.lake.sql(f"""
        SELECT label_class, count(*) AS flows,
               count(*) FILTER (WHERE oracle_rows > 0) AS matched,
               count(*) FILTER (WHERE oracle_rows_dir_agree > 0) AS dir_agree,
               coalesce(sum(oracle_rows), 0) AS oracle_rows,
               max(oracle_rows) AS max_rows_per_flow,
               avg(oracle_rows) FILTER (WHERE oracle_rows > 0) AS mean_rows_per_flow
        FROM {_src(ctx)} GROUP BY 1 ORDER BY 2 DESC""")
    if not rows:
        return [_no_artefact("orc.coverage", ctx)]

    floor = float(ctx.threshold("oracle_coverage_min", 0.60))
    crit = float(ctx.threshold("oracle_coverage_critical", 0.25))
    doc = _doc(ctx)
    orc, jn = doc.get("oracle", {}), doc.get("join", {})

    per_class, flows, matched, dir_agree = [], 0, 0, 0
    for cls, n, m, d, orows, mx, mean in rows:
        n, m, d = int(n), int(m), int(d)
        flows, matched, dir_agree = flows + n, matched + m, dir_agree + d
        per_class.append({"label_class": cls, "flows": n, "matched": m,
                          "coverage": _share(m, n),
                          "matched_direction_agrees": d,
                          "oracle_rows": int(orows),
                          "max_oracle_rows_per_flow": int(mx or 0),
                          "mean_oracle_rows_per_flow": round(mean, 3) if mean else None})
    rate = _share(matched, flows)
    thin = [c for c in per_class if (c["coverage"] or 0) < floor]

    metrics = {
        "flows_in_oracle_span": flows, "flows_matched": matched,
        "coverage": rate, "coverage_min": floor, "coverage_critical": crit,
        "flows_direction_agrees": dir_agree,
        "oracle_rows_total": orc.get("rows_total"),
        "oracle_artefact_rows": orc.get("artefact_rows"),
        "oracle_rows_matched": orc.get("matched_rows"),
        "oracle_rows_unmatched": orc.get("unmatched_rows"),
        "clock_delta_seconds": jn.get("clock_delta_seconds"),
        "classes_below_floor": len(thin),
        **_provenance(ctx),
    }
    findings = [Finding(
        check_id="orc.coverage", dataset=ctx.dataset, severity=Severity.INFO,
        title=f"external oracle matches {(rate or 0):.2%} of our flows "
              f"({matched:,} of {flows:,})",
        detail="Share of our flows for which some manually-forensicated oracle row "
               "overlaps in time on the same unordered 5-tuple. This number gates "
               "every other figure in this tier: an agreement rate computed over a "
               "small unrepresentative match set describes the match set, not the "
               "labels. Flow COUNTS are deliberately not compared -- CICFlowMeter's "
               "120-second active timeout splits one Zeek connection into many "
               "oracle rows, which is what the fan-out columns show.",
        metrics=metrics,
        evidence=per_class + [{"artefact_kind": k.get("artefact"), "rows": k.get("rows")}
                              for k in orc.get("artefact_kinds", [])],
        repro=f"SELECT label_class, count(*), count(*) FILTER (WHERE oracle_rows > 0) "
              f"FROM {_src(ctx)} GROUP BY 1",
    )]

    if rate is not None and rate < floor:
        findings.append(Finding(
            check_id="orc.coverage", dataset=ctx.dataset,
            severity=Severity.CRITICAL if rate < crit else Severity.MAJOR,
            title=f"oracle coverage is {rate:.2%}, below the {floor:.0%} floor",
            detail="Most of our flows have no counterpart in the external label set, "
                   "so the agreement figures below are computed over a minority of "
                   "the data and must not be quoted as dataset-level accuracy. Check "
                   "the clock first: the two sides derive from the same PCAPs, so a "
                   "non-trivial clock_delta_seconds means the join is failing on time "
                   "rather than the labels failing on substance.",
            metrics=metrics, evidence=thin or per_class,
            repro=f"SELECT count(*) FILTER (WHERE oracle_rows > 0)::DOUBLE / count(*) "
                  f"FROM {_src(ctx)}",
        ))

    unmapped = orc.get("unmapped_labels") or []
    if unmapped:
        rows_unmapped = ctx.lake.one(
            f"SELECT count(*) FROM {_src(ctx)} "
            f"WHERE oracle_top_class = {_lit(UNMAPPED)}")[0]
        findings.append(Finding(
            check_id="orc.coverage", dataset=ctx.dataset, severity=Severity.MAJOR,
            title=f"{len(unmapped)} oracle label(s) have no mapping into our taxonomy",
            detail="The oracle labelled these flows as something, and we could not say "
                   "what. They are carried as class `unmapped` and excluded from every "
                   "agreement denominator rather than counted as agreement or as "
                   "disagreement -- but until they are mapped in "
                   "src/label_validation/oracle/stage.py's ORACLE_TO_CANONICAL, this tier is not "
                   "reading the whole of what the oracle said.",
            metrics={"unmapped_labels": len(unmapped),
                     "flows_with_unmapped_top_class": int(rows_unmapped)},
            evidence=[{"oracle_label": lb} for lb in unmapped],
            repro=f"SELECT DISTINCT oracle_top_label FROM {_src(ctx)} "
                  f"WHERE oracle_top_class = {_lit(UNMAPPED)}",
        ))
    return findings


@check(id="orc.class_agreement", tier=3,
       title="per-class agreement with the external oracle on overlapping matches",
       datasets=ORACLE_DATASETS, needs=("labelled", "oracle"),
       max_severity=Severity.CRITICAL)
def class_agreement(ctx):
    """The headline: where an independent re-labelling and ours actually differ.

    Restricted to matches that overlap in time AND agree on direction, because
    those are the only comparisons where the two sides are certainly describing
    the same traffic from the same side. A class disagreement here is the one
    genuinely diagnostic case in the whole adjudication table -- it is not a flow
    accounting artefact, not a timeout artefact and not the attempted axis; it is
    two labellings of the same packets that do not match.

    `unmapped` oracle labels are excluded from the denominator, so this is
    agreement among flows we could read, not agreement inflated by flows we could
    not. Purity is reported alongside: a uid overlapped by oracle rows that
    disagree with EACH OTHER is a weaker comparison than one overlapped by rows
    that are unanimous, and averaging the two would hide it.

    A low rate here does not by itself say we are wrong -- see orc.missed_attacks
    for the windows where the fault is demonstrably ours -- but it does say the
    class cannot be quoted as ground truth without saying which ground.
    """
    pairs = ctx.lake.sql(f"""
        SELECT label_class, oracle_top_class, count(*) AS flows,
               avg(oracle_purity) AS mean_purity,
               count(*) FILTER (WHERE oracle_purity >= 1.0) AS unanimous
        FROM {_src(ctx)}
        WHERE oracle_rows_dir_agree > 0
        GROUP BY 1, 2 ORDER BY flows DESC""")
    if not pairs:
        return [_no_artefact("orc.class_agreement", ctx)]

    floor = float(ctx.threshold("oracle_class_agreement_min", 0.90))
    crit = float(ctx.threshold("oracle_class_agreement_critical", 0.50))
    min_flows = int(ctx.threshold("oracle_agreement_min_flows", 100))

    by_class, matrix = {}, []
    for ours, theirs, n, purity, unan in pairs:
        n = int(n)
        c = by_class.setdefault(ours, {"label_class": ours, "decided": 0, "agree": 0,
                                       "unmapped": 0, "unanimous": 0})
        if theirs == UNMAPPED:
            c["unmapped"] += n
        else:
            c["decided"] += n
            c["unanimous"] += int(unan)
            if theirs == ours:
                c["agree"] += n
        matrix.append({"ours": ours, "oracle": theirs, "flows": n,
                       "mean_purity": round(purity, 4) if purity else None})
    for c in by_class.values():
        c["agreement_rate"] = _share(c["agree"], c["decided"])
        c["unanimous_share"] = _share(c["unanimous"], c["decided"])

    decided = sum(c["decided"] for c in by_class.values())
    agree = sum(c["agree"] for c in by_class.values())
    # A class below the floor on a handful of flows is a curiosity; the verdict is
    # reserved for classes with enough matched flows to carry one.
    judged = [c for c in by_class.values() if c["decided"] >= min_flows]
    weak = [c for c in judged if (c["agreement_rate"] or 0) < floor]
    broken = [c for c in weak if (c["agreement_rate"] or 0) < crit]

    regions, unresolved = _regions(ctx)
    by_region, region_pairs = [], []
    if regions:
        # Grouped by the class pair rather than aggregated into a list per
        # region: list() accumulates every value before it can be de-duplicated,
        # and a region covering four hours of a flood holds millions of flows.
        case = _region_case(regions)
        agg = {}
        for reg, ours, theirs, n in ctx.lake.sql(f"""
            SELECT {case} AS region, label_class, oracle_top_class, count(*) AS flows
            FROM {_src(ctx)}
            WHERE oracle_rows_dir_agree > 0 AND {case} IS NOT NULL
            GROUP BY 1, 2, 3 ORDER BY flows DESC"""):
            n = int(n)
            a = agg.setdefault(reg, {"flows": 0, "agree": 0, "ours": set(),
                                     "theirs": set()})
            a["flows"] += n
            a["agree"] += n if ours == theirs else 0
            a["ours"].add(ours)
            a["theirs"].add(theirs)
            region_pairs.append({"region": reg, "ours": ours, "oracle": theirs,
                                 "flows": n})
        for r in regions:
            a = agg.get(r["name"])
            by_region.append({
                "region": r["name"], "prediction": r["expect"],
                "window_utc": _window(r["t0"], r["t1"]), "note": r["why"],
                "flows": a["flows"] if a else 0,
                "our_classes": sorted(a["ours"]) if a else [],
                "oracle_classes": sorted(a["theirs"]) if a else [],
                "agree": a["agree"] if a else 0,
                "agreement_rate": _share(a["agree"], a["flows"]) if a else None})
        by_region.sort(key=lambda r: -r["flows"])

    metrics = {"flows_compared": decided, "flows_agreeing": agree,
               "agreement_rate": _share(agree, decided),
               "agreement_min": floor, "agreement_critical": crit,
               "min_flows_for_verdict": min_flows,
               "classes_compared": len(judged), "classes_below_floor": len(weak),
               "excluded_unmapped": sum(c["unmapped"] for c in by_class.values()),
               **{f"rate_{c['label_class']}": c["agreement_rate"]
                  for c in by_class.values()},
               **_provenance(ctx)}

    findings = [Finding(
        check_id="orc.class_agreement", dataset=ctx.dataset, severity=Severity.INFO,
        title=f"external agreement {(_share(agree, decided) or 0):.2%} over "
              f"{decided:,} direction-agreeing matches",
        detail="Per canonical class, the share of time-overlapping matches on which "
               "an independent manual re-labelling assigns the same class we do. "
               "Restricted to matches where both sides agree which end originated "
               "the connection, so the comparison is of two readings of the same "
               "traffic rather than of two flow-construction conventions. Oracle "
               "labels with no mapping into our taxonomy are excluded from the "
               "denominator, never counted as agreement.",
        metrics=metrics,
        evidence=(list(by_class.values())
                  + _capped(matrix, metrics, "confusion_pairs")
                  + _capped(by_region, metrics, "known_wrong_regions", 20)),
        repro=f"SELECT label_class, oracle_top_class, count(*) FROM {_src(ctx)} "
              f"WHERE oracle_rows_dir_agree > 0 GROUP BY 1, 2 ORDER BY 3 DESC",
    )]
    if weak:
        weak_metrics = {**metrics, "classes": [c["label_class"] for c in weak],
                        "flows": sum(c["decided"] for c in weak)}
        findings.append(Finding(
            check_id="orc.class_agreement", dataset=ctx.dataset,
            severity=Severity.CRITICAL if broken else Severity.MAJOR,
            title=f"{len(weak)} class(es) disagree with the external oracle on more "
                  f"than {1 - floor:.0%} of matched flows",
            detail="These are class disagreements on flows both sides saw, from the "
                   "same direction, overlapping in time -- the one comparison in this "
                   "tier that no flow-construction difference explains away. Each is "
                   "either our window, our taxonomy mapping, or their forensics; the "
                   "confusion matrix in the evidence says which pairs are being "
                   "confused, and orc.missed_attacks says which of them fall in "
                   "windows CIC is already known to have got wrong.",
            metrics=weak_metrics,
            evidence=weak + _capped(
                [m for m in matrix if m["ours"] in {c["label_class"] for c in weak}
                 and m["ours"] != m["oracle"]], weak_metrics, "disagreeing_pairs", 30),
            repro=f"SELECT label_class, oracle_top_class, count(*) FROM {_src(ctx)} "
                  f"WHERE oracle_rows_dir_agree > 0 GROUP BY 1, 2",
        ))
    if unresolved:
        findings.append(Finding(
            check_id="orc.class_agreement", dataset=ctx.dataset, severity=Severity.MINOR,
            title=f"{len(unresolved)} known-wrong window(s) could not be located in "
                  f"the manifest",
            detail="These regions are defined relative to a manifest rule that no "
                   "longer exists under that name and date, so the prediction attached "
                   "to them was not tested. Recorded rather than dropped: an untested "
                   "prediction is not a prediction that held.",
            metrics={"regions": len(unresolved)}, evidence=unresolved,
        ))
    return findings


@check(id="orc.missed_attacks", tier=3,
       title="flows we call benign that the external oracle calls an attack",
       datasets=ORACLE_DATASETS, needs=("labelled", "oracle"),
       max_severity=Severity.CRITICAL)
def missed_attacks(ctx):
    """Attack traffic sitting in our benign class, according to someone else.

    Every flow we call benign is benign by the closed-world assumption -- no rule
    matched it -- so a missed attack does not raise an error anywhere in the
    pipeline. It becomes training data for `benign`, and a detector trained on it
    is taught that the attack is normal. That is the most expensive single defect
    this whole module can find, which is why it is the one check here that reaches
    CRITICAL on its own.

    Severity is raised where the disagreement lands in a window CIC is already
    known to have got wrong -- the 2017 Cool Disk infiltration phase, the NMAP
    sweep that starts before its published time, the six-hour Ares C2 tail, the
    2018 second-stage infiltration. In those, the fault is not a judgement call:
    we transcribed CIC's schedule correctly and CIC's schedule omitted the
    attack, so the oracle is right and we inherited the error.

    Flows whose oracle rows are ALL `- Attempted` are separated out and graded
    down. An attempt that sent no payload against a closed port is a defensible
    thing to call benign, and this check must not spend its severity on it.
    """
    regions, _ = _regions(ctx, expect=("missed", "label"))
    region = _region_case(regions)
    # count(DISTINCT) and min(), never list(): list() accumulates every value in
    # the group before anything can de-duplicate or slice it, and one group here
    # can be millions of flows on a 62M-row zone. The named victim is an example
    # to look up, and is labelled as one.
    rows = ctx.lake.sql(f"""
        SELECT oracle_top_label, oracle_top_class, {region} AS region, capture,
               count(*) AS flows,
               count(*) FILTER (WHERE oracle_attempted_rows >= oracle_rows) AS attempted_only,
               min(ts) AS lo, max(ts) AS hi,
               avg(oracle_purity) AS mean_purity,
               min(uid) AS example_uid,
               count(DISTINCT "id.resp_h") AS distinct_victims,
               min("id.resp_h") AS example_victim
        FROM {_src(ctx)}
        WHERE label_class = 'benign' AND oracle_rows_dir_agree > 0
          AND oracle_top_class NOT IN ('benign', {_lit(UNMAPPED)})
        GROUP BY 1, 2, 3, 4 ORDER BY flows DESC""")

    benign = ctx.lake.one(
        f"SELECT count(*) FROM {_src(ctx)} WHERE label_class = 'benign'")[0]
    crit_rate = float(ctx.threshold("oracle_missed_attack_rate_critical", 0.001))

    if not rows:
        return [Finding(
            check_id="orc.missed_attacks", dataset=ctx.dataset, severity=Severity.INFO,
            title="no benign flow is called an attack by the external oracle",
            detail="Across every direction-agreeing match, nothing we labelled benign "
                   "was labelled an attack by the independent re-labelling. Stated "
                   "explicitly because a silent check here is indistinguishable from "
                   "one that never ran, and this is the strongest positive result the "
                   "tier can produce.",
            metrics={"benign_flows": int(benign), "missed": 0,
                     "known_wrong_windows_tested": len(regions), **_provenance(ctx)},
            evidence=[{"region": r["name"], "window_utc": _window(r["t0"], r["t1"]),
                       "prediction": r["expect"], "note": r["why"]} for r in regions],
        )]

    groups, genuine, attempted_only, in_known = [], 0, 0, 0
    for lb, cls, reg, cap, n, att, lo, hi, purity, uid, nvic, vic in rows:
        n, att = int(n), int(att)
        genuine += n - att
        attempted_only += att
        if reg:
            in_known += n
        groups.append({"oracle_label": lb, "oracle_class": cls,
                       "known_wrong_window": reg, "capture": cap, "flows": n,
                       "attempted_only": att, "genuine": n - att,
                       "window_utc": _window(lo, hi),
                       "mean_purity": round(purity, 4) if purity else None,
                       "example_uid": uid, "distinct_victims": int(nvic),
                       "example_victim": vic})
    total = genuine + attempted_only
    rate = _share(genuine, int(benign))

    # In a known-wrong window the adjudication is settled: CIC's own schedule is
    # missing the attack, so the disagreement indicts us. Elsewhere it is still a
    # candidate missed attack, but the window may simply be a boundary we placed
    # a few seconds differently.
    severity = (Severity.CRITICAL if (in_known or (rate or 0) > crit_rate)
                else Severity.MAJOR)
    named = sorted({g["known_wrong_window"] for g in groups
                    if g["known_wrong_window"]})
    metrics = {"benign_flows": int(benign), "missed_genuine": genuine,
               "missed_attempted_only": attempted_only, "missed_total": total,
               "missed_rate_of_benign": rate,
               "missed_rate_critical": crit_rate,
               "flows_in_known_wrong_windows": in_known,
               "known_wrong_windows_hit": named,
               "known_wrong_windows_tested": len(regions),
               "oracle_classes": sorted({g["oracle_class"] for g in groups}),
               **_provenance(ctx)}
    return [Finding(
        check_id="orc.missed_attacks", dataset=ctx.dataset, severity=severity,
        title=f"{genuine:,} benign flow(s) are attacks according to the external "
              f"oracle" + (f", {in_known:,} of them in {len(named)} window(s) CIC is "
                           f"known to have got wrong" if in_known else ""),
        detail="These flows carry no rule match, so they are benign only because the "
               "published schedule does not mention them -- and an independent manual "
               "re-labelling of the same packets says an attack was running. Every one "
               "is currently training data for the benign class."
               + (f" {in_known:,} of them fall inside a window the CNS2022 forensics "
                  f"already established that CIC mislabelled ({', '.join(named)}), "
                  f"where the fault is ours by inheritance: the transcription is "
                  f"faithful and the source schedule is wrong. The correction belongs "
                  f"in the manifest." if in_known else "")
               + (f" A further {attempted_only:,} flow(s) match only `- Attempted` "
                  f"oracle rows and are excluded from the count above: an attempt that "
                  f"delivered nothing is a defensible thing to call benign."
                  if attempted_only else ""),
        metrics=metrics,
        scope={"known_wrong_windows": named},
        evidence=_capped(groups, metrics, "groups") + [
            {"region": r["name"], "window_utc": _window(r["t0"], r["t1"]),
             "prediction": r["expect"], "note": r["why"]} for r in regions],
        repro=f"SELECT oracle_top_label, count(*) FROM {_src(ctx)} "
              f"WHERE label_class = 'benign' AND oracle_rows_dir_agree > 0 "
              f"AND oracle_top_class NOT IN ('benign', '{UNMAPPED}') GROUP BY 1",
    )]


@check(id="orc.false_attacks", tier=3,
       title="flows we call an attack that the external oracle calls benign",
       datasets=ORACLE_DATASETS, needs=("labelled", "oracle"),
       max_severity=Severity.CRITICAL)
def false_attacks(ctx):
    """The other direction: attack classes carrying ordinary traffic.

    Our labelling is time plus endpoints, so a window that is a few minutes too
    wide sweeps in whatever else those hosts were doing. The oracle read the
    packets instead, and where it says benign the flow is a false positive we
    minted ourselves -- it inflates the attack class, and any detector evaluated
    against it is credited for finding traffic that was never an attack.

    The `- Attempted` axis is separated out and NOT counted as a disagreement.
    Our label model has one axis; theirs has two. A flow we call `FTP-Patator`
    that they call `FTP-Patator - Attempted` is a distinction we do not make, not
    a claim we got wrong -- so only flows whose oracle rows say plain BENIGN,
    with no attempted attack row anywhere in the overlap, count against us here.
    """
    rows = ctx.lake.sql(f"""
        SELECT label_class, label_raw, capture,
               count(*) AS flows,
               count(*) FILTER (WHERE oracle_attempted_rows > 0) AS with_attempted,
               min(ts) AS lo, max(ts) AS hi,
               avg(oracle_purity) AS mean_purity, min(uid) AS example_uid
        FROM {_src(ctx)}
        WHERE label_class <> 'benign' AND oracle_rows_dir_agree > 0
          AND oracle_top_class = 'benign'
        GROUP BY 1, 2, 3 ORDER BY flows DESC""")

    totals = {r[0]: (int(r[1]), int(r[2])) for r in ctx.lake.sql(f"""
        SELECT label_class, count(*),
               count(*) FILTER (WHERE oracle_rows_dir_agree > 0)
        FROM {_src(ctx)} WHERE label_class <> 'benign' GROUP BY 1""")}
    if not rows:
        return [Finding(
            check_id="orc.false_attacks", dataset=ctx.dataset, severity=Severity.INFO,
            title="no attack flow is called benign by the external oracle",
            detail="On every direction-agreeing match, nothing we labelled an attack "
                   "was labelled benign by the independent re-labelling. Reported "
                   "explicitly so a clean result is visible as a result.",
            metrics={"attack_flows": sum(t[0] for t in totals.values()),
                     "attack_flows_matched": sum(t[1] for t in totals.values()),
                     **_provenance(ctx)},
        )]

    warn = float(ctx.threshold("oracle_false_attack_rate_warn", 0.05))
    crit = float(ctx.threshold("oracle_false_attack_rate_critical", 0.25))
    regions, _ = _regions(ctx, expect=("false",))
    region = _region_case(regions)
    in_known = {r[0]: int(r[1]) for r in ctx.lake.sql(f"""
        SELECT {region} AS region, count(*) FROM {_src(ctx)}
        WHERE label_class <> 'benign' AND oracle_rows_dir_agree > 0
          AND oracle_top_class = 'benign' AND {region} IS NOT NULL
        GROUP BY 1""")} if regions else {}

    groups, per_class = [], {}
    for cls, raw, cap, n, att, lo, hi, purity, uid in rows:
        n, att = int(n), int(att)
        c = per_class.setdefault(cls, {"label_class": cls, "flows": totals.get(cls, (0, 0))[0],
                                       "matched": totals.get(cls, (0, 0))[1],
                                       "called_benign": 0, "with_attempted_rows": 0})
        c["called_benign"] += n
        c["with_attempted_rows"] += att
        groups.append({"label_class": cls, "attack": raw, "capture": cap,
                       "flows": n, "with_attempted_oracle_rows": att,
                       "pure_benign": n - att, "window_utc": _window(lo, hi),
                       "mean_purity": round(purity, 4) if purity else None,
                       "example_uid": uid})
    for c in per_class.values():
        c["false_rate_of_matched"] = _share(c["called_benign"], c["matched"])

    pure = sum(g["pure_benign"] for g in groups)
    mixed = sum(g["with_attempted_oracle_rows"] for g in groups)
    matched = sum(t[1] for t in totals.values())
    rate = _share(pure, matched)
    over = [c for c in per_class.values()
            if (c["false_rate_of_matched"] or 0) > warn]
    metrics = {"attack_flows": sum(t[0] for t in totals.values()),
               "attack_flows_matched": matched,
               "called_benign_pure": pure,
               "called_benign_with_attempted_rows": mixed,
               "false_rate_of_matched": rate,
               "false_rate_warn": warn, "false_rate_critical": crit,
               "classes_over_warn": [c["label_class"] for c in over],
               "flows_in_known_wrong_windows": sum(in_known.values()),
               **_provenance(ctx)}

    findings = [Finding(
        check_id="orc.false_attacks", dataset=ctx.dataset,
        severity=(Severity.CRITICAL if (rate or 0) > crit else
                  Severity.MAJOR if over else Severity.INFO),
        title=f"{pure:,} attack flow(s) are benign according to the external oracle"
              + (f" ({rate:.2%} of matched attack flows)" if rate else ""),
        detail="Our labels are a time window and a pair of addresses, so a window "
               "wider than the attack collects whatever else those hosts were doing. "
               "These flows are in an attack class and the independent re-labelling "
               "reads them as ordinary traffic. They inflate the class and credit any "
               "detector evaluated on them with finding something that was not there."
               + (f" A further {mixed:,} flow(s) overlap at least one `- Attempted` "
                  f"oracle row and are reported separately rather than counted here: "
                  f"an attack that connected and delivered nothing is a distinction "
                  f"our label model does not make, so it cannot be a label error of "
                  f"ours." if mixed else "")
               + (f" {sum(in_known.values()):,} of these fall in windows CIC is known "
                  f"to have got wrong ({', '.join(sorted(in_known))}), where the "
                  f"correction belongs in the manifest."
                  if in_known else ""),
        metrics=metrics,
        scope={"known_wrong_windows": sorted(in_known)},
        evidence=list(per_class.values()) + _capped(groups, metrics, "groups"),
        repro=f"SELECT label_class, label_raw, count(*) FROM {_src(ctx)} "
              f"WHERE label_class <> 'benign' AND oracle_rows_dir_agree > 0 "
              f"AND oracle_top_class = 'benign' GROUP BY 1, 2",
    )]
    return findings


@check(id="orc.attempted_axis", tier=3,
       title="attack flows the oracle marks `- Attempted`, by category",
       datasets=ORACLE_DATASETS, needs=("labelled", "oracle"),
       max_severity=Severity.MINOR)
def attempted_axis(ctx):
    """A distinction the oracle makes and our label model cannot.

    The CNS2022 labels carry a second axis: an attack row can be marked
    `- Attempted` with a category saying why it did not execute -- no payload
    sent, port closed, target unresponsive, attack implemented incorrectly. We
    have `label_executed`, which is a payload test applied only to attacks the
    taxonomy lists under requires_payload, and nothing else.

    So this is NOT reported as a labelling error, and no disagreement involving
    the attempted axis counts against us anywhere in this tier. It is reported as
    a measurement of a gap: the share of each attack class that an independent
    forensic pass judged did not actually execute. That is the number a reader
    needs before quoting a class's flow count as evidence the attack happened,
    and it is the same defect the evidence log records for 2018's FTP-BruteForce
    from the other direction.
    """
    rows = ctx.lake.sql(f"""
        SELECT label_class, label_raw,
               count(*) AS flows,
               count(*) FILTER (WHERE oracle_rows > 0) AS matched,
               count(*) FILTER (WHERE oracle_attempted_rows > 0) AS any_attempted,
               count(*) FILTER (WHERE oracle_rows > 0
                                  AND oracle_attempted_rows >= oracle_rows) AS all_attempted,
               {', '.join(f'coalesce(sum(att_cat_{c}), 0)' for c in ATT_CATS)},
               coalesce(sum(att_cat_other), 0)
        FROM {_src(ctx)} WHERE label_class <> 'benign'
        GROUP BY 1, 2 ORDER BY flows DESC""")
    if not rows:
        return [_no_artefact("orc.attempted_axis", ctx)]

    notable = float(ctx.threshold("oracle_attempted_share_notable", 0.25))
    per_attack, by_class, cats = [], {}, {c: 0 for c in ATT_CATS}
    cats_other = 0
    for row in rows:
        cls, raw, n, matched, any_att, all_att = (row[0], row[1], int(row[2]),
                                                  int(row[3]), int(row[4]), int(row[5]))
        counts = [int(v) for v in row[6:6 + len(ATT_CATS)]]
        other = int(row[-1])
        for c, v in zip(ATT_CATS, counts):
            cats[c] += v
        cats_other += other
        c = by_class.setdefault(cls, {"label_class": cls, "flows": 0, "matched": 0,
                                      "any_attempted": 0, "all_attempted": 0})
        c["flows"] += n
        c["matched"] += matched
        c["any_attempted"] += any_att
        c["all_attempted"] += all_att
        per_attack.append({"label_class": cls, "attack": raw, "flows": n,
                           "matched": matched, "flows_with_attempted_row": any_att,
                           "flows_entirely_attempted": all_att,
                           "attempted_share_of_matched": _share(all_att, matched),
                           **{f"cat_{k}": v for k, v in zip(ATT_CATS, counts)},
                           "cat_other": other})
    for c in by_class.values():
        c["attempted_share_of_matched"] = _share(c["all_attempted"], c["matched"])

    matched = sum(c["matched"] for c in by_class.values())
    all_att = sum(c["all_attempted"] for c in by_class.values())
    share = _share(all_att, matched)
    over = [c for c in by_class.values()
            if (c["attempted_share_of_matched"] or 0) > notable]

    breakdown = [{"attempted_category": k, "meaning": ATTEMPTED_CATEGORIES[k],
                  "oracle_rows": v} for k, v in sorted(cats.items()) if v]
    if cats_other:
        breakdown.append({"attempted_category": "other", "meaning": "outside 0-6",
                          "oracle_rows": cats_other})
    metrics = {"attack_flows_matched": matched,
               "flows_entirely_attempted": all_att,
               "attempted_share_of_matched": share,
               "attempted_share_notable": notable,
               "classes_over_notable": [c["label_class"] for c in over],
               **{f"cat_{k}_rows": v for k, v in sorted(cats.items())},
               **_provenance(ctx)}

    return [Finding(
        check_id="orc.attempted_axis", dataset=ctx.dataset,
        severity=Severity.MINOR if over else Severity.INFO,
        title=f"{all_att:,} of {matched:,} matched attack flow(s) "
              f"({(share or 0):.2%}) are `- Attempted` in the external oracle",
        detail="The oracle records, per flow, whether an attack actually executed and "
               "why not. Our labels have no such axis, so this is a gap in our label "
               "model rather than an error in it -- nothing here says a label is "
               "wrong, and no disagreement involving this axis is counted against us "
               "anywhere in this tier. What it constrains is what an attack class may "
               "be quoted as evidence OF: a class whose matched flows are largely "
               "attempts is a class of connection attempts, and a flow count taken "
               "from it is not a count of executed attacks."
               + (f" {len(over)} class(es) are above {notable:.0%}."
                  if over else ""),
        metrics=metrics,
        evidence=(list(by_class.values()) + breakdown
                  + _capped(per_attack, metrics, "attacks", 30)),
        repro=f"SELECT label_class, count(*) FILTER (WHERE oracle_attempted_rows >= "
              f"oracle_rows AND oracle_rows > 0), count(*) FILTER (WHERE oracle_rows > 0) "
              f"FROM {_src(ctx)} WHERE label_class <> 'benign' GROUP BY 1",
    )]


@check(id="orc.direction_reversal", tier=3,
       title="share of oracle matches whose originator disagrees with ours",
       datasets=ORACLE_DATASETS, needs=("labelled", "oracle"),
       max_severity=Severity.MINOR)
def direction_reversal(ctx):
    """A diagnostic for the join, not a claim about any label.

    CICFlowMeter fixes a flow's direction from the first packet it happens to
    see; Zeek fixes it from the SYN. Where a capture starts mid-connection, or
    the handshake is missing, the two disagree about which end is the originator.
    The join is therefore on the UNORDERED 5-tuple, and direction agreement is
    carried as a boolean instead of being required -- so these flows are matched
    rather than silently lost, and the reversal rate is a number rather than a
    hole in the coverage.

    Nothing here says a label is wrong. What a high rate DOES say is that the
    agreement figures in this tier -- which are computed only over
    direction-agreeing matches -- rest on a smaller population than the coverage
    figure suggests, and that any per-flow feature of ours that is defined
    relative to the originator (orig_bytes, orig_pkts, the whole orig/resp split)
    is not describing the same endpoint the oracle's features describe.
    """
    rows = ctx.lake.sql(f"""
        SELECT proto, label_class,
               count(*) FILTER (WHERE oracle_rows > 0) AS matched,
               count(*) FILTER (WHERE oracle_rows_dir_agree > 0) AS dir_agree,
               coalesce(sum(oracle_rows), 0) AS oracle_rows,
               coalesce(sum(oracle_rows_dir_agree), 0) AS oracle_rows_dir_agree
        FROM {_src(ctx)} WHERE oracle_rows > 0
        GROUP BY 1, 2 ORDER BY matched DESC""")
    if not rows:
        return [_no_artefact("orc.direction_reversal", ctx)]

    warn = float(ctx.threshold("oracle_direction_reversal_warn", 0.20))
    detail_rows, matched, agreed, o_rows, o_agreed = [], 0, 0, 0, 0
    for proto, cls, m, d, orows, odir in rows:
        m, d = int(m), int(d)
        matched, agreed = matched + m, agreed + d
        o_rows, o_agreed = o_rows + int(orows), o_agreed + int(odir)
        detail_rows.append({"proto": proto, "label_class": cls, "matched": m,
                            "direction_agrees": d, "reversed": m - d,
                            "reversal_rate": _share(m - d, m)})
    rate = _share(matched - agreed, matched)
    detail_rows.sort(key=lambda r: -(r["reversal_rate"] or 0))
    worst = [r for r in detail_rows if (r["reversal_rate"] or 0) > warn]
    metrics = {"flows_matched": matched, "flows_direction_agrees": agreed,
               "flows_reversed": matched - agreed, "reversal_rate": rate,
               "oracle_rows_matched": o_rows,
               "oracle_rows_direction_agrees": o_agreed,
               "oracle_row_reversal_rate": _share(o_rows - o_agreed, o_rows),
               "reversal_warn": warn,
               "groups_over_warn": len(worst), **_provenance(ctx)}

    return [Finding(
        check_id="orc.direction_reversal", dataset=ctx.dataset,
        severity=Severity.MINOR if (rate or 0) > warn else Severity.INFO,
        title=f"{matched - agreed:,} of {matched:,} oracle matches ({(rate or 0):.2%}) "
              f"have the originator reversed",
        detail="CICFlowMeter fixes direction from the first packet it sees and Zeek "
               "from the SYN, so the two disagree wherever a handshake is absent. The "
               "join keys on the unordered 5-tuple precisely so these flows are "
               "matched instead of lost, and this is the measurement that would "
               "otherwise be invisible. It indicts neither labelling. It does bound "
               "the rest of the tier -- agreement is computed over direction-agreeing "
               "matches only -- and it means any orig/resp-relative feature is not "
               "describing the same endpoint on both sides.",
        metrics=metrics,
        evidence=_capped(detail_rows, metrics, "groups"),
        repro=f"SELECT count(*) FILTER (WHERE oracle_rows_dir_agree > 0)::DOUBLE / "
              f"count(*) FROM {_src(ctx)} WHERE oracle_rows > 0",
    )]
