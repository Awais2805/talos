#!/usr/bin/env python3
"""Tier 1 -- the flows we declined to adjudicate, and what declining cost.

Every other check in this module asks whether a label is right. These ask about
the labels we refused to make. The closed-world assumption means Talos always
produces an answer: a flow no attack window claims is benign, whatever it
actually was. In two measured places that answer is manufactured -- 2019's
inter-window gaps, where a reflection flood is plainly still running at 139k
flows/min but no window says which vector it is, and 2018's infiltration
windows, where CIC names the infected host but never the second stage's targets.
The manifests therefore declare those spans unadjudicable, and labelling marks
every flow inside one `label_quality = 'uncertain'`.

That is a subtraction from the evidence base, so it has to be visible or it
becomes a way to make any number look good. Three things must hold. Every
declared region must match traffic, or it is a stale declaration narrowing
nothing while appearing to. The share withdrawn must be reported per class,
because "99.4% precision" and "99.4% precision on 97.1% of flows" are different
claims and only the second one is honest. And no region may overlap a scheduled
attack in both time and endpoints without an answer, because claiming an attack
ran somewhere and declining to say what happened there are contradictory
positions to hold at once.

The regions come from the manifest through the same loader labelling used, and
match through the same predicate, so a region here cannot come to mean something
subtly different from the region that produced the data. A labelled zone written
before a region was declared cannot slip through either: adding a `quarantine:`
block changes the manifest's content hash, and int.provenance_current blocks on
that until labelling is re-run.
"""

from src.preprocess.label.label_flows import MATCH, load_quarantine, rules_sql
from src.label_validation.core.finding import Finding, Severity
from src.label_validation.core.registry import check


def _regions(ctx):
    """The declared regions, normalised exactly as the labelling stage did."""
    return load_quarantine(ctx.manifest, ctx.rules)


def _window(z):
    return f"{z['date']} {z['start']}-{z['end']}"


@check(id="qtn.declared_regions_fire", tier=1,
       title="every declared quarantine region matches traffic",
       needs=("labelled",), max_severity=Severity.MAJOR)
def declared_regions_fire(ctx):
    """A region that matches nothing withdraws nothing -- while looking like it did.

    This is the quarantine equivalent of sch.rule_yield, and it fails the same
    silent way: a mistyped minute or a victim address that never appears leaves
    the manifest asserting that a span has been excluded, the coverage figure
    reporting 100%, and the traffic the region was written for sitting in the
    pool with a label nobody defends. Since the whole point of declaring regions
    rather than inferring them is that a reader can audit them, a declaration
    that does not bite is worse than no declaration at all.

    Counted per region rather than per reason, and each region counted
    independently, so a flow inside two overlapping regions supports both. The
    question here is whether THIS declaration bites, not how many flows are out.
    """
    regions = _regions(ctx)
    if not regions:
        return []
    floor = int(ctx.threshold("quarantine_region_min_flows", 1))

    q = f"""
    WITH qrules AS ({rules_sql(regions)}),
    conn AS (SELECT * FROM read_parquet('{ctx.q('labelled')}', union_by_name = true))
    SELECT r.rid, count(*) AS n
    FROM conn c JOIN qrules r ON {MATCH}
    GROUP BY 1"""
    got = {int(rid): int(n) for rid, n in ctx.lake.sql(q)}

    findings = []
    for z in regions:
        n = got.get(z["rid"], 0)
        if n >= floor:
            continue
        minutes = max((z["t1"] - z["t0"]) / 60.0, 1e-6)
        findings.append(Finding(
            check_id="qtn.declared_regions_fire", dataset=ctx.dataset,
            severity=Severity.MAJOR,
            title=f"quarantine region {z['reason']} {_window(z)} matched "
                  f"{n:,} flow(s)",
            detail="This region is declared unadjudicable but selects no traffic, so "
                   "nothing is being withdrawn by it. Either its times or its endpoints "
                   "are wrong -- in which case the flows it was written for are still in "
                   "the pool, labelled by the closed-world assumption -- or the region "
                   "describes a span that no longer exists and should be deleted. Both "
                   "leave the manifest claiming an exclusion the data does not show.",
            metrics={"region_id": z["rid"], "reason": z["reason"],
                     "window_local": _window(z),
                     "window_minutes": round(minutes, 1),
                     "flows": n, "min_flows": floor,
                     "victims": z["victims"], "attackers": z["attackers"],
                     "victim_prefix": z["prefix"]},
            scope={"region_id": z["rid"], "reason": z["reason"],
                   "window_local": _window(z)},
            evidence=[{"justification": z["justification"]}],
            repro=q.strip(),
        ))
    return findings


@check(id="qtn.coverage", tier=1,
       title="the share of flows quarantined, per dataset and per class",
       needs=("labelled",), max_severity=Severity.MINOR)
def coverage(ctx):
    """The denominator every accuracy figure downstream has to be quoted against.

    This check reports rather than judges: the INFO finding is emitted on every
    run, including runs where nothing is quarantined at all, because the number
    IS the output. A result stated without it -- "99.4% precision" -- is a claim
    about an unnamed subset of the data, and the subset was chosen by us.

    Per class as well as per dataset, because quarantine is never uniform. It is
    aimed at exactly the regions where the labels were weakest, so it lands
    almost entirely on one or two classes, and a class whose remaining flows are
    a small unrepresentative fraction of what it started as cannot carry the
    same weight afterwards. 2018's `infiltration` is the extreme case: all 8 of
    its flows sit inside a declared region, so the class is left with no
    adjudicable evidence whatsoever and must not be evaluated at all.

    Read off the labelled zone rather than recomputed from the manifest, so this
    is what the data actually says, not what the declarations intend.
    """
    if "label_quality" not in ctx.columns:
        return [Finding(
            check_id="qtn.coverage", dataset=ctx.dataset, severity=Severity.MINOR,
            title="labelled zone carries no label_quality column — coverage is unknown",
            detail="Quarantine marks unadjudicable flows with label_quality and "
                   "quarantine_reason. This zone has neither, so it was written by a "
                   "labelling run that predates them and there is no way to tell from "
                   "the data which flows the schedule could actually speak for. Any "
                   "accuracy figure derived from it is quoted without a denominator.",
            metrics={"label_quality_present": False},
            repro=f"DESCRIBE SELECT * FROM read_parquet('{ctx.q('labelled')}', "
                  f"union_by_name=true)",
        )]

    regions = _regions(ctx)
    q = f"""
    SELECT label_class, label_quality, quarantine_reason, count(*) AS n
    FROM read_parquet('{ctx.q('labelled')}', union_by_name = true)
    GROUP BY 1, 2, 3 ORDER BY 4 DESC"""
    rows = ctx.lake.sql(q)

    per_class, per_reason, total, uncertain = {}, {}, 0, 0
    for cls, quality, reason, n in rows:
        n = int(n)
        total += n
        rec = per_class.setdefault(cls, {"class": cls, "flows": 0, "quarantined": 0})
        rec["flows"] += n
        if quality == "uncertain":
            uncertain += n
            rec["quarantined"] += n
            per_reason[reason] = per_reason.get(reason, 0) + n
    for rec in per_class.values():
        rec["quarantined_share"] = round(rec["quarantined"] / rec["flows"], 6) \
            if rec["flows"] else 0.0
        rec["adjudicable"] = rec["flows"] - rec["quarantined"]

    share = uncertain / total if total else 0.0
    classes = sorted(per_class.values(), key=lambda c: -c["quarantined_share"])
    metrics = {
        "flows_total": total,
        "flows_quarantined": uncertain,
        "quarantine_share": round(share, 6),
        # The figure to quote beside every accuracy number this dataset produces.
        "label_coverage": round(1 - share, 6),
        "regions_declared": len(regions),
        "reasons": {r: v for r, v in sorted(per_reason.items(), key=lambda kv: -kv[1])},
        "classes_wholly_quarantined": sorted(
            c["class"] for c in classes if c["flows"] and not c["adjudicable"]),
    }

    findings = [Finding(
        check_id="qtn.coverage", dataset=ctx.dataset, severity=Severity.INFO,
        title=f"labels cover {100 * (1 - share):.2f}% of flows; {uncertain:,} "
              f"({share:.2%}) are quarantined as unadjudicable",
        detail=(f"{len(regions)} declared region(s) withdraw {uncertain:,} of "
                f"{total:,} flows from the training and evaluation pools. Every "
                f"result from this dataset is a statement about the remaining "
                f"{100 * (1 - share):.2f}% and must be reported that way; the "
                f"withdrawn flows keep their closed-world label so nothing is "
                f"hidden, but nothing about them is being claimed either."
                + (f" Reasons: "
                   + ", ".join(f"{r} {v:,}" for r, v in
                               sorted(per_reason.items(), key=lambda kv: -kv[1]))
                   if per_reason else "")),
        metrics=metrics,
        evidence=classes,
        repro=q.strip(),
    )]

    # A class is not a class any more once most of it is withdrawn: the survivors
    # are whatever the quarantine happened to miss, which is not a sample of
    # anything. Reported separately from the headline so it cannot be lost in it.
    warn = float(ctx.threshold("quarantine_class_share_warn", 0.5))
    for c in classes:
        if c["quarantined_share"] <= warn or not c["flows"]:
            continue
        findings.append(Finding(
            check_id="qtn.coverage", dataset=ctx.dataset, severity=Severity.MINOR,
            title=f"{c['class']}: {c['quarantined_share']:.1%} of the class is "
                  f"quarantined, leaving {c['adjudicable']:,} adjudicable flow(s)",
            detail="Most of this class sits inside a declared region, so what is left is "
                   "not a sample of the class -- it is the part the quarantine did not "
                   "reach. Any per-class figure computed from the remainder describes "
                   "that residue and nothing wider."
                   + (" There is no adjudicable evidence left at all, so the class "
                      "cannot be evaluated and should be dropped rather than reported."
                      if not c["adjudicable"] else ""),
            metrics={**c, "quarantine_class_share_warn": warn},
            scope={"class": c["class"]},
            repro=q.strip(),
        ))
    return findings


@check(id="qtn.overlap_with_attack", tier=1,
       title="no quarantine region contradicts a scheduled attack window",
       max_severity=Severity.MAJOR)
def overlap_with_attack(ctx):
    """Claiming an attack ran there and declining to say what happened there.

    A region overlapping a window in both time and endpoints holds two positions
    at once: the schedule asserts a named attack against those hosts in those
    minutes, and the manifest declines to adjudicate the same flows. One of them
    has to give -- either the window is wrong and should be corrected or removed,
    or the region is over-broad and should be narrowed to the part the schedule
    does not speak for.

    Neither is automatically true, which is why this reports rather than
    resolves, and why the region's written justification travels with the
    finding: the point is to force the contradiction into the open and make
    someone answer it in the manifest. CSE-CIC-IDS-2018 answers it deliberately
    -- its four regions coincide with the four Infiltration windows exactly,
    because the schedule's claim there amounts to 8 unfinished flows and
    accounts for none of the ~175k per bucket around them -- so this fires by
    design on that dataset and the answer is in the manifest beside the region.
    A region with no justification at all has no answer, and is called out
    separately.

    Pure manifest arithmetic, so it needs no data and runs even when the
    labelled zone is missing: a contradiction between two declarations is
    visible in the declarations.
    """
    regions = _regions(ctx)
    if not regions:
        return []

    def ips(x):
        return set(x["attackers"] or []) | set(x["victims"] or [])

    pairs = []
    for z in regions:
        for r in ctx.rules:
            if z["t0"] >= r["t1"] or r["t0"] >= z["t1"]:
                continue
            shared = ips(z) & ips(r)
            # A subnet victim on either side can select the other's endpoints
            # without sharing a literal address, so a prefix counts as contact.
            if not (shared or z["prefix"] or r["prefix"]):
                continue
            overlap = min(z["t1"], r["t1"]) - max(z["t0"], r["t0"])
            pairs.append({
                "region_id": z["rid"], "reason": z["reason"],
                "region_window": _window(z),
                "rule_id": r["rid"], "raw": r["name"], "class": r["canonical"],
                "rule_window": f"{r['date']} {r['start']}-{r['end']}",
                "overlap_seconds": round(overlap, 1),
                "overlap_share_of_region": round(overlap / (z["t1"] - z["t0"]), 4),
                "shared_endpoints": sorted(shared),
                "justification": z["justification"],
            })
    if not pairs:
        return []

    undocumented = [p for p in pairs if not str(p["justification"] or "").strip()]
    regions_hit = {p["region_id"] for p in pairs}
    classes = sorted({p["class"] for p in pairs})
    return [Finding(
        check_id="qtn.overlap_with_attack", dataset=ctx.dataset,
        severity=Severity.MAJOR,
        title=f"{len(regions_hit)} quarantine region(s) overlap {len(pairs)} scheduled "
              f"attack window(s) in both time and endpoints",
        detail=f"The schedule claims {', '.join(classes)} against these hosts in these "
               f"minutes while the manifest declines to adjudicate the same flows. Both "
               f"cannot stand: a window is either evidence of what happened there or it "
               f"is not. Read each region's justification below and either narrow the "
               f"region to the span the schedule does not cover, or withdraw the window."
               + (f" {len(undocumented)} of these region(s) carry NO justification, so "
                  f"there is nothing to read." if undocumented else ""),
        metrics={"overlapping_pairs": len(pairs),
                 "regions_involved": len(regions_hit),
                 "regions_declared": len(regions),
                 "attack_classes": classes,
                 "pairs_without_justification": len(undocumented),
                 "overlap_seconds_total": round(sum(p["overlap_seconds"] for p in pairs), 1)},
        scope={"reasons": sorted({p["reason"] for p in pairs})},
        evidence=sorted(pairs, key=lambda p: -p["overlap_seconds"]),
    )]
