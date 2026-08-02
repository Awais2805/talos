#!/usr/bin/env python3
"""Tier 1 -- is the schedule we labelled from actually the schedule that ran?

Every label in Talos comes from a wall-clock window transcribed out of a CIC
web page, converted to epoch seconds with an offset that no CIC source actually
states, and matched against flow start times. Three independent things can be
wrong -- the transcription, the offset, and the matching rule -- and all three
fail the same way: silently, producing a complete-looking labelled dataset.

The checks here work from one idea. A scheduled attack window, if correct,
separates two traffic regimes: loud inside, quiet outside. So instead of asking
"how many flows did this rule match", they ask "does the traffic change at the
boundary the schedule claims". A window that starts inside a flood, or ends
while it is still running, shows no edge -- and that is measurable without
knowing anything about the attack.

The sweep is deliberately structured as ONE scan per dataset. Each flow that
matches a rule's endpoint predicate is bucketed by its offset from that rule's
start and end; every candidate shift of the window is then a cumulative sum over
that little table, computed offline. Sweeping sixty offsets costs one pass, not
sixty.
"""

from src.preprocess.label.label_flows import _victim_ok, rules_sql
from src.label_validation.core.finding import Finding, Severity
from src.label_validation.core.registry import check

# The endpoint half of label_flows.MATCH, with the time clause removed so the
# sweep can look at traffic either side of the window. This MUST mirror
# label_flows.MATCH: if that predicate changes and this does not, the sweep
# would be measuring a different population than the one that was labelled.
ENDPOINTS = f"""(
    (r.atk IS NOT NULL AND (
        (list_contains(r.atk, c."id.orig_h") AND {_victim_ok('id.resp_h')})
     OR (list_contains(r.atk, c."id.resp_h") AND {_victim_ok('id.orig_h')})
    ))
    OR (r.atk IS NULL AND r.vic IS NOT NULL
        AND (list_contains(r.vic, c."id.orig_h") OR list_contains(r.vic, c."id.resp_h")))
  )"""

EPS = 1e-9


# --------------------------------------------------------------- the sweep

def _sweep(ctx, span, bucket, rules=None):
    """flows per (rule, offset-from-start bucket), for rules widened by `span`.

    Cached on the context: window_sweep, boundary_residue and intra_window_gaps
    all read the same table, and re-scanning 72.5M flows three times to compute
    three views of one measurement would be indefensible.
    """
    rules = rules if rules is not None else ctx.rules
    key = ("_sweep", span, bucket, tuple(r["rid"] for r in rules))
    cached = getattr(ctx, "_sweep_cache", None)
    if cached is None:
        cached = {}
        setattr(ctx, "_sweep_cache", cached)
    if key in cached:
        return cached[key]

    q = f"""
    WITH rules AS ({rules_sql(rules)}),
    conn AS (SELECT * FROM read_parquet('{ctx.q('labelled')}', union_by_name = true))
    SELECT r.rid,
           CAST(floor((c.ts - r.t0) / {bucket}) AS BIGINT) AS b,
           count(*) AS n
    FROM conn c JOIN rules r
      ON c.ts >= r.t0 - {span} AND c.ts < r.t1 + {span}
     AND {ENDPOINTS}
    GROUP BY 1, 2"""
    rows = ctx.lake.sql(q)

    table = {}
    for rid, b, n in rows:
        table.setdefault(int(rid), {})[int(b)] = int(n)
    cached[key] = (table, q)
    return cached[key]


def _rate(hist, lo, hi, bucket):
    """flows/second across bucket indices [lo, hi)."""
    if hi <= lo:
        return 0.0
    total = sum(v for k, v in hist.items() if lo <= k < hi)
    return total / ((hi - lo) * bucket)


def _series(hist, lo, hi, bucket):
    return [(k, hist.get(k, 0) / bucket) for k in range(lo, hi)]


@check(id="sch.window_sweep", tier=1,
       title="attack windows line up with the traffic they claim to bound",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def window_sweep(ctx):
    """Does the traffic change at the boundary the schedule asserts?

    A correct window has a sharp edge: the rate inside is many times the rate
    just outside. Equal rates on both sides of a start mean the window begins
    partway into an attack already in progress -- flows before it are attack
    traffic sitting in the benign class. Equal rates either side of an end mean
    the attack was still running when the schedule said it stopped.

    Where an edge is missing, the check reports the offset at which the traffic
    *does* change, which is the correction to apply to the manifest.
    """
    span = int(ctx.threshold("sweep_span_s", 600))
    bucket = int(ctx.threshold("sweep_bucket_s", 10))
    ratio_min = float(ctx.threshold("edge_ratio_min", 5.0))
    table, q = _sweep(ctx, span, bucket)

    findings = []
    for r in ctx.rules:
        hist = table.get(r["rid"])
        if not hist:
            continue                      # silent rules are sch.rule_yield's business
        win_buckets = max(1, int((r["t1"] - r["t0"]) / bucket))
        span_buckets = max(1, int(span / bucket))
        rate_in = _rate(hist, 0, win_buckets, bucket)
        rate_before = _rate(hist, -span_buckets, 0, bucket)
        rate_after = _rate(hist, win_buckets, win_buckets + span_buckets, bucket)
        if rate_in <= 0:
            continue

        # Where does the traffic actually change? Largest forward jump and
        # largest forward drop in the per-bucket rate series. Only reported when
        # the step is a material fraction of the in-window rate: in a window
        # whose traffic never stops inside the swept range there is no real
        # edge, and the largest of a set of tiny fluctuations is not one.
        ser = _series(hist, -span_buckets, win_buckets + span_buckets, bucket)
        diffs = [(ser[i + 1][1] - ser[i][1], ser[i + 1][0]) for i in range(len(ser) - 1)]
        floor = 0.1 * rate_in
        rise = max(diffs, default=(0, 0))
        drop = min(diffs, default=(0, 0))
        rise_at = rise[1] * bucket if rise[0] >= floor else None
        drop_at = drop[1] * bucket if -drop[0] >= floor else None

        # Capped so a divide-by-nothing reads as "no traffic outside" rather
        # than as a spurious 1e11.
        start_ratio = (rate_in / rate_before) if rate_before > 0 else float("inf")
        end_ratio = (rate_in / rate_after) if rate_after > 0 else float("inf")
        problems = []
        if start_ratio < ratio_min:
            problems.append("start")
        if end_ratio < ratio_min:
            problems.append("end")
        if not problems:
            continue

        metrics = {
            "rule_id": r["rid"], "raw": r["name"], "class": r["canonical"],
            "window_local": f"{r['date']} {r['start']}-{r['end']}",
            "window_seconds": round(r["t1"] - r["t0"], 1),
            "rate_in_per_s": round(rate_in, 3),
            "rate_before_per_s": round(rate_before, 3),
            "rate_after_per_s": round(rate_after, 3),
            "start_edge_ratio": None if start_ratio == float("inf") else round(start_ratio, 2),
            "end_edge_ratio": None if end_ratio == float("inf") else round(end_ratio, 2),
            "edge_ratio_min": ratio_min,
            "detected_rise_offset_s": rise_at,
            "detected_drop_offset_s": drop_at,
            "flows_before_window": sum(v for k, v in hist.items() if k < 0),
            "flows_after_window": sum(v for k, v in hist.items() if k >= win_buckets),
        }
        which = " and ".join(problems)
        sev = (Severity.CRITICAL
               if metrics["flows_before_window"] + metrics["flows_after_window"] > 10_000
               else Severity.MAJOR)
        findings.append(Finding(
            check_id="sch.window_sweep", dataset=ctx.dataset, severity=sev,
            title=f"{r['name']} {r['date']} {r['start']}-{r['end']}: no traffic edge at "
                  f"window {which}",
            detail=f"The rate inside this window is {rate_in:,.1f} flows/s against "
                   f"{rate_before:,.1f} before and {rate_after:,.1f} after. A correct "
                   f"window separates two regimes; this one does not at its {which}, so "
                   f"the schedule boundary is not where the traffic changes."
                   + (f" The rate actually rises {rise_at:+d}s from the declared start."
                      if rise_at is not None else "")
                   + (f" It falls {drop_at:+d}s from the declared start."
                      if drop_at is not None else ""),
            metrics=metrics,
            scope={"rule_id": r["rid"], "raw": r["name"], "class": r["canonical"]},
            evidence=[{"offset_s": k * bucket, "flows": v, "rate_per_s": round(v / bucket, 3)}
                      for k, v in sorted(hist.items())],
            repro=q.strip(),
        ))
    return findings


@check(id="sch.boundary_residue", tier=1,
       title="attack traffic does not continue past the scheduled end",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def boundary_residue(ctx):
    """Flows after t1 that look like the attack are labelled benign by default.

    This is the failure mode that put roughly 613k flows into 2019's benign
    class: a reflection flood does not stop dead at the scheduled minute, and
    every flow in its tail falls outside every window, so the closed-world
    assumption calls it benign. Quantified as the post-window rate against the
    in-window rate, so a slow trickle is distinguished from a flood still
    running at full volume.
    """
    span = int(ctx.threshold("sweep_span_s", 600))
    bucket = int(ctx.threshold("sweep_bucket_s", 10))
    max_ratio = float(ctx.threshold("residue_ratio_max", 0.10))
    table, q = _sweep(ctx, span, bucket)

    findings = []
    for r in ctx.rules:
        hist = table.get(r["rid"])
        if not hist:
            continue
        win_buckets = max(1, int((r["t1"] - r["t0"]) / bucket))
        span_buckets = max(1, int(span / bucket))
        rate_in = _rate(hist, 0, win_buckets, bucket)
        rate_after = _rate(hist, win_buckets, win_buckets + span_buckets, bucket)
        if rate_in <= 0:
            continue
        ratio = rate_after / rate_in
        if ratio <= max_ratio:
            continue
        residue = sum(v for k, v in hist.items() if k >= win_buckets)
        findings.append(Finding(
            check_id="sch.boundary_residue", dataset=ctx.dataset,
            severity=Severity.CRITICAL if residue > 10_000 else Severity.MAJOR,
            title=f"{r['name']} {r['date']} {r['end']}: {residue:,} flow(s) continue "
                  f"past the scheduled end",
            detail=f"For {span}s after this window closes, traffic to the declared "
                   f"endpoints continues at {100 * ratio:.1f}% of its in-window rate. "
                   f"Those flows match no rule and are benign by the closed-world "
                   f"assumption, so an attack tail is sitting in the benign class.",
            metrics={"rule_id": r["rid"], "raw": r["name"], "class": r["canonical"],
                     "residue_flows": residue,
                     "rate_in_per_s": round(rate_in, 3),
                     "rate_after_per_s": round(rate_after, 3),
                     "residue_ratio": round(ratio, 4),
                     "residue_ratio_max": max_ratio,
                     "window_end_utc": r["t1"]},
            scope={"rule_id": r["rid"], "raw": r["name"], "class": r["canonical"]},
            evidence=[{"offset_after_end_s": (k - win_buckets) * bucket, "flows": v}
                      for k, v in sorted(hist.items()) if k >= win_buckets],
            repro=q.strip(),
        ))
    return findings


@check(id="sch.intra_window_gaps", tier=1,
       title="scheduled windows are busy for their whole length",
       needs=("labelled",), max_severity=Severity.MAJOR)
def intra_window_gaps(ctx):
    """Idle stretches inside a window mean benign traffic labelled as attack.

    The exposure is the long windows: 2019's SYN rule runs six hours and its
    TFTP rule nearly four. Everything to the victim in those hours is labelled
    ddos. Where the attack was not actually running, that is benign traffic
    being given an attack label -- the mirror image of boundary residue.
    """
    span = int(ctx.threshold("sweep_span_s", 600))
    bucket = int(ctx.threshold("sweep_bucket_s", 10))
    gap_min = float(ctx.threshold("gap_min_seconds", 120))
    gap_frac = float(ctx.threshold("gap_rate_fraction", 0.05))
    table, q = _sweep(ctx, span, bucket)
    min_buckets = max(1, int(gap_min / bucket))

    findings = []
    for r in ctx.rules:
        hist = table.get(r["rid"])
        if not hist:
            continue
        win_buckets = max(1, int((r["t1"] - r["t0"]) / bucket))
        if win_buckets < min_buckets * 2:
            continue                       # too short for "a gap" to mean anything
        mean_rate = _rate(hist, 0, win_buckets, bucket)
        if mean_rate <= 0:
            continue
        floor = mean_rate * gap_frac

        gaps, run = [], None
        for k in range(win_buckets):
            quiet = (hist.get(k, 0) / bucket) < floor
            if quiet and run is None:
                run = k
            elif not quiet and run is not None:
                if k - run >= min_buckets:
                    gaps.append((run, k))
                run = None
        if run is not None and win_buckets - run >= min_buckets:
            gaps.append((run, win_buckets))
        if not gaps:
            continue

        idle_s = sum((b - a) * bucket for a, b in gaps)
        window_s = win_buckets * bucket
        findings.append(Finding(
            check_id="sch.intra_window_gaps", dataset=ctx.dataset,
            severity=Severity.MAJOR if idle_s > 0.25 * window_s else Severity.MINOR,
            title=f"{r['name']} {r['date']} {r['start']}-{r['end']}: {len(gaps)} idle "
                  f"stretch(es) totalling {idle_s / 60:.0f} min inside the window",
            detail=f"This window is {window_s / 60:.0f} minutes long but is quiet for "
                   f"{idle_s / 60:.0f} of them, at below {100 * gap_frac:.0f}% of its own "
                   f"mean rate. Traffic in those stretches is labelled "
                   f"{r['canonical']} because it falls inside the schedule, not because "
                   f"the attack was running.",
            metrics={"rule_id": r["rid"], "raw": r["name"], "class": r["canonical"],
                     "window_seconds": window_s, "idle_seconds": idle_s,
                     "idle_fraction": round(idle_s / window_s, 4),
                     "mean_rate_per_s": round(mean_rate, 3),
                     "quiet_floor_per_s": round(floor, 3), "gaps": len(gaps)},
            scope={"rule_id": r["rid"], "raw": r["name"], "class": r["canonical"]},
            evidence=[{"gap_start_offset_s": a * bucket, "gap_end_offset_s": b * bucket,
                       "seconds": (b - a) * bucket,
                       "flows": sum(hist.get(k, 0) for k in range(a, b))}
                      for a, b in gaps],
            repro=q.strip(),
        ))
    return findings


@check(id="sch.offset_calibration", tier=1,
       title="the assumed UTC offset is the one the capture was recorded at",
       needs=("labelled",), max_severity=Severity.BLOCKER)
def offset_calibration(ctx):
    """No CIC source states a timezone for any of these three datasets.

    Every manifest asserts a `utc_offset` that was inferred, not published. If
    it is wrong, every window in that dataset moves together by whole hours and
    the labels are systematically misassigned while still looking complete.

    Distinguishing an offset error from a schedule error is the point: a shift
    that fits *all* rules equally is the clock; a shift that fits one rule is
    that rule's transcription. Calibrated on short sharp rules only -- a
    six-hour flood has no edge to align against.
    """
    span = int(ctx.threshold("offset_span_s", 7200))
    bucket = int(ctx.threshold("sweep_bucket_s", 10))
    max_len = float(ctx.threshold("offset_max_rule_seconds", 1800))
    max_rules = int(ctx.threshold("offset_max_rules", 6))

    short = [r for r in ctx.rules if (r["t1"] - r["t0"]) <= max_len]
    short = sorted(short, key=lambda r: r["t1"] - r["t0"])[:max_rules]
    if not short:
        return []
    table, q = _sweep(ctx, span, bucket, rules=short)

    span_buckets = int(span / bucket)
    best = []
    for r in short:
        hist = table.get(r["rid"])
        if not hist:
            continue
        win_buckets = max(1, int((r["t1"] - r["t0"]) / bucket))
        # Slide the window across the swept range; the shift that captures the
        # most traffic is where this rule's window actually belongs. Candidate
        # positions that land on ANOTHER rule's window are excluded: the
        # endpoint predicate cannot tell two attacks between the same pair of
        # hosts apart, so without this the search happily "corrects" a quiet
        # rule onto a neighbouring attack's flood.
        others = [(o["t0"], o["t1"]) for o in ctx.rules if o["rid"] != r["rid"]]
        scores = []
        for shift in range(-span_buckets, span_buckets - win_buckets + 1):
            lo = r["t0"] + shift * bucket
            hi = lo + win_buckets * bucket
            if any(lo < o1 and o0 < hi for o0, o1 in others):
                continue
            scores.append((sum(hist.get(k, 0) for k in range(shift, shift + win_buckets)),
                           shift))
        if not scores:
            continue
        n_best, shift_best = max(scores)
        n_here = sum(hist.get(k, 0) for k in range(0, win_buckets))
        best.append({"rule_id": r["rid"], "raw": r["name"],
                     "window_local": f"{r['date']} {r['start']}-{r['end']}",
                     "best_shift_s": shift_best * bucket,
                     "flows_at_best_shift": n_best,
                     "flows_as_labelled": n_here,
                     "gain": n_best - n_here})
    if not best:
        return []

    # A shift is only evidence if moving the window there wins a materially
    # larger population. Without this floor the search reports a "correction"
    # worth a single flow, which is noise wearing the costume of a finding.
    gain_ratio = float(ctx.threshold("offset_gain_min_ratio", 2.0))
    gain_floor = float(ctx.threshold("offset_gain_min_flows", 500))
    for b in best:
        b["material"] = (b["best_shift_s"] != 0
                         and b["gain"] >= gain_floor
                         and b["flows_at_best_shift"] >= gain_ratio * max(b["flows_as_labelled"], 1))

    material = [b for b in best if b["material"]]
    if not material:
        return []

    shifts = [b["best_shift_s"] for b in material]
    agreed = max(set(shifts), key=shifts.count)
    agreement = shifts.count(agreed) / len(shifts)

    findings = []
    # A consistent non-zero shift across independent rules is a clock problem,
    # not a transcription problem, and it invalidates the whole dataset's labels.
    # It takes several rules to make that case: one rule agreeing with itself is
    # a misplaced window, and calling it a clock error would condemn a whole
    # dataset on the evidence of a single attack.
    min_rules = int(ctx.threshold("offset_min_rules_for_global", 3))
    if agreed != 0 and agreement >= 0.6 and shifts.count(agreed) >= min_rules:
        findings.append(Finding(
            check_id="sch.offset_calibration", dataset=ctx.dataset,
            severity=Severity.BLOCKER,
            title=f"every calibrated rule wants the same {agreed:+d}s shift — the UTC "
                  f"offset is wrong",
            detail=f"{shifts.count(agreed)} of {len(shifts)} independently calibrated "
                   f"rules place their traffic at the same offset from where the "
                   f"manifest puts them. A shift shared by unrelated attacks is the "
                   f"capture clock, not the schedule. No CIC source publishes a timezone "
                   f"for this dataset, so the manifest's utc_offset is an assumption and "
                   f"this is evidence against it.",
            metrics={"agreed_shift_s": agreed, "agreed_shift_hours": round(agreed / 3600, 3),
                     "rule_agreement": round(agreement, 3),
                     "rules_calibrated": len(best), "rules_material": len(material)},
            evidence=best, repro=q.strip(),
        ))
    else:
        for b in material:
            findings.append(Finding(
                check_id="sch.offset_calibration", dataset=ctx.dataset,
                severity=Severity.MAJOR,
                title=f"{b['raw']} {b['window_local']}: window is best placed "
                      f"{b['best_shift_s']:+d}s from where the manifest puts it",
                detail="This rule alone wants moving, while the others sit correctly, so "
                       "this is a transcription error in one window rather than a "
                       "dataset-wide clock offset.",
                metrics=b, scope={"rule_id": b["rule_id"], "raw": b["raw"]},
                repro=q.strip(),
            ))
    return findings


# ------------------------------------------------- matching-rule correctness

@check(id="sch.interval_overlap_delta", tier=1,
       title="long-lived flows straddling a window boundary are labelled correctly",
       needs=("labelled",), max_severity=Severity.MAJOR)
def interval_overlap_delta(ctx):
    """Labelling matches on flow START only, so a flow that begins one second
    before a window and runs ten minutes into it is benign.

    Slow-DoS is the exposure by construction: Slowloris and SlowHTTPTest work by
    opening connections and holding them. This re-labels by interval overlap
    (`ts < t1 AND ts + duration >= t0`) and reports how many flows change,
    which is the size of the error the start-only rule introduces.
    """
    if "duration" not in ctx.columns:
        return []
    q = f"""
    WITH rules AS ({rules_sql(ctx.rules)}),
    conn AS (SELECT * FROM read_parquet('{ctx.q('labelled')}', union_by_name = true)),
    ov AS (
      SELECT c.uid, min(r.rid) AS rid
      FROM conn c JOIN rules r
        ON c.ts < r.t1 AND (c.ts + coalesce(c.duration, 0)) >= r.t0 AND {ENDPOINTS}
      GROUP BY c.uid)
    SELECT c.label_class, r.rclass AS overlap_class,
           count(*) AS n, sum(coalesce(c.duration, 0)) AS total_duration
    FROM conn c LEFT JOIN ov USING (uid)
    LEFT JOIN rules r ON r.rid = ov.rid
    WHERE coalesce(r.rclass, 'benign') <> c.label_class
    GROUP BY 1, 2 ORDER BY n DESC"""
    rows = ctx.lake.sql(q)
    if not rows:
        return []
    total = ctx.lake.one(f"SELECT count(*) FROM read_parquet('{ctx.q('labelled')}', "
                         f"union_by_name=true)")[0]
    flips = sum(r[2] for r in rows)
    rate = flips / total if total else 0.0
    warn = float(ctx.threshold("overlap_flip_rate_warn", 0.01))
    return [Finding(
        check_id="sch.interval_overlap_delta", dataset=ctx.dataset,
        severity=Severity.MAJOR if rate > warn else Severity.MINOR,
        title=f"{flips:,} flow(s) ({rate:.3%}) change label when matched by interval "
              f"overlap instead of start time",
        detail="A flow is currently an attack only if it STARTED inside the window. "
               "Matching on overlap instead -- the flow was alive at some point during "
               "the attack -- moves these flows. Slow-DoS attacks hold connections open "
               "by design, so for those the start-only rule systematically under-labels.",
        metrics={"flows_total": total, "flows_flipped": flips,
                 "flip_rate": round(rate, 6), "flip_rate_warn": warn},
        evidence=[{"labelled_as": r[0], "overlap_says": r[1], "flows": r[2],
                   "total_duration_s": round(r[3] or 0, 1)} for r in rows],
        repro=q.strip(),
    )]


@check(id="sch.endpoint_coverage", tier=1,
       title="in-window traffic involves the declared attacker and victim",
       needs=("labelled",), max_severity=Severity.MAJOR)
def endpoint_coverage(ctx):
    """Who else is talking during the attack window?

    A rule that matches only a fraction of the traffic in its own window usually
    has a missing NAT alias: the attacker is present but under a different
    address. That failure already cost 2017's DDoS-LOIT rule 94,614 flows before
    the alias was added, and it produces no error -- just a quiet rule and a
    fat benign class.
    """
    q = f"""
    WITH rules AS ({rules_sql(ctx.rules)}),
    conn AS (SELECT * FROM read_parquet('{ctx.q('labelled')}', union_by_name = true)),
    inwin AS (
      SELECT r.rid, r.rname, c."id.orig_h" AS orig, c."id.resp_h" AS resp,
             {ENDPOINTS} AS endpoint_match
      FROM conn c JOIN rules r ON c.ts >= r.t0 AND c.ts < r.t1)
    SELECT rid, rname,
           count(*) AS in_window,
           sum(CASE WHEN endpoint_match THEN 1 ELSE 0 END) AS matched
    FROM inwin GROUP BY 1, 2 ORDER BY in_window DESC"""
    rows = ctx.lake.sql(q)
    warn = float(ctx.threshold("unexplained_peer_share_warn", 0.10))

    findings = []
    for rid, rname, in_window, matched in rows:
        if not in_window:
            continue
        unexplained = in_window - matched
        share = unexplained / in_window
        if share <= warn or unexplained < 1000:
            continue
        top = ctx.lake.sql(f"""
        WITH rules AS ({rules_sql([r for r in ctx.rules if r['rid'] == rid])}),
        conn AS (SELECT * FROM read_parquet('{ctx.q('labelled')}', union_by_name = true))
        SELECT c."id.orig_h" AS orig, c."id.resp_h" AS resp, count(*) AS n,
               any_value(c.label_class) AS labelled
        FROM conn c JOIN rules r ON c.ts >= r.t0 AND c.ts < r.t1
        WHERE NOT {ENDPOINTS}
        GROUP BY 1, 2 ORDER BY n DESC LIMIT 15""")
        r = next(x for x in ctx.rules if x["rid"] == rid)
        findings.append(Finding(
            check_id="sch.endpoint_coverage", dataset=ctx.dataset, severity=Severity.MAJOR,
            title=f"{rname} {r['date']} {r['start']}-{r['end']}: {share:.0%} of in-window "
                  f"traffic involves neither declared endpoint",
            detail="Traffic inside the scheduled window that does not touch the declared "
                   "attacker or victim is labelled benign. When it is this large a share, "
                   "the usual cause is an address the manifest does not list -- a NAT "
                   "alias, a second capture point, or a wrong victim IP.",
            metrics={"rule_id": rid, "raw": rname, "in_window": in_window,
                     "endpoint_matched": matched, "unexplained": unexplained,
                     "unexplained_share": round(share, 4), "warn_above": warn},
            scope={"rule_id": rid, "raw": rname},
            evidence=[{"orig": t[0], "resp": t[1], "flows": t[2], "labelled": t[3]}
                      for t in top],
            repro=q.strip(),
        ))
    return findings


@check(id="sch.extension_audit", tier=1,
       title="the +60s window extension is applied consistently",
       needs=("labelled",), max_severity=Severity.MINOR)
def extension_audit(ctx):
    """Every window is extended to the end of its final minute, clamped at the
    next window's start. The clamp is silent, so a back-to-back pair leaves one
    rule with no extension at all while its neighbours all get one -- 2019's
    WebDDoS is clamped to zero by the SYN rule starting the same minute.
    """
    clamped = [r for r in ctx.rules if r.get("extended_by", 60) < 60]
    if not clamped:
        return []
    return [Finding(
        check_id="sch.extension_audit", dataset=ctx.dataset, severity=Severity.MINOR,
        title=f"{len(clamped)} rule(s) received a reduced end-of-minute extension",
        detail="Schedules are minute-granular, so each window is extended through the end "
               "of its final minute -- except where the next window starts first, which "
               "clamps it. Those rules lose up to a minute of their own traffic to the "
               "neighbouring class or to benign. Not wrong, but it means those boundaries "
               "are decided by the clamp rather than by the schedule.",
        metrics={"rules_clamped": len(clamped)},
        evidence=[{"rule_id": r["rid"], "raw": r["name"],
                   "window_local": f"{r['date']} {r['start']}-{r['end']}",
                   "extended_by_s": r.get("extended_by"),
                   "lost_s": 60 - r.get("extended_by", 60)} for r in clamped],
    )]


@check(id="sch.rule_yield", tier=1, title="every rule matches a plausible amount of traffic",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def rule_yield(ctx):
    """A rule matching nothing, or almost nothing, has not found its attack.

    Reported against the labelled zone rather than recomputed, so this is
    exactly what the labels say. The per-minute rate matters as much as the
    count: three flows over a 41-minute window is a different problem from
    three flows over ten seconds.
    """
    rows = ctx.lake.sql(f"""
        SELECT rule_id, any_value(label_raw) AS raw, any_value(label_class) AS cls,
               count(*) AS n
        FROM read_parquet('{ctx.q('labelled')}', union_by_name = true)
        WHERE rule_id IS NOT NULL GROUP BY rule_id""")
    got = {int(r[0]): {"raw": r[1], "class": r[2], "n": int(r[3])} for r in rows}
    thin_n = float(ctx.threshold("rule_thin_flows", 10))
    thin_rate = float(ctx.threshold("rule_thin_flows_per_minute", 1.0))

    findings = []
    for r in ctx.rules:
        rec = got.get(r["rid"])
        minutes = max((r["t1"] - r["t0"]) / 60.0, 1e-6)
        n = rec["n"] if rec else 0
        per_min = n / minutes
        if n == 0:
            sev, why = Severity.CRITICAL, "matched nothing at all"
        elif n < thin_n or per_min < thin_rate:
            sev, why = Severity.MAJOR, f"matched only {n:,} flow(s) ({per_min:.2f}/min)"
        else:
            continue
        findings.append(Finding(
            check_id="sch.rule_yield", dataset=ctx.dataset, severity=sev,
            title=f"{r['name']} {r['date']} {r['start']}-{r['end']} {why}",
            detail="A scheduled attack that produced no traffic in the labelled zone was "
                   "either not captured, or is being looked for at the wrong time or "
                   "against the wrong addresses. Either way the class it belongs to is "
                   "not represented by the flows the label claims.",
            metrics={"rule_id": r["rid"], "raw": r["name"], "class": r["canonical"],
                     "flows": n, "window_minutes": round(minutes, 1),
                     "flows_per_minute": round(per_min, 3),
                     "thin_below_flows": thin_n, "thin_below_per_minute": thin_rate},
            scope={"rule_id": r["rid"], "raw": r["name"], "class": r["canonical"]},
            repro=f"SELECT rule_id, count(*) FROM read_parquet('{ctx.q('labelled')}') "
                  f"WHERE rule_id = {r['rid']} GROUP BY 1",
        ))
    return findings


@check(id="sch.rule_ambiguity", tier=1, title="no two rules can claim the same flow",
       max_severity=Severity.MAJOR)
def rule_ambiguity(ctx):
    """Rules overlapping in both time and endpoints are decided by min(rule_id).

    That tie-break is deterministic but arbitrary -- it encodes manifest order,
    not evidence. Any flow in an overlap is labelled by whichever rule happens
    to appear first in the YAML, so an overlap is a latent mislabelling even
    when no flow currently falls in it.
    """
    def ips(r):
        return set(r["attackers"] or []) | set(r["victims"] or [])

    pairs = []
    for i, a in enumerate(ctx.rules):
        for b in ctx.rules[i + 1:]:
            if a["t0"] >= b["t1"] or b["t0"] >= a["t1"]:
                continue
            shared = ips(a) & ips(b)
            if not (shared or a["prefix"] or b["prefix"]):
                continue
            overlap = min(a["t1"], b["t1"]) - max(a["t0"], b["t0"])
            pairs.append({"rule_a": a["rid"], "raw_a": a["name"], "class_a": a["canonical"],
                          "rule_b": b["rid"], "raw_b": b["name"], "class_b": b["canonical"],
                          "overlap_seconds": round(overlap, 1),
                          "shared_endpoints": sorted(shared),
                          "winner_by_min_rid": min(a["rid"], b["rid"])})
    if not pairs:
        return []
    cross_class = [p for p in pairs if p["class_a"] != p["class_b"]]
    return [Finding(
        check_id="sch.rule_ambiguity", dataset=ctx.dataset,
        severity=Severity.MAJOR if cross_class else Severity.MINOR,
        title=f"{len(pairs)} rule pair(s) overlap in time and endpoints"
              + (f", {len(cross_class)} across different classes" if cross_class else ""),
        detail="Where two rules can both claim a flow, the lower rule id wins. That is "
               "manifest ordering, not evidence. Overlaps spanning two different canonical "
               "classes are the ones that matter: which class a flow lands in is then "
               "decided by YAML order.",
        metrics={"overlapping_pairs": len(pairs), "cross_class_pairs": len(cross_class)},
        evidence=pairs,
    )]


@check(id="sch.class_day_conformance", tier=1,
       title="no class appears on a day that schedules none of it",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def class_day_conformance(ctx):
    """Attack labels outside their scheduled capture are impossible by construction.

    Labelling can only assign a class inside one of its own windows, so a
    violation here means the capture-to-day mapping is wrong, or two captures
    were merged, or a timestamp did not survive conversion. It is a strong
    integrity signal precisely because the labelling logic cannot produce it.
    """
    rows = ctx.lake.sql(f"""
        SELECT capture, label_class, count(*) AS n, min(ts) AS ts_min, max(ts) AS ts_max
        FROM read_parquet('{ctx.q('labelled')}', union_by_name = true)
        WHERE label_class <> 'benign' GROUP BY 1, 2""")
    # What the schedule permits: a class may appear only within one of its windows.
    windows = {}
    for r in ctx.rules:
        windows.setdefault(r["canonical"], []).append((r["t0"], r["t1"]))

    bad = []
    for cap, cls, n, tmin, tmax in rows:
        spans = windows.get(cls, [])
        if not spans:
            bad.append({"capture": cap, "class": cls, "flows": n,
                        "reason": "class has no scheduled window at all"})
            continue
        lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
        if tmin < lo or tmax > hi:
            bad.append({"capture": cap, "class": cls, "flows": n,
                        "ts_min": tmin, "ts_max": tmax,
                        "scheduled_from": lo, "scheduled_to": hi,
                        "reason": "flows fall outside every window for this class"})
    if not bad:
        return []
    return [Finding(
        check_id="sch.class_day_conformance", dataset=ctx.dataset, severity=Severity.CRITICAL,
        title=f"{len(bad)} (capture, class) group(s) fall outside their schedule",
        detail="Labelling assigns a class only inside that class's own windows, so flows "
               "carrying it outside them cannot have come from the matching logic. The "
               "capture-to-day mapping, or the timestamps, are wrong.",
        metrics={"groups": len(bad), "flows": sum(b["flows"] for b in bad)},
        evidence=bad,
    )]


@check(id="sch.duration_fits_window", tier=1,
       title="no flow outlives the window that labelled it",
       needs=("labelled",), max_severity=Severity.MAJOR)
def duration_fits_window(ctx):
    """A flow longer than its own attack window was not produced by that attack.

    Matching is on flow START only, so a flow that begins inside a window keeps
    the label however long it then runs. Where it runs longer than the entire
    window, one of two things is true and neither supports the label: the flow was
    already in progress and the window merely caught its start, or the window is
    far shorter than the activity it claims to bound.

    CSE-CIC-IDS-2018 `infiltration` is the measured case. Its longest flow is
    13,254.7 s -- 3.68 hours -- against a longest scheduled infiltration window of
    5,820 s (97 minutes), a 2.28x overrun. The class is 8 flows, all of them
    marked label_executed = true, and none of them a completed connection:
    conn_state is OTH for 50%, S1 for 37.5% and RSTR for 12.5%. A whole class of
    overrunning flows that never finished a handshake is mid-stream traffic
    captured by a time window, not an attack; a single long download would look
    quite different, which is why the conn_state mix travels with the finding.
    """
    if "duration" not in ctx.columns or "label_class" not in ctx.columns:
        return []
    ratio = float(ctx.threshold("duration_overrun_ratio", 1.0))

    # The longest window the schedule gives each class, and which rule it is.
    longest = {}
    for r in ctx.rules:
        span = r["t1"] - r["t0"]
        if span > longest.get(r["canonical"], (0.0, None))[0]:
            longest[r["canonical"]] = (
                span, f"{r['name']} {r['date']} {r['start']}-{r['end']}")
    if not longest:
        return []

    # One CASE keyed on the class puts each class's own limit into the same scan,
    # so the overrun count and the conn_state mix of the overrunning flows cost
    # nothing beyond the aggregate that was needed anyway.
    limit = ("CASE label_class " + " ".join(
        f"WHEN '{cls}' THEN {span * ratio}" for cls, (span, _) in longest.items())
        + " END")
    inlist = ", ".join(f"'{cls}'" for cls in longest)
    states = (f"histogram(conn_state) FILTER (WHERE duration > {limit})"
              if "conn_state" in ctx.columns else "CAST(NULL AS VARCHAR)")
    q = f"""
    SELECT label_class, count(*) AS flows,
           max(duration) AS duration_max,
           approx_quantile(duration, 0.99) AS duration_p99,
           count(*) FILTER (WHERE duration > {limit}) AS overrunning,
           {states} AS states
    FROM read_parquet('{ctx.q('labelled')}', union_by_name = true)
    WHERE label_class IN ({inlist}) AND duration IS NOT NULL
    GROUP BY 1 ORDER BY 3 DESC"""
    rows = ctx.lake.sql(q)

    findings = []
    for cls, flows, dur_max, dur_p99, overrunning, states_map in rows:
        window_s, rule = longest[cls]
        if dur_max is None or dur_max <= window_s * ratio:
            continue
        dist = dict(states_map) if states_map else {}
        seen = sum(dist.values())
        # SF is the only Zeek state meaning a connection that both established and
        # closed normally. A long SF flow is a plausible session; an overrunning
        # class with no SF in it never completed one, which is the 2018
        # infiltration signature and a much stronger claim against the label.
        completed = dist.get("SF", 0)
        share = completed / seen if seen else 0.0
        findings.append(Finding(
            check_id="sch.duration_fits_window", dataset=ctx.dataset,
            severity=Severity.MAJOR if share < 0.5 else Severity.MINOR,
            title=f"{cls}: longest flow runs {dur_max:,.1f}s against a {window_s:,.0f}s "
                  f"window ({dur_max / window_s:.2f}x)",
            detail=f"{overrunning:,} {cls} flow(s) last longer than the longest window "
                   f"the schedule gives that class ({rule}). Labelling matches on flow "
                   f"start, so such a flow either began before the attack and was caught "
                   f"mid-stream by the window, or the window is far too short for what it "
                   f"claims to cover. Either way the label is not supported by the "
                   f"schedule that assigned it."
                   + (f" {seen - completed} of the {seen} overrunning flow(s) never "
                      f"completed a connection ({', '.join(f'{k} {v}' for k, v in sorted(dist.items(), key=lambda kv: -kv[1]))}), "
                      f"so this is not one long session running over."
                      if seen and share < 0.5 else ""),
            metrics={"class": cls, "class_flows": flows,
                     "longest_window_s": round(window_s, 1),
                     "longest_window_rule": rule,
                     "duration_max_s": round(dur_max, 1),
                     "duration_p99_s": round(dur_p99, 1) if dur_p99 is not None else None,
                     "overrun_ratio": round(dur_max / window_s, 3),
                     "overrunning_flows": int(overrunning or 0),
                     "overrunning_completed": completed,
                     "overrunning_completed_share": round(share, 4),
                     "duration_overrun_ratio": ratio},
            scope={"class": cls},
            evidence=[{"conn_state": k, "overrunning_flows": v,
                       "share": round(v / seen, 4) if seen else None}
                      for k, v in sorted(dist.items(), key=lambda kv: -kv[1])],
            repro=q.strip(),
        ))
    return findings
