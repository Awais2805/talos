#!/usr/bin/env python3
"""Tier 0 -- does the labelled dataset hold together at all?

These are arithmetic, not judgement. If the labelled zone has fewer rows than
the source it was built from, or carries a manifest hash that no longer exists
on disk, then every number produced downstream -- every EDA histogram, every
class count, every accuracy figure -- describes something other than what we
think we are looking at. Nothing else in this module runs until these pass.
"""

from src.preprocess.label.label_flows import MATCH, rules_sql
from src.label_validation.core.finding import Finding, Severity
from src.label_validation.core.registry import check

# Columns the labelling stage is contractually required to add. Their absence
# means the labelled zone was written by something other than label_flows.py.
LABEL_COLUMNS = ["label_binary", "label_class", "label_raw", "rule_id",
                 "label_executed", "label_source", "rules_matched",
                 "manifest_sha", "taxonomy_sha", "labelled_at", "capture"]

KEY_COLUMNS = ["ts", "uid", "id.orig_h", "id.resp_h", "proto"]


@check(id="int.schema_stability", tier=0, title="labelled zone carries every expected column",
       needs=("labelled",), max_severity=Severity.BLOCKER)
def schema_stability(ctx):
    missing = [c for c in LABEL_COLUMNS if c not in ctx.columns]
    if not missing:
        return []
    return [Finding(
        check_id="int.schema_stability", dataset=ctx.dataset, severity=Severity.BLOCKER,
        title=f"labelled zone is missing {len(missing)} label column(s)",
        detail="The labelled zone does not carry columns that label_flows.py always "
               "writes. Either it was produced by a different version of the stage, or "
               "the glob is pointing at the wrong zone.",
        metrics={"missing_count": len(missing)},
        evidence=[{"column": c} for c in missing],
        repro=f"DESCRIBE SELECT * FROM read_parquet('{ctx.q('labelled')}', union_by_name=true)",
    )]


@check(id="int.row_parity", tier=0, title="labelled row count matches its source",
       needs=("labelled", "conn"), max_severity=Severity.BLOCKER)
def row_parity(ctx):
    """Labelling is a LEFT JOIN onto conn, so it must be row-preserving.

    A shortfall means flows were dropped; an excess means the join fanned out
    and some flows are counted more than once. Both silently corrupt every class
    count downstream, and neither raises an error at labelling time.
    """
    n_lab = ctx.lake.one(f"SELECT count(*) FROM read_parquet('{ctx.q('labelled')}', "
                         f"union_by_name=true)")[0]
    n_src = ctx.lake.one(f"SELECT count(*) FROM read_parquet('{ctx.q('conn')}', "
                         f"union_by_name=true)")[0]
    if n_lab == n_src:
        return []
    delta = n_lab - n_src
    return [Finding(
        check_id="int.row_parity", dataset=ctx.dataset, severity=Severity.BLOCKER,
        title=f"labelled zone has {abs(delta):,} {'more' if delta > 0 else 'fewer'} "
              f"rows than its source",
        detail="Labelling attaches columns to every conn flow via a LEFT JOIN and must "
               "preserve the row count exactly. A difference means flows were dropped or "
               "duplicated, so every per-class count derived from this zone is wrong.",
        metrics={"labelled_rows": n_lab, "source_rows": n_src, "delta": delta,
                 "delta_rate": round(delta / n_src, 9) if n_src else None},
        repro=f"SELECT count(*) FROM read_parquet('{ctx.q('labelled')}');\n"
              f"SELECT count(*) FROM read_parquet('{ctx.q('conn')}');",
    )]


@check(id="int.uid_unique", tier=0, title="Zeek uids are unique within the dataset",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def uid_unique(ctx):
    """A repeated uid means the same flow was ingested twice.

    Zeek mints uids per capture, so duplicates point at overlapping pcap splits
    or a log file converted to parquet more than once. Duplicated flows inflate
    whichever class they belong to and leak across any train/test split made by
    sampling rows.
    """
    n, d = ctx.lake.one(f"SELECT count(*), count(DISTINCT uid) FROM "
                        f"read_parquet('{ctx.q('labelled')}', union_by_name=true)")
    dupes = n - d
    if dupes <= 0:
        return []
    rate = dupes / n
    worst = ctx.lake.sql(f"""
        SELECT uid, count(*) AS n, count(DISTINCT capture) AS captures,
               count(DISTINCT label_class) AS classes
        FROM read_parquet('{ctx.q('labelled')}', union_by_name=true)
        GROUP BY uid HAVING count(*) > 1 ORDER BY n DESC LIMIT 20""")
    # A duplicate carrying two different labels is worse than a plain duplicate:
    # it is a flow the dataset disagrees with itself about.
    conflicting = sum(1 for r in worst if r[3] > 1)
    return [Finding(
        check_id="int.uid_unique", dataset=ctx.dataset,
        severity=Severity.CRITICAL if rate > ctx.threshold("uid_duplicate_rate_max", 0.0)
        else Severity.MINOR,
        title=f"{dupes:,} duplicate uid(s) ({rate:.4%} of flows)",
        detail="Zeek uids identify a connection uniquely within a capture. Repeats mean "
               "the same flow entered the lake more than once -- overlapping pcap splits, "
               "or a log converted twice. Duplicates inflate their class and leak across "
               "any row-sampled split."
               + (f" {conflicting} of the sampled duplicates carry more than one "
                  f"label_class." if conflicting else ""),
        metrics={"rows": n, "distinct_uids": d, "duplicates": dupes,
                 "duplicate_rate": round(rate, 9),
                 "sampled_with_conflicting_labels": conflicting},
        evidence=[{"uid": r[0], "n": r[1], "captures": r[2], "label_classes": r[3]}
                  for r in worst],
        repro=f"SELECT uid, count(*) FROM read_parquet('{ctx.q('labelled')}') "
              f"GROUP BY uid HAVING count(*) > 1",
    )]


@check(id="int.key_nulls", tier=0, title="identity and timing columns are never null",
       needs=("labelled",), max_severity=Severity.BLOCKER)
def key_nulls(ctx):
    cols = [c for c in KEY_COLUMNS if c in ctx.columns]
    if not cols:
        return []
    sel = ", ".join(f'sum(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)' for c in cols)
    row = ctx.lake.one(f"SELECT count(*), {sel} FROM "
                       f"read_parquet('{ctx.q('labelled')}', union_by_name=true)")
    total, counts = row[0], row[1:]
    bad = [(c, n) for c, n in zip(cols, counts) if n]
    if not bad:
        return []
    return [Finding(
        check_id="int.key_nulls", dataset=ctx.dataset, severity=Severity.BLOCKER,
        title=f"{len(bad)} identity/timing column(s) contain nulls",
        detail="Labelling matches on ts and endpoint addresses. A null in any of them "
               "means the flow could never have matched a rule and fell to benign by "
               "default, regardless of what it actually was.",
        metrics={"rows": total,
                 **{f"null_{c}": n for c, n in bad},
                 **{f"null_rate_{c}": round(n / total, 9) for c, n in bad if total}},
        evidence=[{"column": c, "nulls": n, "rate": round(n / total, 9) if total else None}
                  for c, n in bad],
        repro=f"SELECT {sel} FROM read_parquet('{ctx.q('labelled')}')",
    )]


@check(id="int.label_wellformed", tier=0, title="label columns are mutually consistent",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def label_wellformed(ctx):
    """The label columns encode the same fact several ways; they must agree.

    label_binary, label_class, label_raw, rule_id, rules_matched and label_source
    are all derived from whether a rule matched. Any disagreement between them
    means the labelling query itself is wrong, which no amount of downstream
    care can repair.
    """
    q = f"""
    SELECT count(*) AS n,
      sum(CASE WHEN label_binary <> (label_class <> 'benign') THEN 1 ELSE 0 END),
      sum(CASE WHEN (rule_id IS NULL) <> (label_class = 'benign') THEN 1 ELSE 0 END),
      sum(CASE WHEN (label_raw IS NULL) <> (label_class = 'benign') THEN 1 ELSE 0 END),
      sum(CASE WHEN (label_class = 'benign') AND coalesce(rules_matched, 0) <> 0
               THEN 1 ELSE 0 END),
      sum(CASE WHEN label_source <> CASE WHEN label_class = 'benign'
               THEN 'schedule-closed-world' ELSE 'schedule' END THEN 1 ELSE 0 END),
      sum(CASE WHEN label_class = 'benign' AND label_executed IS NOT NULL
               THEN 1 ELSE 0 END)
    FROM read_parquet('{ctx.q('labelled')}', union_by_name=true)"""
    n, binary, rid, raw, rmatched, src, exec_benign = ctx.lake.one(q)
    invariants = [
        ("label_binary disagrees with label_class", binary),
        ("rule_id null-ness disagrees with label_class", rid),
        ("label_raw null-ness disagrees with label_class", raw),
        ("benign flow carries rules_matched > 0", rmatched),
        ("label_source disagrees with label_class", src),
        ("benign flow carries a non-null label_executed", exec_benign),
    ]
    broken = [(name, cnt) for name, cnt in invariants if cnt]
    if not broken:
        return []
    return [Finding(
        check_id="int.label_wellformed", dataset=ctx.dataset, severity=Severity.CRITICAL,
        title=f"{len(broken)} label invariant(s) violated",
        detail="The label columns restate the same fact in several forms and must agree. "
               "Disagreement means the labelling query is producing rows it did not "
               "intend to, so the labels cannot be reasoned about column by column.",
        metrics={"rows": n, **{f"violations_{i}": c for i, (_, c) in enumerate(broken)}},
        evidence=[{"invariant": name, "violations": cnt,
                   "rate": round(cnt / n, 9) if n else None} for name, cnt in broken],
        repro=q.strip(),
    )]


@check(id="int.provenance_current", tier=0,
       title="labels were produced by the manifests currently on disk",
       needs=("labelled",), max_severity=Severity.BLOCKER)
def provenance_current(ctx):
    """Content hashes in the data vs the files that are supposed to explain it.

    This is the stale-label detector, and the only tamper detector we have. If a
    manifest has been edited since labelling ran, the labels in the lake no
    longer correspond to any schedule anyone can read -- and every finding this
    module produces would be judging the data against the wrong rules.
    """
    rows = ctx.lake.sql(f"""
        SELECT manifest_sha, taxonomy_sha, count(*) AS n
        FROM read_parquet('{ctx.q('labelled')}', union_by_name=true)
        GROUP BY 1, 2 ORDER BY n DESC""")
    findings = []
    if len(rows) > 1:
        findings.append(Finding(
            check_id="int.provenance_current", dataset=ctx.dataset,
            severity=Severity.BLOCKER,
            title=f"labelled zone mixes {len(rows)} manifest/taxonomy versions",
            detail="Different flows in the same zone were labelled from different "
                   "manifests. The zone is a blend of two labelling runs and no single "
                   "schedule explains it.",
            metrics={"distinct_provenance": len(rows)},
            evidence=[{"manifest_sha": r[0], "taxonomy_sha": r[1], "flows": r[2]}
                      for r in rows],
        ))
    stale = [r for r in rows
             if r[0] != ctx.manifest_sha or r[1] != ctx.taxonomy_sha]
    if stale:
        findings.append(Finding(
            check_id="int.provenance_current", dataset=ctx.dataset,
            severity=Severity.BLOCKER,
            title="labels are stale: manifest/taxonomy on disk has changed since labelling",
            detail="The labelled data records the manifest and taxonomy hashes it was "
                   "built from, and they no longer match the files on disk. Re-run "
                   "labelling before trusting anything derived from this zone.",
            metrics={"expected_manifest_sha": ctx.manifest_sha,
                     "expected_taxonomy_sha": ctx.taxonomy_sha,
                     "stale_flows": sum(r[2] for r in stale)},
            evidence=[{"manifest_sha": r[0], "taxonomy_sha": r[1], "flows": r[2]}
                      for r in stale],
            repro=f"SELECT DISTINCT manifest_sha, taxonomy_sha FROM "
                  f"read_parquet('{ctx.q('labelled')}')",
        ))
    return findings


@check(id="int.ts_in_capture", tier=0, title="flow timestamps fall inside their capture",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def ts_in_capture(ctx):
    """Every capture is one day of one testbed; its flows must be dated that day.

    A flow timestamped outside its own capture's span is a clock or conversion
    fault, and because labelling is a time-window match, such a flow is labelled
    against the wrong part of the schedule -- or falls outside every window and
    becomes benign by default.
    """
    rows = ctx.lake.sql(f"""
        SELECT capture, count(*) AS n,
               min(ts) AS ts_min, max(ts) AS ts_max,
               sum(CASE WHEN ts IS NULL OR ts <= 0 THEN 1 ELSE 0 END) AS bad_ts,
               count(DISTINCT strftime(to_timestamp(ts) AT TIME ZONE 'UTC', '%Y-%m-%d'))
                 AS utc_days
        FROM read_parquet('{ctx.q('labelled')}', union_by_name=true)
        GROUP BY capture ORDER BY capture""")
    findings, spans = [], []
    for cap, n, tmin, tmax, bad, days in rows:
        spans.append({"capture": cap, "flows": n, "ts_min": tmin, "ts_max": tmax,
                      "span_hours": round((tmax - tmin) / 3600, 2) if tmin and tmax else None,
                      "utc_days_touched": days, "nonpositive_ts": bad})
    broken = [s for s in spans if s["nonpositive_ts"]]
    if broken:
        findings.append(Finding(
            check_id="int.ts_in_capture", dataset=ctx.dataset, severity=Severity.CRITICAL,
            title=f"{sum(s['nonpositive_ts'] for s in broken):,} flow(s) have a "
                  f"null or non-positive timestamp",
            detail="A flow with no usable timestamp cannot be matched against the "
                   "schedule and is benign by default whatever it actually was.",
            metrics={"affected_captures": len(broken),
                     "flows": sum(s["nonpositive_ts"] for s in broken)},
            evidence=broken,
        ))
    # A capture spanning more than two UTC days is suspicious: CIC captures are
    # single working days, and three days means either a bad conversion or two
    # captures merged under one folder.
    sprawl = [s for s in spans if (s["utc_days_touched"] or 0) > 2]
    if sprawl:
        findings.append(Finding(
            check_id="int.ts_in_capture", dataset=ctx.dataset, severity=Severity.MAJOR,
            title=f"{len(sprawl)} capture(s) span more than two UTC days",
            detail="Each capture is one working day of one testbed. A wider span means "
                   "two captures merged under one folder, or timestamps that did not "
                   "survive conversion.",
            metrics={"captures": len(sprawl)}, evidence=sprawl,
        ))
    if not findings:
        return []
    return findings


# Zeek reports two byte counts per direction and they are not two measurements of
# the same thing: *_bytes is payload inferred from the TCP sequence-number delta,
# *_ip_bytes is summed from the IP headers actually seen. Payload is a subset of
# what crossed the wire, so the first can never exceed the second.
BYTE_PAIRS = [("orig", "orig_bytes", "orig_ip_bytes", "orig_pkts"),
              ("resp", "resp_bytes", "resp_ip_bytes", "resp_pkts")]

# Per side: violations, worst payload, worst wire, implied bytes/packet, worst uid.
PER_SIDE = 5


@check(id="int.bytes_physically_possible", tier=0,
       title="payload bytes never exceed the bytes seen on the wire",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def bytes_physically_possible(ctx):
    """Zeek's byte counters wrap, and they wrap in the classes that carry the label.

    `orig_bytes` is payload derived from the TCP sequence-number delta, so on
    scan and RST traffic -- where the peer never completes a handshake and the
    numbers run backwards -- it wraps to something enormous. `orig_ip_bytes` is
    counted off the IP headers that were actually observed, so it cannot. In
    CIC-IDS-2017 `portscan`, orig_bytes.max is 1,985,545,102 against an
    orig_ip_bytes.max of 21,258 and an orig_pkts.max of 119: an implied 16.7 MB
    per packet. The same wrap is in 2017 `dos` (1,567,578,927 bytes in at most 27
    packets) and in 2018 `botnet` and `benign`.

    A failure here is not a tidiness problem. `orig_bytes` carries the largest
    label correlation in 2017 (point-biserial -0.442) and the largest attack-class
    mutual information (0.835) -- the strongest single label signal in the dataset
    rides on a column that is corrupt precisely in the classes that produce it. Any
    behavioural expectation written against it, and any model trained on it, inherits
    the wrap and cannot be told apart from one that learned the attack. Where this
    fires, `orig_ip_bytes` / `resp_ip_bytes` are the counted columns and should be
    used in its place.
    """
    pairs = [p for p in BYTE_PAIRS if p[1] in ctx.columns and p[2] in ctx.columns]
    if not pairs or "label_class" not in ctx.columns:
        return []

    any_bad = " OR ".join(f'"{pay}" > "{wire}"' for _, pay, wire, _ in pairs)
    sel = [f"sum(CASE WHEN {any_bad} THEN 1 ELSE 0 END)"]
    for _side, pay, wire, pkts in pairs:
        bad = f'"{pay}" > "{wire}"'
        sel += [
            f"sum(CASE WHEN {bad} THEN 1 ELSE 0 END)",
            f'max("{pay}") FILTER (WHERE {bad})',
            f'max("{wire}") FILTER (WHERE {bad})',
            # Row-wise, so this is a real per-flow rate rather than one column's
            # maximum divided by another's.
            (f'max("{pay}" / nullif("{pkts}", 0)) FILTER (WHERE {bad})'
             if pkts in ctx.columns else "CAST(NULL AS DOUBLE)"),
            (f'arg_max(uid, "{pay}" - "{wire}") FILTER (WHERE {bad})'
             if "uid" in ctx.columns else "CAST(NULL AS VARCHAR)"),
        ]
    q = f"""
    SELECT label_class, count(*) AS n, {', '.join(sel)}
    FROM read_parquet('{ctx.q('labelled')}', union_by_name = true)
    GROUP BY label_class ORDER BY 2 DESC"""
    rows = ctx.lake.sql(q)

    crit = float(ctx.threshold("impossible_bytes_class_rate_critical", 0.001))
    evidence, flows_total, impossible_total, worst_rate = [], 0, 0, 0.0
    for row in rows:
        cls, n, bad_any = row[0], int(row[1]), int(row[2] or 0)
        flows_total += n
        if not bad_any:
            continue
        impossible_total += bad_any
        rate = bad_any / n if n else 0.0
        worst_rate = max(worst_rate, rate)
        rec = {"class": cls, "flows": n, "impossible_flows": bad_any,
               "impossible_rate": round(rate, 9)}
        for i, (side, _pay, _wire, _pkts) in enumerate(pairs):
            bad, pmax, wmax, bpp, uid = row[3 + i * PER_SIDE: 3 + (i + 1) * PER_SIDE]
            if not bad:
                continue
            rec.update({
                f"{side}_impossible": int(bad),
                f"{side}_payload_bytes_max": pmax,
                f"{side}_wire_bytes_max": wmax,
                f"{side}_implied_bytes_per_packet": round(bpp, 1) if bpp else None,
                f"{side}_worst_uid": uid,
            })
        evidence.append(rec)

    if not evidence:
        return []
    material = [e["class"] for e in evidence if e["impossible_rate"] > crit]
    return [Finding(
        check_id="int.bytes_physically_possible", dataset=ctx.dataset,
        severity=Severity.CRITICAL if material else Severity.MAJOR,
        title=f"{impossible_total:,} flow(s) report more payload bytes than wire bytes "
              f"across {len(evidence)} class(es)"
              + (f"; {', '.join(sorted(material))} materially affected" if material else ""),
        detail="Payload bytes are a subset of the bytes that crossed the wire, so a flow "
               "where orig_bytes exceeds orig_ip_bytes (or resp_bytes exceeds "
               "resp_ip_bytes) is arithmetically impossible: Zeek derives the first from "
               "TCP sequence numbers, which wrap on scan and RST traffic, and counts the "
               "second off the wire, which cannot. These are the same columns that carry "
               "the largest label correlation in 2017 (point-biserial -0.442) and the "
               "largest attack-class mutual information (0.835), so the corruption sits "
               "exactly where the label signal is and every expectation or model built on "
               "it inherits the wrap. Use orig_ip_bytes and resp_ip_bytes instead.",
        metrics={"flows": flows_total, "impossible_flows": impossible_total,
                 "impossible_rate": round(impossible_total / flows_total, 9)
                 if flows_total else None,
                 "classes_affected": len(evidence),
                 "worst_class_rate": round(worst_rate, 9),
                 "material_above_rate": crit,
                 "classes_material": material},
        evidence=evidence,
        repro=q.strip(),
    )]


@check(id="int.capture_spans_disjoint", tier=0,
       title="captures do not overlap in wall-clock time",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def capture_spans_disjoint(ctx):
    """Two captures covering the same second cannot both be addressed by a schedule.

    The labelling predicate in src/preprocess/label/label_flows.py is time plus
    endpoint addresses -- there is no capture term. A window therefore selects a
    wall-clock interval across the whole dataset at once, so the spans must be
    disjoint or the schedule is asking a question the data cannot answer: the same
    second exists in two captures and the rule addresses both.

    CSE-CIC-IDS-2018 fails this. `Friday-02-03-2018` spans 3.51 days;
    `Thursday-01-03-2018` and `Friday-23-02-2018` span 2.47 days each; every other
    capture spans the ~0.5 days its name implies. Five capture pairs overlap, the
    worst by 59.36 hours, and three captures share a ts_min of 1519733906.348 to
    the millisecond -- which is not a long capture, it is one stretch of traffic
    present in three folders.

    The second consequence is that `capture` cannot be used as a train/test split
    key, because the overlapping traffic would sit on both sides of the split.
    """
    if "capture" not in ctx.columns or "ts" not in ctx.columns:
        return []
    q = f"""
    SELECT capture, count(*) AS n, min(ts) AS ts_min, max(ts) AS ts_max
    FROM read_parquet('{ctx.q('labelled')}', union_by_name = true)
    WHERE ts IS NOT NULL
    GROUP BY capture ORDER BY 3"""
    caps = [{"capture": c, "flows": int(n), "ts_min": lo, "ts_max": hi,
             "span_hours": round((hi - lo) / 3600, 2)}
            for c, n, lo, hi in ctx.lake.sql(q) if lo is not None and hi is not None]
    if not caps:
        return []

    max_span = float(ctx.threshold("capture_max_span_hours", 24))
    findings = []

    # Rounded to the millisecond, which is the precision the collision was measured
    # at: two captures starting at the same second by coincidence is conceivable,
    # at the same millisecond it is the same packet seen twice.
    starts = {}
    for c in caps:
        starts.setdefault(round(c["ts_min"], 3), []).append(c)
    shared = [(t, group) for t, group in starts.items() if len(group) > 1]
    if shared:
        findings.append(Finding(
            check_id="int.capture_spans_disjoint", dataset=ctx.dataset,
            severity=Severity.CRITICAL,
            title=f"{sum(len(g) for _, g in shared)} capture(s) begin at an identical "
                  f"timestamp",
            detail="Captures sharing a first timestamp to the millisecond did not begin at "
                   "the same moment by chance -- they contain the same traffic. The flows "
                   "are counted more than once in every class total, and because the "
                   "labelling predicate has no capture term, one schedule window labels "
                   "all the copies. This is the strongest evidence that the capture "
                   "folders are not the independent recordings their names claim.",
            metrics={"colliding_ts_min_values": len(shared),
                     "captures_involved": sum(len(g) for _, g in shared)},
            evidence=[{"ts_min": t, "captures": [c["capture"] for c in g],
                       "flows": [c["flows"] for c in g],
                       "span_hours": [c["span_hours"] for c in g]}
                      for t, g in sorted(shared)],
            repro=q.strip(),
        ))

    pairs = []
    for i, a in enumerate(caps):
        for b in caps[i + 1:]:
            overlap = min(a["ts_max"], b["ts_max"]) - max(a["ts_min"], b["ts_min"])
            if overlap <= 0:
                continue
            pairs.append({"capture_a": a["capture"], "capture_b": b["capture"],
                          "overlap_hours": round(overlap / 3600, 2),
                          "a_span_hours": a["span_hours"], "b_span_hours": b["span_hours"],
                          "a_flows": a["flows"], "b_flows": b["flows"],
                          "identical_start": round(a["ts_min"], 3) == round(b["ts_min"], 3)})
    if pairs:
        worst = max(p["overlap_hours"] for p in pairs)
        findings.append(Finding(
            check_id="int.capture_spans_disjoint", dataset=ctx.dataset,
            severity=Severity.CRITICAL,
            title=f"{len(pairs)} capture pair(s) overlap in wall-clock time, "
                  f"the worst by {worst:,.2f} hours",
            detail="Labelling matches on time and endpoints only, so a window selects a "
                   "wall-clock interval across every capture at once. Where two captures "
                   "cover the same second, a rule written for one of them silently claims "
                   "traffic from the other, and no capture-level correction can separate "
                   "them afterwards. It also rules out `capture` as a split key: the "
                   "overlapping traffic would appear in both train and test.",
            metrics={"captures": len(caps), "overlapping_pairs": len(pairs),
                     "worst_overlap_hours": worst},
            evidence=sorted(pairs, key=lambda p: -p["overlap_hours"]),
            repro=q.strip(),
        ))

    sprawl = [c for c in caps if c["span_hours"] > max_span]
    if sprawl:
        findings.append(Finding(
            check_id="int.capture_spans_disjoint", dataset=ctx.dataset,
            severity=Severity.MAJOR,
            title=f"{len(sprawl)} capture(s) span longer than {max_span:g} hours",
            detail="A capture named for a single day should span well under a day. A "
                   "longer span means two recordings merged under one folder, a stray "
                   "flow carrying a timestamp from elsewhere, or a conversion that did "
                   "not preserve the clock -- and a span that wide is what makes captures "
                   "overlap each other in the first place.",
            metrics={"captures": len(caps), "over_span": len(sprawl),
                     "max_span_hours": max_span,
                     "widest_span_hours": max(c["span_hours"] for c in sprawl)},
            evidence=sorted(sprawl, key=lambda c: -c["span_hours"]),
            repro=q.strip(),
        ))
    return findings


# Features whose nulls the EDA spec coalesces to zero before it measures anything
# (src/eda/spec.py, Feature.trans). For these a null and a real zero are the same
# number by the time any statistic sees them, which is what makes them dangerous.
COALESCED_TO_ZERO = ("duration", "orig_bytes", "resp_bytes")

NULL_AUDIT_COLUMNS = ("duration", "orig_bytes", "resp_bytes", "service", "conn_state")


@check(id="int.null_not_zero", tier=0,
       title="null features are reported separately from zero ones",
       needs=("labelled",), max_severity=Severity.MAJOR)
def null_not_zero(ctx):
    """"Zeek could not measure it" and "it measured zero" are different facts.

    In CIC-DDoS-2019, 9,454,727 `ddos` flows (13.16%) and 34,453 `benign` flows
    carry a null `duration`, and `orig_bytes` / `resp_bytes` behave the same way.
    The EDA spec coalesces those nulls to 0 before computing anything, so the
    entire "duration == 0" mass in that dataset is really "Zeek could not measure
    it" -- an instantaneous flow and an unmeasurable one land on the same number,
    in the very features used to argue a class is flood-shaped.

    This is a property of the data, not a fault in the labelling: nothing here
    says a label is wrong. What it says is which features a downstream claim may
    rest on. A behavioural expectation reading a zero in an affected class cannot
    tell the two cases apart, so it must either exclude the nulls explicitly or
    not use that feature for that class.
    """
    cols = [c for c in NULL_AUDIT_COLUMNS if c in ctx.columns]
    if not cols or "label_class" not in ctx.columns:
        return []
    sel = ", ".join(f'sum(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)' for c in cols)
    q = f"""
    SELECT label_class, count(*) AS n, {sel}
    FROM read_parquet('{ctx.q('labelled')}', union_by_name = true)
    GROUP BY label_class ORDER BY 2 DESC"""
    rows = ctx.lake.sql(q)

    notable = float(ctx.threshold("null_rate_notable", 0.05))
    findings = []
    for i, col in enumerate(cols):
        per_class, nulls_total, flows_total = [], 0, 0
        for row in rows:
            cls, n, nulls = row[0], int(row[1]), int(row[2 + i] or 0)
            flows_total += n
            nulls_total += nulls
            if not nulls:
                continue
            per_class.append({"class": cls, "flows": n, "nulls": nulls,
                              "null_rate": round(nulls / n, 9) if n else None,
                              "share_of_all_nulls": None})
        over = [c for c in per_class if (c["null_rate"] or 0) > notable]
        if not over:
            continue
        for c in per_class:
            c["share_of_all_nulls"] = round(c["nulls"] / nulls_total, 9) if nulls_total else None
        worst = max(over, key=lambda c: c["null_rate"])
        coalesced = col in COALESCED_TO_ZERO
        findings.append(Finding(
            check_id="int.null_not_zero", dataset=ctx.dataset,
            # A null in a coalesced numeric feature becomes an indistinguishable
            # zero; a null in a categorical one becomes its own visible level, so
            # it constrains less and is graded lower.
            severity=Severity.MAJOR if coalesced else Severity.MINOR,
            title=f"{col} is null for {worst['null_rate']:.2%} of {worst['class']} flows"
                  + (f" and {len(over) - 1} other class(es)" if len(over) > 1 else ""),
            detail=(f"{nulls_total:,} flow(s) carry no {col}. "
                    + ("The EDA spec coalesces this feature's nulls to 0, so those flows "
                       "are indistinguishable from flows that genuinely measured zero, and "
                       "a class whose zero mass is really missing data will look "
                       "flood-shaped whether or not it is. "
                       if coalesced else
                       "Nulls in this feature stay visible as their own level rather than "
                       "collapsing onto a real value, but they are still absent evidence "
                       "rather than a measurement. ")
                    + "Nothing here says the labels are wrong -- it constrains which "
                      "features a downstream claim about the affected classes may rest "
                      "on, and requires that nulls be excluded explicitly rather than "
                      "read as zeros."),
            metrics={"feature": col, "flows": flows_total, "nulls": nulls_total,
                     "null_rate": round(nulls_total / flows_total, 9) if flows_total else None,
                     "coalesced_to_zero": coalesced,
                     "classes_over_threshold": len(over),
                     "worst_class": worst["class"],
                     "worst_class_null_rate": worst["null_rate"],
                     "null_rate_notable": notable},
            scope={"feature": col},
            evidence=sorted(per_class, key=lambda c: -(c["null_rate"] or 0)),
            repro=f'SELECT label_class, count(*), '
                  f'sum(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) '
                  f"FROM read_parquet('{ctx.q('labelled')}') GROUP BY 1",
        ))
    return findings


def _capture_glob(conn_glob, capture):
    """The conn zone narrowed to one capture directory.

    Labelling recovers `capture` by stripping the zone's base prefix off each
    parquet's path and taking the first path component, so the same split of the
    glob puts it back. Narrowing is what makes the re-labelling below affordable:
    CIC-DDoS-2019's two captures are ~36M flows each, and scanning both to check
    one of them would cost more than the rest of tier 0 put together.
    """
    if "**" not in conn_glob:
        return None
    base, tail = conn_glob.split("**", 1)
    return f"{base}{capture}/**{tail}"


@check(id="int.relabel_determinism", tier=0,
       title="re-running the labelling logic reproduces the labels in the lake",
       needs=("labelled", "conn"), max_severity=Severity.CRITICAL)
def relabel_determinism(ctx):
    """Labelling is supposed to be a pure function of (flows, manifest, taxonomy).

    Nothing else in Talos is allowed to influence a label: not the order rows
    arrive in, not which machine ran the stage, not when it ran. That is what
    lets every downstream number be quoted against a manifest sha instead of
    against a particular afternoon's run. This check is the only thing that tests
    the claim -- it takes one capture, applies the current rules to the `conn`
    zone with the same predicate that produced the labels, and compares the
    resulting per-class counts against what the labelled zone actually holds.

    A mismatch means the stage returned a different answer from the same inputs.
    That is worse than a wrong label, because a wrong label is at least stable
    enough to be found and corrected: an unstable one makes every figure derived
    from this zone unreproducible, including the ones in this report. No amount
    of downstream care survives it, and nothing else here would notice --
    int.row_parity would still balance, int.label_wellformed would still find
    every column agreeing with every other, and the class counts would still look
    entirely reasonable. They would just not be the counts anyone else gets.

    Reused deliberately, not re-implemented: `MATCH` and `rules_sql` are imported
    from the labelling stage, so the SQL here is byte-identical to the SQL that
    wrote the data. That is the point. A second, independently-written matcher
    would be testing whether two queries agree, which is a different and much
    weaker question -- it would fail whenever the copy drifted and pass whenever
    both were wrong in the same way. What this asks instead is whether the ONE
    matcher we have, run again over the same flows, still lands in the same
    place. So it is a reproducibility check, not a correctness check; sch.* is
    where correctness lives.

    What it can actually catch, given that: a labelled zone built from a source
    other than the conn zone it claims, source parquet rewritten since labelling,
    a `capture` mapping that has moved under the data, a join that fans out
    non-deterministically, and any tie-break between overlapping rules that is
    not the documented min(rule_id).

    Scoped to the SMALLEST capture on purpose. Determinism is a property of the
    stage rather than of any one day's traffic, so one capture tests it as well
    as twelve, and the cheapest capture keeps a tier-0 check cheap enough to run
    on every pass.
    """
    conn_glob = ctx.q("conn")
    needed = ("capture", "label_class", "manifest_sha", "taxonomy_sha")
    if not conn_glob or any(c not in ctx.columns for c in needed):
        return []

    # One pass over four columns: how big each capture is, and which manifest
    # wrote it. Asking both here means the provenance guard below costs nothing.
    caps = [c for c in ctx.lake.sql(f"""
        SELECT capture, count(*) AS n,
               min(manifest_sha) AS man_lo, max(manifest_sha) AS man_hi,
               min(taxonomy_sha) AS tax_lo, max(taxonomy_sha) AS tax_hi
        FROM read_parquet('{ctx.q('labelled')}', union_by_name = true)
        GROUP BY capture ORDER BY n""") if c[1]]
    if not caps:
        return []
    capture, n_actual, man_lo, man_hi, tax_lo, tax_hi = caps[0]

    # If the labels were written by a manifest that is no longer the one on disk,
    # re-labelling them with today's rules WILL differ -- and would differ for a
    # reason int.provenance_current has already reported as a BLOCKER. Raising a
    # CRITICAL for the same cause would be a second alarm for one fault, and the
    # louder one would be the wrong diagnosis: the labels are stale, not
    # non-deterministic. Say what could not be assessed instead.
    if (man_lo != ctx.manifest_sha or man_hi != ctx.manifest_sha
            or tax_lo != ctx.taxonomy_sha or tax_hi != ctx.taxonomy_sha):
        return [Finding(
            check_id="int.relabel_determinism", dataset=ctx.dataset,
            severity=Severity.INFO,
            title="determinism not assessed: the labels were written by a different "
                  "manifest than the one on disk",
            detail="This check re-applies the CURRENT rules and compares the result "
                   "against the stored labels, which is only a test of determinism when "
                   "both sides derive from the same manifest and taxonomy. They do not "
                   "here, so any difference would be the edit, not the stage. Re-run "
                   "labelling and this check becomes meaningful again; until then see "
                   "int.provenance_current, which is the finding that matters.",
            metrics={"capture": capture, "capture_flows": n_actual,
                     "labelled_manifest_sha": man_lo if man_lo == man_hi else "(mixed)",
                     "labelled_taxonomy_sha": tax_lo if tax_lo == tax_hi else "(mixed)",
                     "manifest_sha_on_disk": ctx.manifest_sha,
                     "taxonomy_sha_on_disk": ctx.taxonomy_sha},
            scope={"capture": capture},
        )]

    narrow = _capture_glob(conn_glob, capture)
    if narrow and not (ctx.lake.one(f"SELECT count(*) FROM glob('{narrow}')") or [0])[0]:
        narrow = None
    if narrow:
        src = f"SELECT * FROM read_parquet('{narrow}', union_by_name = true)"
    else:
        # No directory of that name under the conn zone, so fall back to the
        # derivation labelling itself uses. Costs a full scan, but a wrong answer
        # obtained cheaply is not the trade this module makes.
        base = conn_glob.split("**", 1)[0]
        src = (f"SELECT * EXCLUDE (filename) FROM read_parquet('{conn_glob}', "
               f"filename = true, union_by_name = true) "
               f"WHERE split_part(replace(filename, '{base}', ''), '/', 1) = '{capture}'")

    # The labelling stage's own shape: one winning rule per uid by min(rid), then
    # a LEFT JOIN back onto conn so every flow keeps its row. Anything simpler --
    # joining rules straight onto conn -- would fan a flow out across every rule
    # that claims it and inflate the classes that overlap.
    q = f"""
    WITH rules AS ({rules_sql(ctx.rules)}),
    conn AS ({src}),
    hits AS (
      SELECT c.uid, min(r.rid) AS rid
      FROM conn c JOIN rules r ON {MATCH}
      GROUP BY c.uid
    ),
    matched AS (
      SELECT h.uid, r.rclass FROM hits h JOIN rules r ON r.rid = h.rid
    )
    SELECT coalesce(m.rclass, 'benign') AS label_class, count(*) AS n
    FROM conn c LEFT JOIN matched m USING (uid)
    GROUP BY 1"""
    expected = {cls: int(n) for cls, n in ctx.lake.sql(q)}
    exp_total = sum(expected.values())

    # Zero source flows under this capture is not a determinism result. It means
    # the labelled zone names a capture the conn zone does not contain, which is
    # a provenance question about where this data came from, and calling it
    # non-determinism would put a CRITICAL on the wrong cause.
    if not exp_total:
        return [Finding(
            check_id="int.relabel_determinism", dataset=ctx.dataset,
            severity=Severity.INFO,
            title=f"determinism not assessed: capture {capture!r} holds "
                  f"{n_actual:,} labelled flow(s) but no source flows",
            detail="The labelled zone attributes flows to a capture that the conn zone "
                   "has none of, so there is nothing to re-label and nothing to compare "
                   "against. Either the labelled zone was built from a source other than "
                   "this lake's conn zone, or the capture directories have been renamed "
                   "since. Both are provenance faults rather than determinism faults.",
            metrics={"capture": capture, "labelled_flows": n_actual,
                     "source_flows": 0, "conn_glob": narrow or conn_glob},
            scope={"capture": capture}, repro=q.strip(),
        )]

    aq = f"""
    SELECT label_class, count(*) AS n
    FROM read_parquet('{ctx.q('labelled')}', union_by_name = true)
    WHERE capture = '{capture}' GROUP BY 1"""
    actual = {cls: int(n) for cls, n in ctx.lake.sql(aq)}
    act_total = sum(actual.values())

    rows = []
    for cls in sorted(set(expected) | set(actual)):
        e, g = expected.get(cls, 0), actual.get(cls, 0)
        rows.append({"class": cls, "expected": e, "actual": g, "delta": g - e,
                     "delta_rate": round((g - e) / e, 9) if e else None})
    off = [r for r in rows if r["delta"]]
    if not off:
        return []

    moved = sum(abs(r["delta"]) for r in off)
    row_gap = act_total - exp_total
    worst = max(off, key=lambda r: abs(r["delta"]))
    return [Finding(
        check_id="int.relabel_determinism", dataset=ctx.dataset,
        severity=Severity.CRITICAL,
        title=f"{capture}: re-labelling disagrees with the stored labels on "
              f"{len(off)} class(es), {moved:,} flow(s)",
        detail="The current rules were applied to this capture's source flows with the "
               "same predicate that wrote the labelled zone, and the per-class counts "
               "came out different. Labelling is meant to be a pure function of its "
               "source flows, its manifest and its taxonomy -- given the same three it "
               "must return the same labels -- so a difference means the stage does not "
               "reproduce, and every class count, histogram and accuracy figure derived "
               "from this zone describes one particular run rather than the dataset."
               + (f" The two sides also disagree on row count by {abs(row_gap):,} "
                  f"({'labelled has more' if row_gap > 0 else 'source has more'}), so "
                  f"this is flows appearing or disappearing rather than flows moving "
                  f"between classes; int.row_parity is the finding to read first."
                  if row_gap else
                  " Row counts agree exactly, so the same flows are being sorted into "
                  "different classes -- the matching itself is unstable, not the input."),
        metrics={"capture": capture,
                 "captures_in_zone": len(caps),
                 "chosen_as": "smallest by flow count",
                 "source_flows": exp_total, "labelled_flows": act_total,
                 "row_delta": row_gap,
                 "classes_compared": len(rows), "classes_disagreeing": len(off),
                 "flows_disagreeing": moved,
                 "disagreement_rate": round(moved / exp_total, 9),
                 "worst_class": worst["class"], "worst_class_delta": worst["delta"],
                 "manifest_sha": ctx.manifest_sha, "taxonomy_sha": ctx.taxonomy_sha},
        scope={"capture": capture},
        evidence=sorted(rows, key=lambda r: -abs(r["delta"])),
        repro=f"-- expected, recomputed from the conn zone:\n{q.strip()}\n\n"
              f"-- actual, as stored:\n{aq.strip()}",
    )]
