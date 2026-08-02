#!/usr/bin/env python3
"""Attach ground-truth labels to Zeek conn flows from the attack manifests.

Labels never come from the traffic -- they come from the published attack
schedules, transcribed into src/preprocess/manifests/<dataset>.yaml. This
applies them: a flow is an attack if its start time falls inside an attack
window AND its endpoints match that attack's attacker/victim pair. Everything
else inside a scheduled capture is benign by the closed-world assumption.

Per-dataset wrinkles the manifests encode, all handled here:
  * local wall-clock times + a UTC offset (2019 carries a DIFFERENT offset per
    day -- Nov 3 is ADT, Dec 1 is AST)
  * NAT aliases: one host appears under several IPs depending on capture point
    (2017's Kali behind the firewall, 2018's dual VPC/public addresses)
  * victim-only matching for reflection attacks with spoofed sources (2019)
  * subnet victims (2017's infiltrated host scanning its own LAN)

Raw attack names are mapped to canonical classes via manifests/taxonomy.yaml
so classes mean the same thing across datasets -- without that, cross-domain
evaluation is meaningless.

Some flows cannot be adjudicated from the schedule at all: the gaps between
2019's reflection floods, where the attack plainly has not stopped but no
window says which vector it is; 2018's infiltration windows, where CIC names
the infected host but never the second stage's targets. Calling those benign
under the closed-world assumption is a guess wearing a label. The manifests
therefore carry an optional `quarantine:` list -- declared time+endpoint
regions, each with a written justification -- and every flow inside one is
marked `label_quality = 'uncertain'` with the region's `quarantine_reason`.
It keeps its label_binary/label_class, so nothing downstream breaks and the
closed-world reading stays visible; what changes is that downstream pools can
exclude it and report coverage alongside accuracy. Quarantine is DECLARED, like
the attack windows and for the same reason: labels here are never inferred from
the traffic.

--out writes EVERY flow, benign and attack alike, so the result is a complete
labelled dataset rather than an attack list downstream has to reconstruct from.
Each row carries its provenance: originating capture, the rule that fired, and
content hashes of the manifest and taxonomy that produced it.

Usage:
    python src/preprocess/label/label_flows.py --dataset cic-ids-2017
    python src/preprocess/label/label_flows.py --dataset cic-ids-2018 \
        --out s3://bkt/labels/cic-ids-2018.parquet
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

# label_flows.py sits one level deeper than the other stage scripts
# (src/preprocess/label/ rather than src/preprocess/), so the repo root is
# parents[3], not parents[2]. Getting this wrong pointed MANIFESTS at a
# directory that has never existed and left the stage unrunnable.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.common.lake import Lake  # noqa: E402  -- needs ROOT on sys.path first

# TALOS_MANIFESTS lets the validation fixtures drive this stage against a
# synthetic schedule without putting test manifests next to the real ones.
MANIFESTS = Path(os.environ.get("TALOS_MANIFESTS")
                 or ROOT / "src" / "preprocess" / "manifests")
REPORTS = ROOT / "reports"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="dataset name (must have a manifest)")
    p.add_argument("--source", default=None,
                   help="conn parquet glob (default: the dataset's conn files in the lake)")
    p.add_argument("--out", default=None,
                   help="override the output path (default: the lake's labelled zone)")
    p.add_argument("--no-write", action="store_true",
                   help="report only; do not write the labelled dataset")
    p.add_argument("--report", default=None,
                   help="machine-readable run report (default: reports/labelling_<dataset>.json)")
    p.add_argument("--config", default=str(ROOT / "config.yml"))
    # Same guard rails the EDA stage carries: labelling 2019 scans 72.5M flows and
    # spills exactly like profiling does, and this box has frozen on that twice.
    p.add_argument("--memory-limit", default="8GB")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--temp-dir", default=None)
    p.add_argument("--max-temp", default="20GB")
    return p.parse_args()


# ----------------------------------------------------------------- config

def load_config(path):
    """Bucket/region/zone prefixes come from config.yml, not constants here."""
    cfg = yaml.safe_load(Path(path).read_text())
    return (cfg["aws"]["bucket"], cfg["aws"]["region"],
            cfg["lake"]["parquets"], cfg["lake"].get("labelled", "labelled"))


def sha(path):
    """Content hash -- more honest provenance than a hand-bumped version field."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------- manifest -> rules

def epoch(date, hhmm, offset):
    """Local wall-clock (date, HH:MM) at a UTC offset -> epoch seconds."""
    sign = -1 if offset.startswith("-") else 1
    h, m = offset.lstrip("+-").split(":")
    tz = timezone(sign * timedelta(hours=int(h), minutes=int(m)))
    return datetime.strptime(f"{date} {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=tz).timestamp()


def load_rules(dataset):
    """Normalise a manifest into flat rules: one per attack window."""
    man = yaml.safe_load((MANIFESTS / f"{dataset}.yaml").read_text())
    taxdoc = yaml.safe_load((MANIFESTS / "taxonomy.yaml").read_text())
    tax = taxdoc["raw_to_canonical"]
    needs_payload = set(taxdoc.get("requires_payload", []))
    days = man.get("days")  # 2019: per-day date + offset; others: dataset-level

    rules = []
    for i, a in enumerate(man["attacks"]):
        if days:
            d = days[a["day"]]
            date, off = str(d["date"]), d["utc_offset"]
        else:
            date, off = str(a["date"]), man["utc_offset"]
        if a["name"] not in tax:
            sys.exit(f"attack '{a['name']}' missing from taxonomy.yaml raw_to_canonical")
        sub = a.get("victim_subnet")
        rules.append({
            "rid": i,
            "name": a["name"],
            "canonical": tax[a["name"]],
            "t0": epoch(date, a["start"], off),
            "t1": epoch(date, a["end"], off),   # extended below
            "attackers": [str(x) for x in a.get("attacker_ips", [])] or None,
            "victims": [str(x) for x in a.get("victim_ips", [])] or None,
            # a /24 victim subnet becomes a cheap string prefix test
            "prefix": sub.rsplit(".", 1)[0] + "." if sub else None,
            "needs_payload": a["name"] in needs_payload,
            "date": date, "start": a["start"], "end": a["end"],
        })

    # Schedules are minute-granular: "13:29 - 13:34" means the attack ran THROUGH
    # 13:34:59, not until 13:34:00, so each window is extended to the end of its
    # final minute (MATCH uses ts < t1). Without this every rule silently drops
    # its last minute -- on flood datasets, hundreds of thousands of flows each.
    # The extension is clamped at the next window's start so back-to-back
    # attacks (2019's WebDDoS 13:18-13:29 then SYN 13:29-13:34) never overlap.
    starts = sorted(r["t0"] for r in rules)
    for r in rules:
        nxt = next((s for s in starts if s >= r["t1"]), None)
        # t1_raw keeps the schedule's own end time so the extension stays
        # auditable: validation needs to attribute flows to the extra minute
        # rather than take it on trust.
        r["t1_raw"] = r["t1"]
        r["t1"] = min(r["t1"] + 60, nxt) if nxt is not None else r["t1"] + 60
        r["extended_by"] = r["t1"] - r["t1_raw"]
    return man, rules


def validate_rules(rules):
    """Fail fast on manifest mistakes that would silently corrupt labels."""
    errs = []
    for r in rules:
        if r["t1"] <= r["t0"]:
            errs.append(f"{r['name']} {r['date']}: end {r['end']} <= start {r['start']}")
        if not (r["attackers"] or r["victims"] or r["prefix"]):
            errs.append(f"{r['name']} {r['date']}: no attacker and no victim — "
                        "would label every flow in the window")
    if errs:
        sys.exit("manifest validation failed:\n  " + "\n  ".join(errs))


# -------------------------------------------------- manifest -> quarantine

def load_quarantine(man, rules):
    """Normalise the manifest's `quarantine:` list into the attack-rule shape.

    A quarantine region is a span of time and a set of endpoints we have
    declined to adjudicate. It exists because the alternative is worse: the
    closed-world assumption turns every flow no window claims into a benign
    one, and in the places where an attack is obviously still running, that
    manufactures a benign class out of attack tail. Declining to answer is a
    defensible position; answering wrongly is not.

    Regions are DECLARED in the manifest, never inferred per flow from the
    traffic, so the rule that labels in Talos come from the published schedule
    survives quarantine intact -- and a reader can open one file and see every
    span we have withdrawn, with the reason we withdrew it.

    They are normalised into exactly the rule shape `rules_sql()` serialises so
    that the endpoint predicate applied to a region is literally the same SQL
    applied to an attack window. A second matcher would drift from the first,
    and the drift would be invisible.
    """
    days = man.get("days")          # 2019: per-day date + offset; others: dataset-level
    regions = []
    for i, z in enumerate(man.get("quarantine") or []):
        if days:
            d = days[z["day"]]
            date, off = str(d["date"]), d["utc_offset"]
        else:
            date, off = str(z["date"]), man["utc_offset"]
        sub = z.get("victim_subnet")
        regions.append({
            # name/canonical carry the reason because rules_sql() -- written for
            # attack rules -- reads those keys. The region then travels through
            # the same VALUES table and the same MATCH as everything else.
            "rid": i, "name": z.get("reason"), "canonical": z.get("reason"),
            "reason": z.get("reason"), "justification": z.get("justification"),
            "t0": epoch(date, z["start"], off),
            "t1": epoch(date, z["end"], off),   # extended below
            "attackers": [str(x) for x in z.get("attacker_ips", [])] or None,
            "victims": [str(x) for x in z.get("victim_ips", [])] or None,
            "prefix": sub.rsplit(".", 1)[0] + "." if sub else None,
            "needs_payload": False,
            "date": date, "start": z["start"], "end": z["end"],
        })

    # Regions are transcribed off the same minute-granular clock as the windows
    # they sit between, so "11:46 - 11:50" means through 11:50:59 here too, and
    # the same end-of-minute extension applies. It is clamped at the next
    # attack's start for the same reason the attack rules clamp: a region must
    # never reach into a window and quarantine flows the schedule CAN account
    # for. On 2019's inter-window gaps the clamp cancels the extension exactly,
    # because each gap ends where the next attack begins -- which is what makes
    # the regions tile the gaps without touching either neighbour.
    starts = sorted(r["t0"] for r in rules)
    for z in regions:
        nxt = next((s for s in starts if s >= z["t1"]), None)
        z["t1_raw"] = z["t1"]
        z["t1"] = min(z["t1"] + 60, nxt) if nxt is not None else z["t1"] + 60
        z["extended_by"] = z["t1"] - z["t1_raw"]
    return regions


# A reason is an identifier -- it is written into every quarantined row and
# grouped on downstream. The sentence goes in `justification`.
REASON_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-_")


def validate_quarantine(regions):
    """Fail fast on a region that would remove more than it declares."""
    errs = []
    for z in regions:
        where = f"{z['reason']} {z['date']} {z['start']}-{z['end']}"
        if not z["reason"] or set(str(z["reason"])) - REASON_CHARS:
            errs.append(f"{where}: reason must be a lower-case identifier "
                        f"([a-z0-9-_]); the prose belongs in justification")
        if not str(z["justification"] or "").strip():
            errs.append(f"{where}: no justification — an undocumented region is a "
                        "silent narrowing of the dataset, which is the thing "
                        "quarantine exists to prevent")
        if z["t1"] <= z["t0"]:
            errs.append(f"{where}: end {z['end']} <= start {z['start']}")
        if not (z["attackers"] or z["victims"] or z["prefix"]):
            errs.append(f"{where}: no attacker, victim or subnet — "
                        "would quarantine every flow in the window")
    if errs:
        sys.exit("quarantine validation failed:\n  " + "\n  ".join(errs))


# ---------------------------------------------------------- query building

def _victim_ok(col):
    """A victim constraint may be an IP list, a subnet prefix, or genuinely absent.

    Absent means "any peer"; it must NOT be inferred just because the IP list is
    empty on a subnet rule, or the rule matches the whole capture.
    """
    return f"""(
        (r.vic IS NULL AND r.vpfx IS NULL)
        OR coalesce(list_contains(r.vic, c."{col}"), false)
        OR coalesce(starts_with(c."{col}", r.vpfx), false)
      )"""


MATCH = f"""
  c.ts >= r.t0 AND c.ts < r.t1
  AND (
    -- attacker known: the flow must run between attacker and victim, either way
    (r.atk IS NOT NULL AND (
        (list_contains(r.atk, c."id.orig_h") AND {_victim_ok('id.resp_h')})
     OR (list_contains(r.atk, c."id.resp_h") AND {_victim_ok('id.orig_h')})
    ))
    -- reflection attacks: sources are spoofed, so match the victim alone
    OR (r.atk IS NULL AND r.vic IS NOT NULL
        AND (list_contains(r.vic, c."id.orig_h") OR list_contains(r.vic, c."id.resp_h")))
  )
"""


def rules_sql(rules):
    """Rules as an inline VALUES table.

    NULLs need explicit types: a column that is NULL for every rule (e.g. the
    attacker list on reflection-only datasets) would otherwise bind as INTEGER.
    """
    def lst(v):
        return "CAST(NULL AS VARCHAR[])" if not v else "[" + ", ".join(f"'{x}'" for x in v) + "]"

    rows = ",\n    ".join(
        f"({r['rid']}, '{r['name']}', '{r['canonical']}', {r['t0']}, {r['t1']}, "
        f"{lst(r['attackers'])}, {lst(r['victims'])}, "
        f"{'CAST(NULL AS VARCHAR)' if not r['prefix'] else repr(r['prefix'])}, "
        f"{'true' if r['needs_payload'] else 'false'})"
        for r in rules)
    return ("SELECT * FROM (VALUES\n    " + rows +
            "\n  ) AS r(rid, rname, rclass, t0, t1, atk, vic, vpfx, needs_payload)")


def _quarantine_ctes(regions):
    """The declared no-adjudication regions, matched by the SAME predicate.

    Emitted only when a manifest declares regions, so a dataset without a
    `quarantine:` block produces byte-identical SQL to the version of this
    stage that predates the feature.

    A flow inside two overlapping regions takes the lowest region id, the same
    deterministic tie-break attack rules use. Overlapping regions are a
    manifest smell rather than an error: whichever reason wins, the flow is out
    of the pool either way.
    """
    if not regions:
        return ""
    return f""",
      qrules AS ({rules_sql(regions)}),
      qhits AS (
        SELECT c.uid, min(r.rid) AS qid
        FROM conn c JOIN qrules r ON {MATCH}
        GROUP BY c.uid
      ),
      quarantined AS (
        SELECT h.uid, h.qid, r.rname AS qreason
        FROM qhits h JOIN qrules r ON r.rid = h.qid
      )"""


def _base_ctes(rules, src, base, regions=()):
    """rules + conn + the matched-attack set (+ the quarantined set, if any).

    `capture` is recovered from each parquet's path -- the 1:1 mirror keeps the
    originating pcap folder in the tree, so provenance survives into the labels.

    Picking the winning rule per flow is a GROUP BY, not a window function:
    min(rid) gives the same deterministic choice as row_number()=1 and count(*)
    the same overlap tally, but as a hash aggregate rather than a sort over
    every matched row. On flood datasets that is the difference between a hash
    table and spilling tens of GB (2019 matches ~70M of its 72.5M flows).
    """
    return f"""
      WITH rules AS ({rules_sql(rules)}),
      conn AS (
        SELECT * EXCLUDE (filename),
               split_part(replace(filename, '{base}', ''), '/', 1) AS capture
        FROM read_parquet('{src}', filename = true, union_by_name = true)
      ),
      hits AS (
        SELECT c.uid, min(r.rid) AS rid, count(*) AS nrules
        FROM conn c JOIN rules r ON {MATCH}
        GROUP BY c.uid
      ),
      matched AS (
        SELECT h.uid, h.rid, h.nrules, r.rname, r.rclass, r.needs_payload
        FROM hits h JOIN rules r ON r.rid = h.rid
      )""" + _quarantine_ctes(regions)


# Did the attack actually execute? Only meaningful where the attack requires the
# attacker to send payload (see taxonomy.yaml requires_payload). NULL = not
# applicable, or Zeek could not determine the byte count -- which is NOT the same
# as a confirmed empty connection and must not be counted as one.
EXECUTED = """CASE WHEN m.rid IS NULL OR NOT m.needs_payload THEN NULL
                   WHEN c.orig_bytes IS NULL THEN NULL
                   WHEN c.orig_bytes > 0 THEN true
                   ELSE false END"""


def attack_query(rules, src, base):
    """Attack flows only -- all the report needs.

    Benign is total minus attack, and the total comes from parquet footers, so
    the report never pays for a LEFT JOIN across every flow in the dataset.
    """
    return _base_ctes(rules, src, base) + f"""
      SELECT m.rclass, m.rname, m.rid, count(*) AS n,
             sum(CASE WHEN m.nrules > 1 THEN 1 ELSE 0 END) AS overlaps,
             sum(CASE WHEN {EXECUTED} THEN 1 ELSE 0 END) AS executed,
             sum(CASE WHEN {EXECUTED} = false THEN 1 ELSE 0 END) AS attempted
      FROM matched m JOIN conn c USING (uid)
      GROUP BY 1, 2, 3 ORDER BY 4 DESC"""


def benign_qa_query(rules, src, base):
    """Automated benign-class audit.

    Benign traffic does not arrive in large synchronised bursts aimed at one
    host. Where it does, the labelling has missed an attack -- this is exactly
    how the 2019 DNS misalignment and the off-by-one-minute bug were caught, so
    the check runs on every labelling run rather than by hand.
    """
    ips = sorted({ip for r in rules for ip in (r["victims"] or [])})
    inlist = ", ".join(f"'{i}'" for i in ips) or "''"
    return _base_ctes(rules, src, base) + f"""
      SELECT c.capture,
             strftime(to_timestamp(floor(c.ts/600)*600) AT TIME ZONE 'UTC',
                      '%Y-%m-%d %H:%M') AS bucket,
             count(*) AS n,
             sum(CASE WHEN c."id.orig_h" IN ({inlist})
                        OR c."id.resp_h" IN ({inlist}) THEN 1 ELSE 0 END) AS victim_flows
      FROM conn c LEFT JOIN matched m USING (uid)
      WHERE m.rid IS NULL
      GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10"""


def quarantine_query(rules, src, base, regions):
    """Per-region quarantine tally, for the run report.

    Grouped by region rather than by reason so a declared region that matches
    nothing is visible at labelling time, exactly as a silent attack rule is.
    Costs one extra scan and is therefore run only when regions are declared.
    """
    return _base_ctes(rules, src, base, regions) + """
      SELECT qn.qid, any_value(qn.qreason) AS qreason, count(*) AS n,
             sum(CASE WHEN m.rid IS NOT NULL THEN 1 ELSE 0 END) AS attack_labelled
      FROM conn c JOIN quarantined qn USING (uid)
      LEFT JOIN matched m USING (uid)
      GROUP BY 1 ORDER BY 3 DESC"""


def labelled_query(rules, src, base, dataset, man_sha, tax_sha, started, regions=()):
    """The persisted dataset: every conn flow, every original column, plus labels.

    Downstream reads one table -- it never has to join labels back onto flows or
    reconstruct benign by anti-join.

    A quarantined flow keeps the label the schedule gave it and is flagged
    beside it, rather than being given a third label_class value: every
    invariant tying label_binary, label_class, rule_id, label_raw and
    label_source together still holds, downstream code that has never heard of
    quarantine still reads the closed-world label, and code that has can filter
    on one column.
    """
    quality = ("""CASE WHEN qn.qreason IS NULL THEN 'certain'
                       ELSE 'uncertain' END               AS label_quality,
             qn.qreason                                   AS quarantine_reason"""
               if regions else
               """'certain'                                    AS label_quality,
             CAST(NULL AS VARCHAR)                        AS quarantine_reason""")
    join = "\n      LEFT JOIN quarantined qn USING (uid)" if regions else ""
    return _base_ctes(rules, src, base, regions) + f"""
      SELECT c.*,
             m.rid IS NOT NULL                          AS label_binary,
             coalesce(m.rclass, 'benign')               AS label_class,
             m.rname                                    AS label_raw,
             m.rid                                      AS rule_id,
             {EXECUTED}                                 AS label_executed,
             CASE WHEN m.rid IS NOT NULL THEN 'schedule'
                  ELSE 'schedule-closed-world' END      AS label_source,
             coalesce(m.nrules, 0)                      AS rules_matched,
             {quality},
             '{dataset}' AS dataset, '{man_sha}' AS manifest_sha,
             '{tax_sha}' AS taxonomy_sha, '{started}' AS labelled_at
      FROM conn c LEFT JOIN matched m USING (uid){join}"""


# ---------------------------------------------------------------- reporting

def write_report(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main():
    a = parse_args()
    bucket, region, parq, lbl = load_config(a.config)
    man, rules = load_rules(a.dataset)
    validate_rules(rules)
    regions = load_quarantine(man, rules)
    validate_quarantine(regions)

    # `capture` is recovered by stripping `base` off each parquet's path, so base
    # has to describe the tree we are actually reading -- not the lake's, when
    # --source points somewhere else.
    src = a.source or f"s3://{bucket}/{parq}/{a.dataset}/**/conn.parquet"
    if not a.source:
        base = f"s3://{bucket}/{parq}/{a.dataset}/"
    elif "**" in a.source:
        base = a.source.split("**")[0]
    else:
        # A source naming files directly still needs a base to strip, or every
        # capture comes out as the empty string.
        base = a.source.rsplit("/", 1)[0] + "/"
    # labelling writes by default -- the point of the stage is labelled data
    out = None if a.no_write else (a.out or f"s3://{bucket}/{lbl}/{a.dataset}/conn.parquet")
    lake = Lake(region, a.memory_limit, a.temp_dir, a.threads, a.max_temp,
                require_s3=src.startswith("s3://"))
    man_sha = sha(MANIFESTS / f"{a.dataset}.yaml")
    tax_sha = sha(MANIFESTS / "taxonomy.yaml")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"dataset   {a.dataset}")
    print(f"source    {src}")
    print(f"output    {out or '(report only — --no-write)'}")
    print(f"manifest  {man_sha}   taxonomy {tax_sha}")
    print(f"rules     {len(rules)} windows -> "
          f"{len(set(r['canonical'] for r in rules))} canonical classes")
    print(f"quarantine {len(regions)} declared region(s)"
          + (f" -> {', '.join(sorted({z['reason'] for z in regions}))}" if regions else "")
          + "\n")

    total = lake.sql(f"SELECT count(*) FROM read_parquet('{src}')")[0][0]
    rows = lake.sql(attack_query(rules, src, base))

    attack = sum(r[3] for r in rows)
    overlaps = sum(r[4] for r in rows)

    print(f"{'canonical class':<15}{'raw attack name':<25}{'flows':>13}"
          f"{'executed':>11}{'attempted':>11}")
    for cls, raw, _, n, _, ex, att in rows:
        note = "  <-- mostly attempts" if att and att > n / 2 else ""
        print(f"{cls:<15}{raw:<25}{n:>13,}"
              f"{(f'{ex:,}' if ex else '-'):>11}{(f'{att:,}' if att else '-'):>11}{note}")
    print(f"{'-' * 56}\n{'ATTACK':<42}{attack:>14,}")
    print(f"{'BENIGN (closed-world)':<42}{total - attack:>14,}")
    print(f"{'TOTAL':<42}{total:>14,}")
    if total:
        print(f"\nattack ratio {100 * attack / total:.3f}%")

    fired = {r[2] for r in rows}
    silent = [r for r in rules if r["rid"] not in fired]
    if silent:
        print(f"\n!! {len(silent)} rule(s) matched NOTHING — check times/IPs:")
        for r in silent:
            print(f"     {r['name']:<24} {r['date']} {r['start']}-{r['end']} local")
    else:
        print("\nall rules matched at least one flow")
    if overlaps:
        print(f"note: {overlaps:,} flow(s) matched >1 rule; lowest rule id wins")

    # ---- automated benign-class audit -------------------------------------
    qa = lake.sql(benign_qa_query(rules, src, base))
    suspect = [(cap, b, n, v) for cap, b, n, v in qa
               if n > max(10_000, 0.005 * total) and v > 0.9 * n]
    print("\nbenign-class audit (largest unmatched bursts):")
    if suspect:
        print(f"  !! {len(suspect)} burst(s) look like MISSED ATTACKS — large, and "
              f"almost entirely aimed at a known victim:")
        for cap, b, n, v in suspect[:5]:
            print(f"     {cap:<24}{b}Z  {n:>12,} flows, {100*v/n:.1f}% victim-involving")
        print("     -> a rule's window is probably wrong; check the schedule against "
              "the traffic")
    else:
        print("  no benign bursts concentrated on a victim — labelling looks complete")

    # ---- declared quarantine ----------------------------------------------
    # Coverage is reported here, next to the class counts, because an accuracy
    # figure quoted without it is not a claim about the dataset -- it is a claim
    # about whichever part of the dataset we were willing to adjudicate.
    qrows = lake.sql(quarantine_query(rules, src, base, regions)) if regions else []
    by_region = {int(q[0]): {"reason": q[1], "flows": int(q[2]),
                             "attack_labelled": int(q[3] or 0)} for q in qrows}
    quarantined = sum(v["flows"] for v in by_region.values())
    if regions:
        print("\nquarantine (declared unadjudicable regions):")
        for z in regions:
            got = by_region.get(z["rid"], {"flows": 0, "attack_labelled": 0})
            note = "   <-- MATCHED NOTHING" if not got["flows"] else ""
            print(f"  {z['reason']:<26}{z['date']} {z['start']}-{z['end']} local"
                  f"{got['flows']:>13,} flows{note}")
        silent_z = [z for z in regions if not by_region.get(z["rid"], {}).get("flows")]
        if silent_z:
            print(f"  !! {len(silent_z)} region(s) matched NOTHING — a stale declaration "
                  f"narrows nothing and hides nothing")
        if total:
            print(f"  {quarantined:,} flow(s) quarantined ({100 * quarantined / total:.3f}%) "
                  f"— {100 * (total - quarantined) / total:.3f}% of the dataset is "
                  f"adjudicable")

    report = {
        "dataset": a.dataset, "source": src, "started_utc": started,
        "manifest_sha": man_sha, "taxonomy_sha": tax_sha,
        "rules_total": len(rules), "rules_fired": len(fired), "rules_silent":
            [{"rule_id": r["rid"], "name": r["name"], "date": r["date"],
              "start": r["start"], "end": r["end"]} for r in silent],
        "flows_total": total, "flows_attack": attack, "flows_benign": total - attack,
        "attack_ratio": (attack / total) if total else 0.0,
        "flows_multi_rule": overlaps,
        "flows_executed": sum(r[5] or 0 for r in rows),
        "flows_attempted": sum(r[6] or 0 for r in rows),
        "qa_benign_bursts": [{"capture": q[0], "bucket_utc": q[1], "flows": q[2],
                              "victim_involving": q[3],
                              "victim_pct": round(100*q[3]/q[2], 1) if q[2] else 0}
                             for q in qa],
        "qa_suspect_bursts": len(suspect),
        "quarantine_regions": len(regions),
        "flows_quarantined": quarantined,
        "quarantine_share": (quarantined / total) if total else 0.0,
        # The honest denominator: what fraction of the dataset the labels are
        # actually making a claim about.
        "label_coverage": ((total - quarantined) / total) if total else 0.0,
        "by_quarantine_region": [
            {"region_id": z["rid"], "reason": z["reason"],
             "date": z["date"], "start": z["start"], "end": z["end"],
             "flows": by_region.get(z["rid"], {}).get("flows", 0),
             "attack_labelled": by_region.get(z["rid"], {}).get("attack_labelled", 0),
             "justification": z["justification"]} for z in regions],
        "by_class": dict({c: sum(r[3] for r in rows if r[0] == c) for c in {r[0] for r in rows}},
                         **{"benign": total - attack}),
        "by_rule": [{"rule_id": r[2], "raw": r[1], "class": r[0], "flows": r[3],
                     "executed": r[5], "attempted": r[6]} for r in rows],
        "output": out,
    }
    rpath = a.report or REPORTS / f"labelling_{a.dataset}.json"
    write_report(rpath, report)
    print(f"\nrun report -> {rpath}")

    if out:
        print(f"writing labelled flows -> {out}")
        lq = labelled_query(rules, src, base, a.dataset, man_sha, tax_sha, started,
                            regions)
        lake.sql(f"COPY ({lq}) TO '{out}' (FORMAT PARQUET)")
        print(f"done — {total:,} labelled flows written "
              f"({attack:,} attack, {total - attack:,} benign"
              + (f", {quarantined:,} quarantined" if regions else "") + ")")


if __name__ == "__main__":
    main()
