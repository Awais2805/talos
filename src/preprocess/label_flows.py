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

--out writes EVERY flow, benign and attack alike, so the result is a complete
labelled dataset rather than an attack list downstream has to reconstruct from.
Each row carries its provenance: originating capture, the rule that fired, and
content hashes of the manifest and taxonomy that produced it.

Usage:
    python src/preprocess/label_flows.py --dataset cic-ids-2017
    python src/preprocess/label_flows.py --dataset cic-ids-2018 \
        --out s3://bkt/labels/cic-ids-2018.parquet
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = Path(__file__).parent / "manifests"
REPORTS = ROOT / "reports"
REFRESH_EVERY = 300


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


# ------------------------------------------------------------ lake access

def _cli_creds():
    try:
        out = subprocess.run(["aws", "configure", "export-credentials", "--format", "env"],
                             capture_output=True, text=True).stdout
        return dict(l[7:].split("=", 1) for l in out.splitlines() if l.startswith("export "))
    except Exception:
        return {}


class Lake:
    """DuckDB connection that keeps its S3 secret fresh across long scans."""

    def __init__(self, region):
        self.region, self._minted = region, 0.0
        self.con = duckdb.connect()
        self.con.execute("INSTALL httpfs; LOAD httpfs;")
        if not self.refresh(force=True):
            sys.exit("no AWS credentials from the CLI — run 'aws login' and retry")

    def refresh(self, force=False):
        if not force and time.time() - self._minted < REFRESH_EVERY:
            return True
        cr = _cli_creds()
        akid, sec, tok = (cr.get("AWS_ACCESS_KEY_ID"), cr.get("AWS_SECRET_ACCESS_KEY"),
                          cr.get("AWS_SESSION_TOKEN"))
        if not (akid and sec):
            return False
        extra = f", SESSION_TOKEN '{tok}'" if tok else ""
        self.con.execute(f"CREATE OR REPLACE SECRET lake (TYPE S3, KEY_ID '{akid}', "
                         f"SECRET '{sec}'{extra}, REGION '{self.region}');")
        self._minted = time.time()
        return True

    def sql(self, query):
        self.refresh()
        return self.con.execute(query).fetchall()


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
    tax = yaml.safe_load((MANIFESTS / "taxonomy.yaml").read_text())["raw_to_canonical"]
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
            "t1": epoch(date, a["end"], off),
            "attackers": [str(x) for x in a.get("attacker_ips", [])] or None,
            "victims": [str(x) for x in a.get("victim_ips", [])] or None,
            # a /24 victim subnet becomes a cheap string prefix test
            "prefix": sub.rsplit(".", 1)[0] + "." if sub else None,
            "date": date, "start": a["start"], "end": a["end"],
        })
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
  c.ts >= r.t0 AND c.ts <= r.t1
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
        f"{'CAST(NULL AS VARCHAR)' if not r['prefix'] else repr(r['prefix'])})"
        for r in rules)
    return ("SELECT * FROM (VALUES\n    " + rows +
            "\n  ) AS r(rid, rname, rclass, t0, t1, atk, vic, vpfx)")


def _base_ctes(rules, src, base):
    """rules + conn + the matched-attack set.

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
        SELECT h.uid, h.rid, h.nrules, r.rname, r.rclass
        FROM hits h JOIN rules r ON r.rid = h.rid
      )"""


def attack_query(rules, src, base):
    """Attack flows only -- all the report needs.

    Benign is total minus attack, and the total comes from parquet footers, so
    the report never pays for a LEFT JOIN across every flow in the dataset.
    """
    return _base_ctes(rules, src, base) + """
      SELECT rclass, rname, rid, count(*) AS n,
             sum(CASE WHEN nrules > 1 THEN 1 ELSE 0 END) AS overlaps
      FROM matched GROUP BY 1, 2, 3 ORDER BY 4 DESC"""


def labelled_query(rules, src, base, dataset, man_sha, tax_sha, started):
    """The persisted dataset: every conn flow, every original column, plus labels.

    Downstream reads one table -- it never has to join labels back onto flows or
    reconstruct benign by anti-join.
    """
    return _base_ctes(rules, src, base) + f"""
      SELECT c.*,
             m.rid IS NOT NULL                          AS label_binary,
             coalesce(m.rclass, 'benign')               AS label_class,
             m.rname                                    AS label_raw,
             m.rid                                      AS rule_id,
             CASE WHEN m.rid IS NOT NULL THEN 'schedule'
                  ELSE 'schedule-closed-world' END      AS label_source,
             coalesce(m.nrules, 0)                      AS rules_matched,
             '{dataset}' AS dataset, '{man_sha}' AS manifest_sha,
             '{tax_sha}' AS taxonomy_sha, '{started}' AS labelled_at
      FROM conn c LEFT JOIN matched m USING (uid)"""


# ---------------------------------------------------------------- reporting

def write_report(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main():
    a = parse_args()
    bucket, region, parq, lbl = load_config(a.config)
    man, rules = load_rules(a.dataset)
    validate_rules(rules)

    base = f"s3://{bucket}/{parq}/{a.dataset}/"
    src = a.source or f"{base}**/conn.parquet"
    # labelling writes by default -- the point of the stage is labelled data
    out = None if a.no_write else (a.out or f"s3://{bucket}/{lbl}/{a.dataset}/conn.parquet")
    lake = Lake(region)
    man_sha = sha(MANIFESTS / f"{a.dataset}.yaml")
    tax_sha = sha(MANIFESTS / "taxonomy.yaml")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"dataset   {a.dataset}")
    print(f"source    {src}")
    print(f"output    {out or '(report only — --no-write)'}")
    print(f"manifest  {man_sha}   taxonomy {tax_sha}")
    print(f"rules     {len(rules)} windows -> "
          f"{len(set(r['canonical'] for r in rules))} canonical classes\n")

    total = lake.sql(f"SELECT count(*) FROM read_parquet('{src}')")[0][0]
    rows = lake.sql(attack_query(rules, src, base))

    attack = sum(r[3] for r in rows)
    overlaps = sum(r[4] for r in rows)

    print(f"{'canonical class':<16}{'raw attack name':<26}{'flows':>14}")
    for cls, raw, _, n, _ in rows:
        print(f"{cls:<16}{raw:<26}{n:>14,}")
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

    report = {
        "dataset": a.dataset, "source": src, "started_utc": started,
        "manifest_sha": man_sha, "taxonomy_sha": tax_sha,
        "rules_total": len(rules), "rules_fired": len(fired), "rules_silent":
            [{"rule_id": r["rid"], "name": r["name"], "date": r["date"],
              "start": r["start"], "end": r["end"]} for r in silent],
        "flows_total": total, "flows_attack": attack, "flows_benign": total - attack,
        "attack_ratio": (attack / total) if total else 0.0,
        "flows_multi_rule": overlaps,
        "by_class": dict({c: sum(r[3] for r in rows if r[0] == c) for c in {r[0] for r in rows}},
                         **{"benign": total - attack}),
        "by_rule": [{"rule_id": r[2], "raw": r[1], "class": r[0], "flows": r[3]}
                    for r in rows],
        "output": out,
    }
    rpath = a.report or REPORTS / f"labelling_{a.dataset}.json"
    write_report(rpath, report)
    print(f"\nrun report -> {rpath}")

    if out:
        print(f"writing labelled flows -> {out}")
        lq = labelled_query(rules, src, base, a.dataset, man_sha, tax_sha, started)
        lake.sql(f"COPY ({lq}) TO '{out}' (FORMAT PARQUET)")
        print(f"done — {total:,} labelled flows written "
              f"({attack:,} attack, {total - attack:,} benign)")


if __name__ == "__main__":
    main()
