#!/usr/bin/env python3
"""Draw the sample a person adjudicates, with the evidence needed to judge it.

Every label in this lake comes from a published attack schedule -- a time window
and a pair of addresses, transcribed from a web page and applied to Zeek conn
flows. The validation checks can prove a label is internally inconsistent, rank
flows a model disbelieves, and show where the schedule and the traffic disagree.
None of them can say what share of the labels are *right*, because all of them
reason from the same document the labels came from. Only a human looking at
flows and saying yes or no produces a number that is a measurement.

This file draws that sample and hands it over in a form that can be worked
through. `score.py` turns the returned verdicts into per-class precision with
confidence intervals. Between them they are the difference between "our labels
look right" and a stated figure with an interval and a method attached.

Two properties make the sample defensible rather than merely convenient:

  * It is a deterministic function of the data. Membership is decided by
    hash(uid || salt), not by DuckDB's USING SAMPLE -- reservoir sampling is not
    reproducible across thread counts, so a redraw on another machine would
    produce different rows and the resulting figure could never be checked.
    Concretely, the sample is the `quota` flows of each class with the SMALLEST
    hash(uid || salt) % 1e6. That definition has a useful consequence: raising
    --per-class extends the sample rather than replacing it, so a partly
    adjudicated file can be topped up and no work is thrown away.
  * The sampling frame goes out beside it. Per-class population, quota drawn,
    seed, salt, manifest and taxonomy hashes, and the exact SQL, all in a JSON
    sidecar. Without the frame a sample is an anecdote: nobody can tell what it
    was drawn from, reweight it to the population, or draw it again.

Sizing the effort. The half-width of a 95% interval on a proportion p measured
from n adjudications is about 1.96*sqrt(p(1-p)/n):

    n        p = 0.99    p = 0.95    p = 0.90    p = 0.50
     100      +/-1.9pp    +/-4.3pp    +/-5.9pp    +/-9.8pp
     400      +/-1.0pp    +/-2.1pp    +/-2.9pp    +/-4.9pp
    1000      +/-0.6pp    +/-1.4pp    +/-1.9pp    +/-3.1pp
    2000      +/-0.4pp    +/-1.0pp    +/-1.3pp    +/-2.2pp

400 per class is the default: it buys +/-2-3pp where a weak-supervision label
set is expected to land, at roughly a day of one person's attention per dataset.
2000 buys ~+/-1pp for five times the work. The case that decides the default is
the clean one -- if all n adjudications come back correct, the Wilson lower
bound is n/(n + z^2), so 400 perfect adjudications support "at least 99.05%",
2000 support 99.81%, and 100 support only 96.3%. A 99% claim cannot be made from
100 flows however clean they are. (Those figures are the normal approximation,
used here only for sizing; score.py reports Wilson intervals, which is what the
numbers above are checked against.)

The evidence bundle is the point of the export. An adjudicator given a uid and a
class has to go back to the lake for every single flow; an adjudicator given the
5-tuple, the conn summary, the rule window the flow fell in, how far into that
window it started, and whatever http/ssl/dns/ssh/ftp/weird records Zeek attached
to the same uid can decide most flows in seconds. Zeek's other logs are the load
-bearing part: they were produced by parsing packets and know nothing about the
schedule, so they are the only evidence in the export that could contradict it.

Usage:
    python src/label_validation/certify/sample.py --dataset cic-ids-2017
    python src/label_validation/certify/sample.py --dataset cic-ids-2017 --per-class 2000
    python src/label_validation/certify/sample.py --dataset toy --lake-root /tmp/fixture --no-s3
    # re-attach evidence to a part-adjudicated file, keeping the verdicts:
    python src/label_validation/certify/sample.py --dataset toy --rehydrate reports/validation/samples/toy.csv
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import yaml

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "config.yml").is_file())
sys.path.insert(0, str(ROOT))

from src.common.lake import Lake                                    # noqa: E402
from src.label_validation.core import paths as _paths  # noqa: E402
from src.preprocess.label.label_flows import (                      # noqa: E402
    MANIFESTS, load_rules, sha, validate_rules,
)
from src.label_validation.runner import load_config, probe_sources          # noqa: E402

REPORTS = ROOT / "reports" / "validation" / "samples"
POLICY = _paths.POLICY

# hash(uid || salt) is taken modulo this before it is compared or ordered, so
# the sampling predicate is a small integer that can be pasted into a repro
# query and re-run by hand.
MOD = 1_000_000

# Conn columns carried into the CSV, in reading order. Both byte pairs are
# exported deliberately: *_bytes is Zeek's sequence-number-derived payload and
# wraps on scan and RST traffic (docs/label_evidence.md B7), *_ip_bytes is
# counted off the wire. An adjudicator looking at a "1.9 GB" scan flow needs to
# see both to know which number to believe.
CONN_COLUMNS = [
    ("ts", "DOUBLE"), ("uid", "VARCHAR"),
    ("id.orig_h", "VARCHAR"), ("id.orig_p", "BIGINT"),
    ("id.resp_h", "VARCHAR"), ("id.resp_p", "BIGINT"),
    ("proto", "VARCHAR"), ("service", "VARCHAR"), ("duration", "DOUBLE"),
    ("orig_bytes", "BIGINT"), ("resp_bytes", "BIGINT"),
    ("orig_ip_bytes", "BIGINT"), ("resp_ip_bytes", "BIGINT"),
    ("orig_pkts", "BIGINT"), ("resp_pkts", "BIGINT"),
    ("conn_state", "VARCHAR"), ("history", "VARCHAR"), ("capture", "VARCHAR"),
    ("label_class", "VARCHAR"), ("label_raw", "VARCHAR"), ("rule_id", "INTEGER"),
    ("label_executed", "BOOLEAN"), ("label_source", "VARCHAR"),
    ("rules_matched", "BIGINT"),
]

# The independent evidence channels, joined to conn on uid. (csv column, zeek
# column, type, aggregate). One flow can produce many records in a log, so each
# field is collapsed to one cell; min() rather than "the first by ts" because a
# redraw has to reproduce the same cell exactly, and ties on ts do not. The
# per-log record count travels alongside so a collapsed cell is never mistaken
# for the whole story. ssh is the exception -- the largest auth_attempts and
# whether any attempt succeeded are the facts that adjudicate a brute-force
# label, not the alphabetically smallest ones.
EVIDENCE = {
    "http": [("http_method", "method", "VARCHAR", "min"),
             ("http_uri", "uri", "VARCHAR", "min"),
             ("http_status", "status_code", "BIGINT", "min"),
             ("http_host", "host", "VARCHAR", "min"),
             ("http_user_agent", "user_agent", "VARCHAR", "min")],
    "ssl": [("ssl_server_name", "server_name", "VARCHAR", "min"),
            ("ssl_version", "version", "VARCHAR", "min")],
    "dns": [("dns_query", "query", "VARCHAR", "min"),
            ("dns_qtype", "qtype_name", "VARCHAR", "min"),
            ("dns_rcode", "rcode_name", "VARCHAR", "min")],
    "ssh": [("ssh_auth_attempts", "auth_attempts", "BIGINT", "max"),
            ("ssh_auth_success", "auth_success", "BOOLEAN", "bool_or"),
            ("ssh_client", "client", "VARCHAR", "min"),
            ("ssh_server", "server", "VARCHAR", "min")],
    "ftp": [("ftp_user", "user", "VARCHAR", "min"),
            ("ftp_command", "command", "VARCHAR", "min"),
            ("ftp_reply_code", "reply_code", "BIGINT", "min")],
    "weird": [("weird_name", "name", "VARCHAR", "min"),
              ("weird_addl", "addl", "VARCHAR", "min")],
}
EVIDENCE_ORDER = ["http", "ssl", "dns", "ssh", "ftp", "weird"]

# What the adjudicator fills in. Kept at the far left of the CSV so neither
# column needs scrolling to on a forty-column sheet.
VERDICT_COLUMNS = ["verdict", "verdict_note"]

# Everything the labelling stage knew about WHY this flow carries this class,
# derived from the manifest rather than the lake.
RULE_COLUMNS = ["rule_name", "rule_window_local", "rule_window_utc",
                "rule_endpoints", "rule_needs_payload", "rule_offset_s",
                "rule_window_s", "nearest_rule", "nearest_rule_gap_s"]

TMP = "talos_cert_sample"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--per-class", type=int, default=None,
                   help="flows to draw per label_class (default: policy, or 400)")
    p.add_argument("--seed", type=int, default=None,
                   help="changes the draw; recorded in the sidecar (default: policy, or 1729)")
    p.add_argument("--salt", default=None,
                   help="hash salt; with --seed it fixes membership entirely")
    p.add_argument("--rehydrate", default=None,
                   help="an adjudicated CSV: redraw the frame, re-attach evidence, "
                        "and carry every verdict across by uid")
    p.add_argument("--out", default=None,
                   help="CSV path (default: reports/validation/samples/<dataset>.csv)")
    p.add_argument("--frame", default=None,
                   help="JSON sidecar path (default: alongside the CSV)")
    p.add_argument("--config", default=str(ROOT / "config.yml"))
    p.add_argument("--policy", default=str(POLICY))
    p.add_argument("--lake-root", default=None,
                   help="read parquet from this root instead of s3://<bucket>")
    p.add_argument("--no-s3", action="store_true")
    # Same guard rails as the runner: this scans the labelled zone once.
    p.add_argument("--memory-limit", default="8GB")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--temp-dir", default=None)
    p.add_argument("--max-temp", default="20GB")
    return p.parse_args()


def threshold(policy, key, default=None):
    """policy.yaml is read directly here -- this is a CLI, not a registered check.

    Missing keys fall back to the default rather than failing, so the sampler
    runs against a policy file that has not been extended yet; whatever value
    was used is written into the sidecar either way.
    """
    return (policy.get("thresholds") or {}).get(key, default)


def lit(v):
    return "'" + str(v).replace("'", "''") + "'"


# ------------------------------------------------------------------- the draw

def class_populations(lake, glob):
    """label_class -> flows. The denominator for every rate score.py reports.

    Also the weights that turn a stratified sample back into a statement about
    the dataset: the draw flattens the prior deliberately, so without these the
    per-class figures cannot be combined into an overall one.
    """
    rows = lake.sql(f"SELECT label_class, count(*) FROM read_parquet({lit(glob)}, "
                    f"union_by_name=true) GROUP BY 1 ORDER BY 2 DESC")
    return {str(r[0]): int(r[1]) for r in rows}


def quotas(populations, per_class):
    """How many flows to draw from each class.

    Equal quotas, capped at the class's own size. A class smaller than the quota
    is adjudicated in full -- that is a census, not a sample, and score.py marks
    it as one so its interval is not read as sampling error that does not exist.
    """
    return {c: min(per_class, n) for c, n in populations.items() if n > 0}


def sample_sql(glob, columns, populations, quota, salt, cuts):
    """The `quota` flows of each class with the smallest hash(uid || salt).

    Two stages, and the second is the definition -- the first is only there to
    keep the cost down. The WHERE clause admits every flow whose hash falls
    under a per-class cut chosen to let through a few times the quota; the
    window function then ranks what survived and keeps exactly the quota. Since
    ranking is by the same hash the cut was applied to, the result is precisely
    "the quota smallest hashes in the class" as long as the cut admitted at
    least that many, which the caller verifies and widens until true.

    Ordering by hash and truncating, rather than testing hash < cut alone, is
    what makes the quota exact and the samples nested: a larger quota keeps
    every row a smaller one kept.
    """
    sel = ", ".join(f'"{c}"' if c in columns else f'CAST(NULL AS {t}) AS "{c}"'
                    for c, t in CONN_COLUMNS)
    cut_case = " ".join(f"WHEN {lit(c)} THEN {n}" for c, n in sorted(cuts.items()))
    quota_case = " ".join(f"WHEN {lit(c)} THEN {n}" for c, n in sorted(quota.items()))
    return f"""
WITH admitted AS (
  SELECT {sel},
         hash(uid || {lit(salt)}) % {MOD} AS sample_hash
  FROM read_parquet({lit(glob)}, union_by_name=true)
  WHERE uid IS NOT NULL
    AND (hash(uid || {lit(salt)}) % {MOD}) <
        CASE label_class {cut_case} ELSE 0 END
), ranked AS (
  SELECT *, row_number() OVER (PARTITION BY label_class
                               ORDER BY sample_hash, uid) AS pick
  FROM admitted
)
SELECT * EXCLUDE (pick) FROM ranked
WHERE pick <= CASE label_class {quota_case} ELSE 0 END""".strip()


def cuts_for(populations, quota, margin):
    """Per-class hash cut: enough headroom that the quota is reachable.

    The hash is uniform over [0, MOD), so a cut of MOD*quota*margin/pop admits
    about `margin` times the quota. margin=4 makes falling short of 400 a
    ~1e-30 event, but it is checked rather than assumed, because a shortfall
    would silently under-sample a class instead of erroring.
    """
    out = {}
    for c, pop in populations.items():
        want = quota.get(c, 0)
        out[c] = MOD if want >= pop else min(MOD, int(MOD * want * margin / pop) + 64)
    return out


def draw(lake, glob, columns, populations, quota, salt):
    """Materialise the sample, widening the prefilter until every quota is met.

    Returns (sql, attempts). The temp table is the join target for the evidence
    logs below, so the labelled zone is scanned exactly once however many logs
    are attached.
    """
    attempts = []
    for margin in (4, 16, None):
        cuts = ({c: MOD for c in populations} if margin is None
                else cuts_for(populations, quota, margin))
        sql = sample_sql(glob, columns, populations, quota, salt, cuts)
        lake.sql(f"CREATE OR REPLACE TEMP TABLE {TMP} AS\n{sql}")
        drawn = {str(r[0]): int(r[1]) for r in
                 lake.sql(f"SELECT label_class, count(*) FROM {TMP} GROUP BY 1")}
        short = {c: quota[c] - drawn.get(c, 0) for c in quota
                 if drawn.get(c, 0) < quota[c]}
        attempts.append({"prefilter_margin": margin, "cuts": cuts,
                         "short_of_quota": short})
        if not short:
            return sql, drawn, attempts
    return sql, drawn, attempts


# -------------------------------------------------------------- the evidence

def evidence_plan(lake, sources, available):
    """Which logs can be joined, which columns each supplies, and why not.

    Zeek's schema drifts inside a single dataset and between them -- ftp.log
    exists only for 2017, 2019's dns.log loses `query` in most files -- so every
    column reference is checked against the log's own union schema first. A log
    that is absent is recorded as absent; the sidecar has to distinguish "no
    HTTP evidence for this flow" from "this dataset has no http.log", because
    only the first is evidence about the label.
    """
    plan, report = [], {}
    for log in EVIDENCE_ORDER:
        if log not in available:
            report[log] = {"present": False, "joined": False,
                           "reason": "no files in the lake for this dataset"}
            continue
        try:
            schema = lake.columns(sources[log])
        except Exception as exc:                       # noqa: BLE001
            report[log] = {"present": True, "joined": False,
                           "reason": f"unreadable: {type(exc).__name__}: {exc}"}
            continue
        if "uid" not in schema:
            report[log] = {"present": True, "joined": False,
                           "reason": "no uid column, so it cannot be joined to a flow"}
            continue
        fields = [f for f in EVIDENCE[log] if f[1] in schema]
        plan.append((log, fields))
        report[log] = {"present": True, "joined": True,
                       "columns_used": [f[1] for f in fields],
                       "columns_missing": [f[1] for f in EVIDENCE[log]
                                           if f[1] not in schema]}
    return plan, report


def evidence_sql(sources, plan):
    """One aggregate per log, semi-joined to the sample, then left-joined on.

    The `uid IN (SELECT uid FROM sample)` is what keeps this affordable: each
    log is read once and reduced to at most one row per sampled flow before
    anything is joined, so the cost is a scan of the log rather than a scan
    multiplied by the number of evidence columns.
    """
    ctes, joins, cols = [], [], []
    for log, fields in plan:
        agg = ", ".join(f'{a}("{col}") AS {out}' for out, col, _t, a in fields)
        ctes.append(f"""ev_{log} AS (
  SELECT uid, count(*) AS {log}_records{", " + agg if agg else ""}
  FROM read_parquet({lit(sources[log])}, union_by_name=true)
  WHERE uid IS NOT NULL AND uid IN (SELECT uid FROM {TMP})
  GROUP BY uid
)""")
        joins.append(f"LEFT JOIN ev_{log} ON ev_{log}.uid = s.uid")
        cols.append(f"coalesce(ev_{log}.{log}_records, 0) AS {log}_records")
        cols += [f"ev_{log}.{out}" for out, _c, _t, _a in fields]
    if not ctes:
        return f"SELECT * FROM {TMP} s"
    return (f"WITH {', '.join(ctes)}\n"
            f"SELECT s.*, {', '.join(cols)}\nFROM {TMP} s\n" + "\n".join(joins))


# ------------------------------------------------------- rules -> readable why

def rule_index(rules):
    return {int(r["rid"]): r for r in rules}


def utc(ts):
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat(timespec="seconds")


def rule_fields(rule, ts):
    """The window a flow was labelled by, as an adjudicator needs to read it.

    `rule_offset_s` is the one that decides boundary cases: a flow starting one
    second into a six-hour window and a flow starting one second before it
    closes carry the same label with very different claims behind them, and the
    known failure mode in this project is exactly a window edge in the wrong
    place (docs/label_evidence.md A2, B3).
    """
    ends = (f"{utc(rule['t0'])} .. {utc(rule['t1'])}"
            + (f" (+{rule['extended_by']:.0f}s minute-end extension)"
               if rule.get("extended_by") else ""))
    peers = " -> ".join(x for x in (
        ",".join(rule["attackers"]) if rule["attackers"] else "(any)",
        ",".join(rule["victims"]) if rule["victims"] else
        (rule["prefix"] + "0/24" if rule["prefix"] else "(any)")) if x)
    return {
        "rule_name": rule["name"],
        "rule_window_local": f"{rule['date']} {rule['start']}-{rule['end']}",
        "rule_window_utc": ends,
        "rule_endpoints": peers,
        "rule_needs_payload": rule["needs_payload"],
        "rule_offset_s": (round(float(ts) - rule["t0"], 3) if ts is not None else None),
        "rule_window_s": round(rule["t1"] - rule["t0"], 3),
        "nearest_rule": None, "nearest_rule_gap_s": None,
    }


def nearest_window(rules, ts):
    """For a benign flow: which attack window it sits closest to, and how far.

    Benign in this lake is a closed-world default -- everything no rule claimed.
    That makes the interesting benign flows the ones just outside a window, and
    a signed gap in seconds is what tells an adjudicator whether they are
    looking at ordinary traffic or at the tail of an attack the schedule cut off
    too early.
    """
    best = None
    for r in rules:
        if r["t0"] <= ts < r["t1"]:
            gap = 0.0
        elif ts < r["t0"]:
            gap = ts - r["t0"]
        else:
            gap = ts - r["t1"]
        if best is None or abs(gap) < abs(best[1]):
            best = (r, gap)
    return best


def blank_rule_fields(rules, ts):
    out = {k: None for k in RULE_COLUMNS}
    if ts is None or not rules:
        return out
    rule, gap = nearest_window(rules, float(ts))
    out["nearest_rule"] = rule["name"]
    out["nearest_rule_gap_s"] = round(gap, 3)
    return out


# ---------------------------------------------------------------- CSV writing

def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


def build_rows(records, names, rules, verdicts):
    """Query rows -> CSV dicts, with the rule context and the verdict columns.

    `verdicts` carries anything already adjudicated (rehydrate); a uid absent
    from it gets an empty pair of cells, which is what score.py counts as
    unadjudicated rather than as agreement.
    """
    idx = rule_index(rules)
    evidence_cols = [n for n in names if n.endswith("_records")]
    out = []
    for rec in records:
        row = dict(zip(names, rec))
        ts, uid, rid = row["ts"], str(row["uid"]), row["rule_id"]
        rule = idx.get(int(rid)) if rid is not None else None
        ctxt = rule_fields(rule, ts) if rule else blank_rule_fields(rules, ts)
        seen = [c[:-len("_records")] for c in evidence_cols if (row.get(c) or 0) > 0]
        got = verdicts.get(uid, {})
        out.append({
            "verdict": got.get("verdict", ""),
            "verdict_note": got.get("verdict_note", ""),
            "ts_utc": utc(ts) if ts is not None else "",
            "evidence_logs": ",".join(seen),
            **ctxt, **row,
        })
    return out


def csv_header(available, plan):
    """Reading order: what to fill in, what is under test, then the evidence."""
    lead = ["uid", "label_class", "label_raw", "rule_id", "label_executed",
            "label_source", "rules_matched"]
    conn = ["ts", "ts_utc", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
            "proto", "service", "duration", "orig_bytes", "resp_bytes",
            "orig_ip_bytes", "resp_ip_bytes", "orig_pkts", "resp_pkts",
            "conn_state", "history", "capture"]
    ev = ["evidence_logs"]
    for log, fields in plan:
        ev.append(f"{log}_records")
        ev += [f[0] for f in fields]
    header = VERDICT_COLUMNS + lead + RULE_COLUMNS + conn + ev + ["sample_hash"]
    return [c for c in header if c in available]


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in header])


def read_verdicts(path):
    """Adjudications already recorded in a CSV, keyed by uid.

    utf-8-sig because the file comes back from a spreadsheet, and Excel writes a
    BOM that would otherwise turn the first header into '\\ufeffverdict' and
    silently lose every verdict in the file.
    """
    out, rows = {}, 0
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if "uid" not in (reader.fieldnames or []):
            sys.exit(f"{path} has no `uid` column — verdicts cannot be matched to flows")
        for row in reader:
            rows += 1
            uid = (row.get("uid") or "").strip()
            verdict = (row.get("verdict") or "").strip()
            note = (row.get("verdict_note") or "").strip()
            if uid and (verdict or note):
                out[uid] = {"verdict": verdict, "verdict_note": note}
    return out, rows


# ----------------------------------------------------------------------- main

def main():
    a = parse_args()
    cfg = load_config(a.config)
    policy = yaml.safe_load(Path(a.policy).read_text()) or {}

    per_class = a.per_class or int(threshold(policy, "certification_per_class_sample", 400))
    seed = a.seed if a.seed is not None else int(threshold(policy, "certification_seed", 1729))
    salt_base = a.salt or str(threshold(policy, "certification_sample_salt", "talos.cert"))
    # Seed and salt are folded into one string because the sampling predicate is
    # a hash of it; keeping them separate in the sidecar keeps the operator's
    # two knobs visible, and either one alone reproduces nothing.
    salt = f"{salt_base}:{seed}"

    bucket, region = cfg["aws"]["bucket"], cfg["aws"]["region"]
    root = a.lake_root.rstrip("/") if a.lake_root else f"s3://{bucket}"
    require_s3 = not a.no_s3 and root.startswith("s3://")
    role = (cfg.get("datasets", {}).get(a.dataset) or {}).get("role", "unknown")

    man, rules = load_rules(a.dataset)
    validate_rules(rules)

    lake = Lake(region, a.memory_limit, a.temp_dir, a.threads, a.max_temp,
                require_s3=require_s3)
    sources, available = probe_sources(lake, root, a.dataset, cfg, require_s3)
    if "labelled" not in available:
        sys.exit(f"no labelled zone for {a.dataset} under {root} — run labelling first")
    columns = lake.columns(sources["labelled"])

    carried, carried_rows = ({}, 0)
    if a.rehydrate:
        carried, carried_rows = read_verdicts(a.rehydrate)

    populations = class_populations(lake, sources["labelled"])
    quota = quotas(populations, per_class)
    total_pop = sum(populations.values())

    print(f"dataset    {a.dataset}  ({role})")
    print(f"root       {root}")
    print(f"population {total_pop:,} flow(s) in {len(populations)} class(es)")
    print(f"draw       {per_class} per class, salt {salt!r}")
    print(f"rules      {len(rules)} windows   manifest "
          f"{sha(MANIFESTS / f'{a.dataset}.yaml')} "
          f"taxonomy {sha(MANIFESTS / 'taxonomy.yaml')}")
    if a.rehydrate:
        print(f"rehydrate  {a.rehydrate}: {len(carried)} adjudicated of "
              f"{carried_rows} row(s)")

    sql, drawn, attempts = draw(lake, sources["labelled"], columns,
                                populations, quota, salt)
    plan, ev_report = evidence_plan(lake, sources, available)
    join_sql = evidence_sql(sources, plan)

    lake.refresh()
    cur = lake.con.execute(join_sql)
    names = [d[0] for d in cur.description]
    records = cur.fetchall()

    rows = build_rows(records, names, rules, carried)
    # Class then hash: stable across redraws, and it puts a class's flows
    # together so an adjudicator builds up context instead of switching every
    # row. Within a class the order is the draw order, so a partly-worked file
    # and its top-up align line for line.
    rows.sort(key=lambda r: (str(r["label_class"]), int(r["sample_hash"]), str(r["uid"])))
    header = csv_header(set(names) | set(RULE_COLUMNS) | set(VERDICT_COLUMNS)
                        | {"ts_utc", "evidence_logs"}, plan)

    out = Path(a.out) if a.out else REPORTS / f"{a.dataset}.csv"
    frame_path = Path(a.frame) if a.frame else out.with_suffix(".frame.json")
    write_csv(out, header, rows)

    drawn_uids = {str(r["uid"]) for r in rows}
    orphans = {u: v for u, v in carried.items() if u not in drawn_uids}
    reattached = len(carried) - len(orphans)

    frame = {
        "certification_frame_version": 1,
        "meta": {
            "dataset": a.dataset, "role": role, "root": root,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest_sha": sha(MANIFESTS / f"{a.dataset}.yaml"),
            "taxonomy_sha": sha(MANIFESTS / "taxonomy.yaml"),
            "manifest_dataset": man.get("dataset"),
            "policy_version": policy.get("version"),
            "duckdb_version": duckdb.__version__,
            "rules_total": len(rules),
        },
        # Everything needed to draw this exact sample again, on another machine,
        # at another thread count. A sample that cannot be redrawn cannot be
        # defended, extended, or audited.
        "sampling": {
            "method": f"the quota flows of each class with the smallest "
                      f"hash(uid || salt) % {MOD}",
            "seed": seed, "salt_base": salt_base, "salt": salt,
            "per_class": per_class, "modulus": MOD,
            "prefilter_attempts": attempts,
            "nested": "raising --per-class extends this sample rather than "
                      "replacing it, so adjudications are never invalidated",
        },
        "population": populations,
        "population_total": total_pop,
        "quota": quota,
        "drawn": drawn,
        "rows": len(rows),
        "census_classes": sorted(c for c in quota if quota[c] >= populations[c]),
        "evidence_logs": ev_report,
        "verdict_vocabulary": ["correct", "wrong", "uncertain", "benign",
                               "<a specific label_class>"],
        "sql": {"sample": sql, "evidence": join_sql},
        "csv": str(out),
        "rehydrated_from": a.rehydrate,
        "verdicts_carried": reattached,
        # Kept rather than dropped: an adjudication that no longer belongs to
        # the frame is still work someone did, and the only place it survives a
        # redraw is here.
        "orphaned_verdicts": [dict(uid=u, **v) for u, v in sorted(orphans.items())],
    }
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_text(json.dumps(frame, indent=2, default=str) + "\n")

    print(f"\n{'class':<18}{'population':>14}{'share':>9}{'quota':>9}"
          f"{'drawn':>9}  note")
    for c in sorted(populations, key=lambda c: -populations[c]):
        note = "census — every flow adjudicated" if quota.get(c, 0) >= populations[c] else ""
        print(f"{c:<18}{populations[c]:>14,}{populations[c] / total_pop:>9.2%}"
              f"{quota.get(c, 0):>9,}{drawn.get(c, 0):>9,}  {note}")
    logs = [f"{k}({'+' if v['joined'] else '-'})" for k, v in ev_report.items()]
    print(f"\nevidence   {' '.join(logs) or '(none)'}")
    if reattached or orphans:
        print(f"carried    {reattached} verdict(s) re-attached, "
              f"{len(orphans)} orphaned (kept in the sidecar)")
    print(f"sample  -> {out}   {len(rows):,} row(s), "
          f"{len(header)} column(s)")
    print(f"frame   -> {frame_path}")
    print("\nfill in the `verdict` column (correct / wrong / uncertain / benign / "
          "a class name),\nthen: python src/label_validation/certify/score.py --dataset "
          f"{a.dataset} --sample {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
