#!/usr/bin/env python3
"""Tier 2 -- does the traffic behave the way the label says it does?

Tier 0 established that the labelled zone holds together and Tier 1 that the
windows match the schedule. Neither reads a packet, so both pass a window that
was transcribed perfectly and aimed at the wrong traffic. That failure mode is
not hypothetical: 2018's DoS-SlowHTTPTest window puts 105,550 flows on port 21,
and its FTP-BruteForce class is 192,294 refused connections out of 192,300. By
every check that only reads the manifest, both are correctly labelled.

So this tier asserts what each class must LOOK like and measures whether it
does. The assertions live in expectations.yaml -- per canonical class, and per
raw attack name where one band cannot describe a class, because `dos` spans Hulk
(fast, loud, payload-carrying) and Slowloris (long, near-silent) and no single
band is true of both. Nothing here knows what a portscan is; adding a class is a
YAML edit.

A finding is never a verdict on the labels alone. The band is falsifiable too,
so every finding reports the measured rate beside the floor it missed and the
SQL that produced it, and the reader decides which of the two is wrong.

Everything is measured in two passes over the labelled zone: one aggregate keyed
by (label_class, label_raw) carrying every predicate as a conditional sum, and
one destination-port distribution. The 2019 zone is 72.5M flows -- a query per
expectation would be forty scans of it.
"""

from pathlib import Path

import yaml

from src.label_validation.core.finding import Finding, Severity
from src.label_validation.core import paths as _paths
from src.label_validation.core.registry import check

EXPECTATIONS = _paths.EXPECTATIONS

# Counters computed for every group whatever the YAML asks for: the executed
# ratio and the fan-out ratios are derived from these, so no expectation can
# cost a scan.
FIXED = ("flows", "exec_true", "exec_false", "exec_null",
         "distinct_orig_h", "distinct_resp_h", "distinct_orig_p", "distinct_resp_p")
IDX = {name: i for i, name in enumerate(FIXED)}
DISTINCT = {"distinct_orig_h": "id.orig_h", "distinct_resp_h": "id.resp_h",
            "distinct_orig_p": "id.orig_p", "distinct_resp_p": "id.resp_p"}

_DOC = None
_SCANS = {}


def _doc():
    global _DOC
    if _DOC is None:
        _DOC = yaml.safe_load(EXPECTATIONS.read_text())
    return _DOC


def _sql(text):
    """One line, so the same band written in two places shares one column."""
    return " ".join(str(text).split())


def _lit(value):
    return "'" + str(value).replace("'", "''") + "'"


def _defaults(doc, key, fallback):
    return (doc.get("defaults") or {}).get(key, fallback)


class Pool:
    """Every predicate in expectations.yaml, deduplicated by its SQL text.

    A predicate whose columns the zone does not have gets no column in the
    aggregate at all. It is recorded in `missing` and reported as unjudged --
    never counted as a pass, which is the failure mode that would make this whole
    tier a rubber stamp on a dataset missing `service`.
    """

    def __init__(self):
        self.order, self.slot, self.missing, self.signature = [], {}, {}, {}

    def add(self, sql, requires, columns):
        sql = _sql(sql)
        absent = [c for c in requires if c not in columns]
        if absent:
            self.missing[sql] = absent
            return None
        if sql not in self.slot:
            self.slot[sql] = len(self.order)
            self.order.append(sql)
        return sql

    def share(self, counts, sql):
        n = counts[IDX["flows"]]
        return counts[len(FIXED) + self.slot[sql]] / n if n else 0.0


def _band(doc, label_class, label_raw):
    """The expectation set governing one group, override merged over class."""
    base = (doc.get("classes") or {}).get(label_class)
    if base is None:
        return None
    over = (doc.get("raw_overrides") or {}).get(label_raw) if label_raw else None
    return {**base, **over} if over else base


def _build_pool(doc, columns):
    pool = Pool()
    bands = list((doc.get("classes") or {}).items()) \
        + list((doc.get("raw_overrides") or {}).items())
    for _, band in bands:
        for p in band.get("predicates") or []:
            pool.add(p["sql"], p.get("requires") or [], columns)
    # Signatures are conjunctions of a class's own predicates and belong to the
    # class, not to a raw attack: benign contamination is a question about the
    # boundary between benign and a class, which no single attack owns.
    for cls, band in (doc.get("classes") or {}).items():
        names = ((band.get("signature") or {}).get("all_of")) or []
        by_name = {p["name"]: p for p in band.get("predicates") or []}
        if not names or any(n not in by_name for n in names):
            continue
        conj = " AND ".join(f"({by_name[n]['sql']})" for n in names)
        needs = [c for n in names for c in (by_name[n].get("requires") or [])]
        if pool.add(conj, needs, columns):
            pool.signature[cls] = _sql(conj)
    return pool


def _scan(ctx):
    """The one aggregate every check in this tier reads.

    Cached per (dataset, source) because the runner builds a Context once per
    dataset and runs the tier's checks back to back: four checks, two scans. The
    grain is (label_class, label_raw) so a per-attack band and a per-class band
    can both be evaluated without a second pass.
    """
    key = (ctx.dataset, ctx.q("labelled"))
    if key in _SCANS:
        return _SCANS[key]
    doc = _doc()
    pool = _build_pool(doc, ctx.columns)
    executed = "label_executed" in ctx.columns

    def when(cond):
        return f"sum(CASE WHEN {cond} THEN 1 ELSE 0 END)" if executed else "0"

    sel = ["label_class", "label_raw", "count(*)",
           when("label_executed"), when("NOT label_executed"),
           when("label_executed IS NULL")]
    sel += [f'approx_count_distinct("{c}")' if c in ctx.columns else "0"
            for c in DISTINCT.values()]
    sel += [f"sum(CASE WHEN ({s}) THEN 1 ELSE 0 END)" for s in pool.order]
    query = (f"SELECT {', '.join(sel)}\n"
             f"FROM read_parquet('{ctx.q('labelled')}', union_by_name=true)\n"
             f"GROUP BY 1, 2")
    groups = {(r[0], r[1]): tuple(int(v or 0) for v in r[2:])
              for r in ctx.lake.sql(query)}
    _SCANS[key] = (doc, pool, groups)
    return _SCANS[key]


def _fold(doc, groups):
    """Fold (class, raw) counts to the grain each band actually describes.

    A raw attack with its own override is judged alone; the rest of a class is
    judged together, because a class band is a claim about the class.
    """
    overrides = doc.get("raw_overrides") or {}
    approx = [IDX[m] for m in DISTINCT]
    folded = {}
    for (cls, raw), counts in groups.items():
        key = (cls, raw if raw in overrides else None)
        prior = folded.get(key)
        if prior is None:
            folded[key] = counts
            continue
        merged = [a + b for a, b in zip(prior, counts)]
        # HyperLogLog cardinalities do not add. The larger member is the only
        # bound a pooled group can honestly claim, so ratio floors are compared
        # against a lower bound of the true distinct count.
        for i in approx:
            merged[i] = max(prior[i], counts[i])
        folded[key] = tuple(merged)
    return folded


def _merge_class(groups, label_class):
    """All groups of one class, summed -- benign should be exactly one of them."""
    rows = [c for (cls, _), c in groups.items() if cls == label_class]
    if not rows:
        return None
    merged = list(rows[0])
    approx = [IDX[m] for m in DISTINCT]
    for counts in rows[1:]:
        merged = [a + b for a, b in zip(merged, counts)]
        for i in approx:
            merged[i] = max(merged[i], counts[i])
    return tuple(merged)


def _where(label_class, label_raw):
    clause = f"label_class = {_lit(label_class)}"
    return clause + (f" AND label_raw = {_lit(label_raw)}" if label_raw else "")


def _severity(worst, failed, evaluated):
    """How wrong, and how broadly wrong.

    One band missed by a little is a band that wants adjusting. Most of a class's
    bands missed by a lot is a class that is not what it says it is, and nothing
    downstream can repair that.
    """
    if worst >= 0.50 and failed * 2 >= evaluated:
        return Severity.CRITICAL
    if worst >= 0.20:
        return Severity.MAJOR
    return Severity.MINOR


@check(id="beh.class_conformance", tier=2,
       title="each class behaves the way its label claims",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def class_conformance(ctx):
    """Declared shape against measured shape, one expectation at a time.

    A failure is not "this class will be hard to model". It says the flows
    carrying a label do not do the thing the label names -- a DoS class whose
    traffic never reaches a web port, a brute-force class that is mostly refused
    connections -- which is a defect in the ground truth, not in the model.
    """
    doc, pool, groups = _scan(ctx)
    slack = ctx.threshold("expectation_slack", 0.05)
    default_min = _defaults(doc, "min_flows", 100)
    findings, unjudged = [], []

    for (cls, raw), counts in sorted(_fold(doc, groups).items(),
                                     key=lambda kv: (kv[0][0], kv[0][1] or "")):
        n = counts[IDX["flows"]]
        label = f"{cls}/{raw}" if raw else cls
        band = _band(doc, cls, raw)
        if band is None:
            unjudged.append({"label_class": cls, "label_raw": raw, "flows": n,
                             "reason": "no expectation set declared for this class"})
            continue
        floor_n = band.get("min_flows", default_min)
        if n < floor_n:
            unjudged.append({"label_class": cls, "label_raw": raw, "flows": n,
                             "reason": f"fewer than {floor_n} flows -- too thin to judge"})
            continue

        results, fails = [], []
        for p in band.get("predicates") or []:
            sql = _sql(p["sql"])
            if sql not in pool.slot:
                absent = ", ".join(pool.missing.get(sql, []))
                unjudged.append({
                    "label_class": cls, "label_raw": raw, "flows": n,
                    "reason": f"expectation '{p['name']}' needs column(s) the "
                              f"labelled zone does not have: {absent}"})
                continue
            rate = pool.share(counts, sql)
            lo, hi = p.get("min_pass"), p.get("max_pass")
            miss = 0.0
            if lo is not None and rate < lo - slack:
                miss = lo - rate
            if hi is not None and rate > hi + slack:
                miss = max(miss, rate - hi)
            row = {"expectation": p["name"], "kind": "predicate",
                   "measured": round(rate, 6),
                   "matched": counts[len(FIXED) + pool.slot[sql]],
                   "min_pass": lo, "max_pass": hi, "miss": round(miss, 6),
                   "sql": sql, "why": _sql(p.get("why", ""))}
            results.append(row)
            if miss:
                fails.append(row)

        for r in band.get("ratios") or []:
            num, den = counts[IDX[r["of"]]], counts[IDX[r["per"]]]
            if not den:
                continue
            value = num / den
            lo, hi = r.get("min"), r.get("max")
            # A ratio is unbounded, so its shortfall is expressed as a fraction
            # of the bound it missed -- otherwise it could not share a severity
            # scale with a pass rate.
            miss = 0.0
            if lo is not None and value < lo * (1 - slack):
                miss = min(1.0, (lo - value) / lo)
            if hi is not None and value > hi * (1 + slack):
                miss = max(miss, min(1.0, (value - hi) / hi))
            row = {"expectation": r["name"], "kind": "ratio",
                   "measured": round(value, 4), "of": r["of"], "per": r["per"],
                   "numerator": num, "denominator": den, "min": lo, "max": hi,
                   "miss": round(miss, 6), "why": _sql(r.get("why", ""))}
            results.append(row)
            if miss:
                fails.append(row)

        if not fails:
            continue
        worst = max(f["miss"] for f in fails)
        preds = [f for f in fails if f["kind"] == "predicate"]
        repro = "SELECT count(*) AS flows"
        repro += "".join(f",\n       sum(CASE WHEN ({f['sql']}) THEN 1 ELSE 0 END) "
                         f"AS {f['expectation']}" for f in preds)
        if any(f["kind"] == "ratio" for f in fails):
            repro += "".join(f',\n       approx_count_distinct("{col}") AS {m}'
                             for m, col in DISTINCT.items())
        repro += (f"\nFROM read_parquet('{ctx.q('labelled')}', union_by_name=true)"
                  f"\nWHERE {_where(cls, raw)}")
        findings.append(Finding(
            check_id="beh.class_conformance", dataset=ctx.dataset,
            severity=_severity(worst, len(fails), len(results)),
            title=f"{label} misses {len(fails)} of {len(results)} behavioural "
                  f"expectation(s), worst by {worst:.1%}",
            detail="The flows carrying this label do not behave the way the label "
                   "names. Either the window is over the wrong traffic, or the band in "
                   "expectations.yaml is wrong -- the measured rate and the floor it "
                   "missed are both reported so the reader can tell which.",
            metrics={"label_class": cls, "label_raw": raw, "flows": n,
                     "expectations": len(results), "failed": len(fails),
                     "worst_miss": round(worst, 6),
                     **{f"rate_{f['expectation']}": f["measured"] for f in fails}},
            evidence=results, scope={"label_class": cls, "label_raw": raw},
            repro=repro,
        ))

    if unjudged:
        findings.append(Finding(
            check_id="beh.class_conformance", dataset=ctx.dataset,
            severity=Severity.INFO,
            title=f"{len(unjudged)} expectation(s) could not be evaluated",
            detail="Recorded rather than dropped: 'nothing could be measured' and "
                   "'everything measured passed' are different claims about the data, "
                   "and only one of them is a pass.",
            metrics={"unevaluated": len(unjudged),
                     "flows": sum(u["flows"] for u in unjudged)},
            evidence=unjudged,
        ))
    return findings


@check(id="beh.benign_contamination", tier=2,
       title="benign traffic does not carry an attack's signature",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def benign_contamination(ctx):
    """The same expectations, pointed the other way.

    Labelling is closed-world: anything no rule matched inside a scheduled
    capture becomes benign. That is only safe if the schedule covers every attack
    that ran. Where it does not, the attack traffic is still in the lake wearing
    a benign label, and a model trained on it learns that the flood is normal.

    A high share here does not name the guilty rule. It says the benign class is
    not a description of benign behaviour, which is enough to stop.
    """
    doc, pool, groups = _scan(ctx)
    benign_class = _defaults(doc, "benign_class", "benign")
    benign = _merge_class(groups, benign_class)
    if not benign or not benign[IDX["flows"]]:
        return []
    n = benign[IDX["flows"]]
    slack = ctx.threshold("expectation_slack", 0.05)
    present = {cls for cls, _ in groups if cls != benign_class}
    findings = []

    for cls, conj in sorted(pool.signature.items()):
        if cls == benign_class:
            continue
        band = (doc.get("classes") or {}).get(cls) or {}
        signature = band.get("signature") or {}
        ceiling = signature.get("max_benign_share")
        if ceiling is None:
            continue
        share = pool.share(benign, conj)
        if share <= ceiling + slack:
            continue
        matched = benign[len(FIXED) + pool.slot[conj]]
        # Each member's own share on benign, so the finding says which part of
        # the signature the benign class is satisfying rather than just that it
        # does. Already in the aggregate; no extra scan.
        members = []
        for p in band.get("predicates") or []:
            sql = _sql(p["sql"])
            if p["name"] in (signature.get("all_of") or []) and sql in pool.slot:
                members.append({"predicate": p["name"],
                                "benign_share": round(pool.share(benign, sql), 6),
                                "sql": sql, "why": _sql(p.get("why", ""))})
        findings.append(Finding(
            check_id="beh.benign_contamination", dataset=ctx.dataset,
            severity=(Severity.CRITICAL if share >= 0.50
                      else Severity.MAJOR if share - ceiling >= 0.10
                      else Severity.MINOR),
            title=f"{share:.1%} of benign flows carry the {cls} signature "
                  f"({matched:,} of {n:,})",
            detail="These flows were labelled benign because no rule matched them, not "
                   "because anything observed them behaving benignly. They satisfy every "
                   "predicate that defines "
                   + (f"{cls}, a class this dataset does label elsewhere -- so the "
                      f"boundary between the two is drawn in the wrong place."
                      if cls in present else
                      f"{cls}, a class this dataset never labels -- so this is attack-"
                      f"shaped traffic no window covers."),
            metrics={"signature_class": cls, "benign_flows": n, "matching": matched,
                     "share": round(share, 6), "max_benign_share": ceiling,
                     "class_labelled_in_dataset": cls in present,
                     **{f"share_{m['predicate']}": m["benign_share"] for m in members}},
            evidence=members, scope={"label_class": benign_class, "signature": cls},
            repro=f"SELECT count(*) AS benign_flows,\n"
                  f"       sum(CASE WHEN ({conj}) THEN 1 ELSE 0 END) AS matching\n"
                  f"FROM read_parquet('{ctx.q('labelled')}', union_by_name=true)\n"
                  f"WHERE label_class = {_lit(benign_class)}",
        ))
    return findings


@check(id="beh.executed_ratio", tier=2,
       title="attacks that require a payload actually sent one",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def executed_ratio(ctx):
    """Flows labelled as an attack the connection never carried out.

    taxonomy.yaml marks the attacks whose execution requires the attacker to send
    bytes, and labelling records label_executed=false for a matched flow that sent
    none. Those flows are connection attempts wearing an attack's name: 2018's
    FTP-BruteForce is 192,294 of them out of 192,300, and 2017's WebAttack-XSS is
    652 of 679. A class made almost entirely of them teaches a model that a
    refused TCP handshake is an FTP brute force, and teaches an evaluation that
    detecting one counts as detecting the attack.
    """
    if "label_executed" not in ctx.columns:
        return []
    _, _, groups = _scan(ctx)
    payload_attacks = set((ctx.taxonomy or {}).get("requires_payload") or [])
    warn = ctx.threshold("unexecuted_share_warn", 0.25)
    critical = ctx.threshold("unexecuted_share_critical", 0.90)
    min_flows = ctx.threshold("unexecuted_min_flows", 20)
    findings = []

    for (cls, raw), counts in sorted(groups.items(),
                                     key=lambda kv: (kv[0][1] or "", kv[0][0])):
        if raw not in payload_attacks:
            continue
        n = counts[IDX["flows"]]
        yes, no = counts[IDX["exec_true"]], counts[IDX["exec_false"]]
        # NULL is "Zeek could not tell", not "did not execute", and must not be
        # counted as either -- so the share is over the flows it could tell about.
        undetermined, decided = counts[IDX["exec_null"]], yes + no
        if decided < min_flows or no / decided <= warn:
            continue
        share = no / decided
        findings.append(Finding(
            check_id="beh.executed_ratio", dataset=ctx.dataset,
            severity=Severity.CRITICAL if share >= critical else Severity.MAJOR,
            title=f"{raw}: {share:.1%} of flows never sent a byte "
                  f"({no:,} of {decided:,})",
            detail=f"{raw} cannot execute without the attacker sending payload, so these "
                   f"flows are connection attempts labelled as the attack itself. They "
                   f"inflate {cls} and, because they are trivially separable from real "
                   f"attack traffic, they inflate any accuracy figure measured on it.",
            metrics={"label_class": cls, "label_raw": raw, "flows": n,
                     "executed": yes, "unexecuted": no, "undetermined": undetermined,
                     "unexecuted_share": round(share, 6),
                     "warn_at": warn, "critical_at": critical},
            evidence=[{"label_executed": "true", "flows": yes},
                      {"label_executed": "false", "flows": no},
                      {"label_executed": "null (byte count unknown)",
                       "flows": undetermined}],
            scope={"label_class": cls, "label_raw": raw},
            repro=f"SELECT label_executed, count(*) AS flows\n"
                  f"FROM read_parquet('{ctx.q('labelled')}', union_by_name=true)\n"
                  f"WHERE label_raw = {_lit(raw)}\n"
                  f"GROUP BY 1 ORDER BY 2 DESC",
        ))
    return findings


@check(id="beh.port_anomaly", tier=2,
       title="each attack's modal destination port matches its class",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def port_anomaly(ctx):
    """Which service the labelled traffic actually reaches.

    An attack window is a time range and a pair of endpoints; nothing in it
    constrains the port. When an attack's traffic concentrates on a port its
    class cannot plausibly target -- a web attack whose modal port is 21 -- the
    window is over the wrong traffic. The published finding that 2018's
    DoS-SlowHTTPTest ran against port 21 rather than a web port is exactly this
    shape, and our own extraction puts 105,550 port-21 flows inside the 2018 dos
    class.

    A diffuse port distribution is not a finding. A scan has no single target to
    contradict, so a class whose modal port is below `min_modal_share` is passed
    over rather than judged on noise.
    """
    if "id.resp_p" not in ctx.columns:
        return []
    doc = _doc()
    query = f"""
    SELECT label_class, label_raw, resp_p, flows, total FROM (
      SELECT label_class, label_raw, "id.resp_p" AS resp_p, count(*) AS flows,
             sum(count(*)) OVER (PARTITION BY label_class, label_raw) AS total,
             row_number() OVER (PARTITION BY label_class, label_raw
                                ORDER BY count(*) DESC, "id.resp_p") AS rnk
      FROM read_parquet('{ctx.q('labelled')}', union_by_name=true)
      WHERE label_raw IS NOT NULL
      GROUP BY 1, 2, 3
    ) WHERE rnk <= 3
    ORDER BY label_class, label_raw, flows DESC"""
    ranked = {}
    for cls, raw, port, flows, total in ctx.lake.sql(query):
        flows, total = int(flows), int(total or 0)
        ranked.setdefault((cls, raw), []).append(
            {"resp_p": int(port) if port is not None else None, "flows": flows,
             "share": round(flows / total, 6) if total else 0.0})

    findings = []
    for (cls, raw), ports in sorted(ranked.items()):
        band = _band(doc, cls, raw) or {}
        expected = band.get("ports")
        if not expected:
            continue
        modal = ports[0]
        floor = band.get("min_modal_share", _defaults(doc, "min_modal_share", 0.30))
        if modal["share"] < floor or modal["resp_p"] in expected:
            continue
        share = modal["share"]
        findings.append(Finding(
            check_id="beh.port_anomaly", dataset=ctx.dataset,
            severity=(Severity.CRITICAL if share >= 0.90
                      else Severity.MAJOR if share >= 0.60 else Severity.MINOR),
            title=f"{raw}: modal destination port is {modal['resp_p']} "
                  f"({share:.1%} of flows), not {'/'.join(str(p) for p in expected)}",
            detail=f"The window labelled {raw} sits on a service its class cannot "
                   f"plausibly be attacking. Either the schedule's endpoints or times "
                   f"are wrong for this attack, or the attack is misnamed in the "
                   f"manifest -- in both cases the flows under this label are not the "
                   f"attack the label describes.",
            metrics={"label_class": cls, "label_raw": raw,
                     "modal_port": modal["resp_p"], "modal_share": share,
                     "modal_flows": modal["flows"],
                     "expected_ports": list(expected),
                     "min_modal_share": floor},
            evidence=ports, scope={"label_class": cls, "label_raw": raw},
            repro=f'SELECT "id.resp_p", count(*) AS flows\n'
                  f"FROM read_parquet('{ctx.q('labelled')}', union_by_name=true)\n"
                  f"WHERE label_raw = {_lit(raw)}\n"
                  f"GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
        ))
    return findings
