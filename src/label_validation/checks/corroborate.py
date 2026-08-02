#!/usr/bin/env python3
"""Tier 3 -- independent evidence: what Zeek's other logs say about the labels.

Every label in this project comes from a published attack schedule -- a time
window and a pair of addresses, transcribed from a web page. Tiers 0-2 can only
ask whether that transcription is internally consistent and whether traffic
inside a window looks different from traffic outside it. Neither can say that
the attack named in the schedule is the attack in the capture, because both
reason from the schedule itself.

This tier can, because Zeek's application-layer logs were produced by parsing
packets and know nothing about the schedule. If a window labelled
WebAttack-SQLInjection holds http records whose URIs carry UNION SELECT, the
label is corroborated by a source that could not have copied it. If a window
labelled FTP-BruteForce holds no ftp.log records at all -- because every
connection was empty and Zeek never instantiated an FTP analyser -- then the
schedule says an attack ran and the packets say nothing happened. That case is
not hypothetical: 2018's FTP-BruteForce covers 97 minutes and 192,300 flows,
every one of them empty, and the 2018 lake has no ftp.log whatsoever.

Four checks remain, each asking a specific question of a specific log. Three
were removed and it is worth saying why, because their absence is a judgement
rather than an oversight:

  cor.corroboration_rate  measured the share of a class's flows appearing in ANY
      application log. Its own benign control refuted it: benign corroborated at
      92.5%, higher than dos. It was measuring whether a flow speaks a parsed
      protocol, which benign traffic does by default -- a coverage statistic
      wearing the name of a correctness one.
  cor.notice_agreement    joined notice.log, which does not exist for any dataset
      in this lake and will not until extraction is re-run with a site policy.
      Code kept alive for a hypothetical input is code nobody can review.
  cor.weird_concentration compared weird.log rates inside and outside windows.
      It ran and found nothing, on a signal too diffuse to act on.

The survivors share a property the removed three lacked: each names a concrete
artefact that either exists in a log or does not, and its absence means
something specific about the label.

Most emit an INFO finding even when clean, because the numbers ARE the output --
a reader who cannot see them cannot distinguish a corroborated label from an
unexamined one. cor.analyzer_presence is silent when clean: its evidence is
purely negative, so there is nothing to say when it is absent.
"""

from src.label_validation.core.finding import Finding, Severity
from src.label_validation.core.registry import check

# Cheap facts about the lake -- does this log exist, what does its union schema
# contain -- cached per dataset. The runner builds one Context per process and
# every check below asks the same questions of the same handful of logs.
_CACHE = {}

# Independent evidence channels used by the surviving checks. conn is
# deliberately absent: it is the log the labels were attached to, so it cannot
# corroborate them.
CHANNELS = ("http", "ssl", "dns", "ssh", "ftp")

# Attacks whose execution REQUIRES a protocol Zeek parses, and the log(s) that
# prove the parser saw it. Evidence in any one of the listed logs satisfies the
# expectation, so an attack that could have run over TLS is not condemned for
# leaving nothing in http.log.
#
# Deliberately NOT listed, because absence there is expected rather than
# suspicious:
#   PortScan, Infiltration-PortScan  -- an empty connection IS the mechanism
#   DoS-Slowloris, DoS-Slowhttptest, DoS-SlowHTTPTest -- these hold sockets open
#       with deliberately incomplete requests; Zeek logs an http record only
#       once a request completes, so an empty http.log is the attack working
#   DDoS-LOIC-UDP, SYN, UDP, UDP-Lag, TFTP -- no application layer to parse
#   NTP, SNMP, SSDP, LDAP, NetBIOS, MSSQL, PortMap -- Zeek does parse several of
#       these, but the runner does not probe ntp/snmp/ldap logs, so the 2019
#       vectors are confirmed by service port instead (cor.reflection_service)
#   Infiltration -- the payload is a Dropbox download and a Meterpreter callback
#       on port 444; neither implies a specific Zeek log
PROTOCOL_EXPECTED = {
    "FTP-Patator": (("ftp",), "credential guessing over FTP"),
    "FTP-BruteForce": (("ftp",), "credential guessing over FTP"),
    "SSH-Patator": (("ssh",), "credential guessing over SSH"),
    "SSH-Bruteforce": (("ssh",), "credential guessing over SSH"),
    "WebAttack-BruteForce": (("http",), "login form submitted over HTTP"),
    "WebAttack-XSS": (("http",), "payload delivered in an HTTP request"),
    "WebAttack-SQLInjection": (("http",), "payload delivered in an HTTP request"),
    "BruteForce-Web": (("http",), "login form submitted over HTTP"),
    "BruteForce-XSS": (("http",), "payload delivered in an HTTP request"),
    "SQL-Injection": (("http",), "payload delivered in an HTTP request"),
    "DoS-Hulk": (("http",), "HTTP GET flood"),
    "DoS-GoldenEye": (("http",), "HTTP keep-alive flood"),
    "DDoS-LOIC-HTTP": (("http",), "LOIC HTTP flood"),
    "DDoS-HOIC": (("http",), "HOIC HTTP flood"),
    "DDoS-LOIT": (("http",), "LOIC HTTP flood"),
    "WebDDoS": (("http",), "HTTP flood"),
    "Heartbleed": (("ssl",), "TLS heartbeat abuse against an OpenSSL server"),
    "Botnet-ARES": (("http", "ssl"), "ARES C2 beacons over HTTP(S)"),
    "Bot": (("http", "ssl"), "Zeus/ARES C2 beacons over HTTP(S)"),
    "DNS": (("dns",), "DNS reflection/amplification"),
}

# 2019's raw attack names name a protocol. Each is checked against the service
# port seen on the flow -- not conn.service, which is not universal across the
# 2019 files. `amplifying` marks the reflection vectors, where far more bytes
# must reach the victim than leave it; SYN/UDP/UDP-Lag/WebDDoS are direct
# floods, where no amplification is expected and none is required.
REFLECTION_VECTORS = {
    "DNS":     (53, "udp", True),
    "NTP":     (123, "udp", True),
    "SNMP":    (161, "udp", True),
    "SSDP":    (1900, "udp", True),
    "LDAP":    (389, "udp", True),
    "NetBIOS": (137, "udp", True),
    "MSSQL":   (1434, "udp", True),
    "PortMap": (111, "udp", True),
    "TFTP":    (69, "udp", True),
    "SYN":     (None, "tcp", False),
    "UDP":     (None, "udp", False),
    "UDP-Lag": (None, "udp", False),
    "WebDDoS": (80, "tcp", False),
}

# conn columns cor.reflection_service reads back out of the labelled zone.
LAB_CONN_COLS = (("proto", "VARCHAR"), ("id.orig_p", "BIGINT"),
                 ("id.resp_p", "BIGINT"), ("id.orig_h", "VARCHAR"),
                 ("id.resp_h", "VARCHAR"), ("orig_ip_bytes", "BIGINT"),
                 ("resp_ip_bytes", "BIGINT"), ("conn_state", "VARCHAR"))

# SQL-injection tells. Each is a token with no ordinary meaning in a URI, so a
# match is evidence rather than coincidence -- the point is to keep the false
# positive rate low enough that a single match is worth reading. Applied to
# lower(uri), which also folds the %-encoded forms (%3C, %27) that CIC's scripts
# submit through GET parameters.
SQLI_SIGNATURES = [
    r"union[^a-z0-9]{1,12}select",                       # UNION SELECT, any spacing
    r"select[^;]{1,80}from",
    r"(?:'|%27)[^a-z0-9]{0,8}(?:or|and)[^a-z0-9]{1,8}(?:'|%27)?[0-9]{1,3}[^a-z0-9]{0,4}=",
    r"\bor[^a-z0-9]{1,4}1[^a-z0-9]{0,4}=[^a-z0-9]{0,4}1",  # ' or 1=1
    r"information_schema|table_schema",
    r"(?:sleep|benchmark|waitfor[^a-z]{1,6}delay)[^a-z]{0,4}\(",   # blind/time-based
    r"group_concat|concat_ws|xp_cmdshell|load_file",
]

# Cross-site scripting tells, same principle: script tags, event handlers and
# the three functions every proof-of-concept payload calls.
XSS_SIGNATURES = [
    r"<script|%3cscript|&lt;script",
    r"</script|%3c%2fscript",
    r"javascript[:%]",
    r"on(?:error|load|mouseover|focus|click)[^a-z]{0,6}=",
    r"(?:alert|prompt|confirm)[^a-z]{0,4}\(",
    r"document[^a-z]{0,4}\.[^a-z]{0,4}cookie",
    r"<(?:img|svg|iframe|body)[^a-z]|%3c(?:img|svg|iframe|body)",
]


# ------------------------------------------------------------------ plumbing

def _lit(v):
    return "'" + str(v).replace("'", "''") + "'"


def _list_lit(vals):
    """A VARCHAR[] literal. NULL needs its type: an untyped NULL in a VALUES
    column that is NULL for every row binds as INTEGER and breaks list_contains."""
    return ("CAST(NULL AS VARCHAR[])" if not vals
            else "[" + ", ".join(_lit(v) for v in vals) + "]")


def _log_present(ctx, log):
    """Does this log actually exist in the lake?

    ctx.sources carries a glob for every probed log type whether or not any
    files matched it, so a check that reads one without asking first dies with a
    file-not-found instead of reporting the absence -- and the absence is the
    finding. A glob() is a listing, not a scan.
    """
    key = (ctx.dataset, "present", log)
    if key not in _CACHE:
        glob = ctx.q(log)
        try:
            row = ctx.lake.one(f"SELECT count(*) FROM glob({_lit(glob)})") if glob else None
            _CACHE[key] = bool(row and row[0])
        except Exception:                       # noqa: BLE001 -- absence, not failure
            _CACHE[key] = False
    return _CACHE[key]


def _log_columns(ctx, log):
    """Union schema of a log, or {} if it has none.

    Zeek's schema drifts inside a single dataset -- 2019's dns.log loses query,
    qtype_name and rcode in most files; ssh.auth_success is missing from 2018 and
    2019 -- so every column reference below is guarded by this. Read from parquet
    footers, once per log per run.
    """
    key = (ctx.dataset, "columns", log)
    if key not in _CACHE:
        if not _log_present(ctx, log):
            _CACHE[key] = {}
            return _CACHE[key]
        try:
            _CACHE[key] = ctx.lake.columns(ctx.q(log))
        except Exception:                       # noqa: BLE001
            # A read that failed is not a log with no columns. Caching {} here
            # would make one transient S3 error look like a permanent schema
            # fact for the rest of the run, and every later caller would build
            # SQL against an empty projection. Left uncached so the next caller
            # retries, and returned empty so callers can guard.
            return {}
    return _CACHE[key]


def _ref(cols, name, alias="e", sqltype="VARCHAR"):
    """A column reference, or a typed NULL when the log does not carry it."""
    return f'{alias}."{name}"' if name in cols else f"CAST(NULL AS {sqltype})"


def _projected(cols, wanted):
    """A SELECT list that keeps its shape when a column is missing everywhere.

    A column absent from some files is NULL-filled by union_by_name; a column
    absent from all of them is a parse error, which would take out the whole
    check rather than the one number that depends on it.
    """
    return ", ".join(f'"{c}"' if c in cols else f'CAST(NULL AS {t}) AS "{c}"'
                     for c, t in wanted)


def _windows(rules):
    """Attack windows merged into disjoint intervals.

    Overlapping rules would otherwise have their seconds counted twice, which
    matters wherever a rate is divided by a duration.
    """
    out = []
    for t0, t1 in sorted((r["t0"], r["t1"]) for r in rules):
        if out and t0 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], t1)
        else:
            out.append([t0, t1])
    return [(a, b) for a, b in out]


def _span(rules):
    return (min(r["t0"] for r in rules), max(r["t1"] for r in rules))


def _rules_cte(rules, name="rules"):
    """The rules under test as an inline table.

    `eps` is attackers and victims in one list because corroboration only asks
    whether a record involves the attack at all. That is deliberately looser than
    label_flows' MATCH, which requires the two ends to be attacker and victim
    respectively: absence of evidence must never be an artefact of testing more
    strictly than the rule that produced the label.
    """
    rows = ",\n        ".join(
        f"({r['rid']}, {_lit(r['name'])}, {_lit(r['canonical'])}, {r['t0']}, {r['t1']}, "
        f"{_list_lit((r['attackers'] or []) + (r['victims'] or []))}, "
        f"{'CAST(NULL AS VARCHAR)' if not r['prefix'] else _lit(r['prefix'])})"
        for r in rules)
    return (f"{name} AS (SELECT * FROM (VALUES\n        {rows}\n      ) "
            f"AS t(rid, rname, rclass, t0, t1, eps, vpfx))")


def _join_on(cols, alias="e", rel="r"):
    t = f"{alias}.ts >= {rel}.t0 AND {alias}.ts < {rel}.t1"
    if "id.orig_h" not in cols or "id.resp_h" not in cols:
        return t          # a log without the 5-tuple can only be matched on time
    return t + f"""
          AND (coalesce(list_contains({rel}.eps, {alias}."id.orig_h"), false)
            OR coalesce(list_contains({rel}.eps, {alias}."id.resp_h"), false)
            OR coalesce(starts_with({alias}."id.orig_h", {rel}.vpfx), false)
            OR coalesce(starts_with({alias}."id.resp_h", {rel}.vpfx), false)
            OR ({rel}.eps IS NULL AND {rel}.vpfx IS NULL))"""


def _read(ctx, log, rules, extra=()):
    """A projected, span-bounded read of one Zeek log.

    Only the columns a check names are read, and the scan is bounded by the first
    and last window under test so DuckDB can drop whole row groups on the parquet
    ts statistics. Both matter: 2018's dns.log alone is 30.5M rows and its ssl.log
    19.2M.
    """
    cols = _log_columns(ctx, log)
    keep = [c for c in ("ts", "uid", "id.orig_h", "id.resp_h") if c in cols]
    keep += [c for c in extra if c in cols and c not in keep]
    t0, t1 = _span(rules)
    sel = ", ".join(f'"{c}"' for c in keep)
    return (f"SELECT {sel} FROM read_parquet({_lit(ctx.q(log))}, union_by_name=true)\n"
            f"        WHERE ts IS NOT NULL AND ts >= {t0} AND ts < {t1}")


def _records_in_windows(ctx, log, rules):
    """Records in `log` inside each rule's window and involving its endpoints.

    One scan per log covering every rule at once, rather than one scan per rule.
    Returns ({rid: (records, ts_min, ts_max)}, query); a rule missing from the
    dict matched nothing, which is the whole point of the exercise.
    """
    # An empty column set means the log is absent, or that reading its schema
    # failed. Either way there is nothing to project, and building the query
    # anyway yields `SELECT  FROM ...` -- a parser error that reads like a code
    # bug rather than the missing input it actually is.
    if not rules or not _log_present(ctx, log) or not _log_columns(ctx, log):
        return {}, ""
    q = f"""
      WITH {_rules_cte(rules)},
      ev AS ({_read(ctx, log, rules)})
      SELECT r.rid, count(*) AS records, min(e.ts) AS lo, max(e.ts) AS hi
      FROM rules r JOIN ev e ON {_join_on(_log_columns(ctx, log))}
      GROUP BY r.rid ORDER BY r.rid"""
    return {int(r[0]): (int(r[1]), r[2], r[3]) for r in ctx.lake.sql(q)}, q.strip()


def _flows_by_rule(ctx):
    """rule_id -> labelled flows. The denominator for everything in this tier."""
    key = (ctx.dataset, "flows_by_rule")
    if key not in _CACHE:
        if "rule_id" not in ctx.columns:
            _CACHE[key] = {}
        else:
            rows = ctx.lake.sql(
                f"SELECT rule_id, count(*) FROM read_parquet({_lit(ctx.q('labelled'))}, "
                f"union_by_name=true) WHERE rule_id IS NOT NULL GROUP BY 1")
            _CACHE[key] = {int(r[0]): int(r[1]) for r in rows}
    return _CACHE[key]


def _share(n, d):
    return round(n / d, 6) if d else None


def _alt(patterns):
    """Signature list -> one RE2 alternation, each branch grouped so a `|` inside
    a pattern cannot swallow the branch beside it."""
    return "|".join(f"(?:{p})" for p in patterns)


# -------------------------------------------------------------------- checks

@check(id="cor.analyzer_presence", tier=3,
       title="protocol-specific attacks left records in that protocol's log",
       needs=("labelled",), max_severity=Severity.CRITICAL)
def analyzer_presence(ctx):
    """Zeek instantiates a protocol analyser only when it sees that protocol.

    So the absence of an application log across an attack window is a statement
    made by the packets: whatever ran there, it was not the protocol the schedule
    names. 2018 is the worked example -- FTP-BruteForce runs 97 minutes and
    carries 192,300 flows, and the 2018 lake has no ftp.log at all, because every
    one of those connections was empty and no FTP analyser was ever created.

    A failure here means the label names an attack the traffic cannot support.
    Either the window or the endpoints are wrong, or the attack did not execute
    and the flows are connection attempts wearing an attack's name.

    `needs` is only ("labelled",) on purpose. Declaring the protocol logs would
    make the runner skip this check exactly when the log is missing -- which is
    the one case it exists to report.
    """
    expected = [(r, PROTOCOL_EXPECTED[r["name"]]) for r in ctx.rules
                if r["name"] in PROTOCOL_EXPECTED]
    if not expected:
        return []
    flows = _flows_by_rule(ctx)
    min_flows = ctx.threshold("analyzer_presence_min_flows", 100)
    # Not in policy.yaml: a log that exists but holds a record for fewer than
    # this fraction of the window's flows is thin rather than absent, so it is
    # reported one severity down.
    thin = ctx.threshold("analyzer_records_per_flow_min", 0.01)

    counts, queries = {}, {}
    for log in sorted({lg for _, (logs, _) in expected for lg in logs}):
        rs = [r for r, (logs, _) in expected if log in logs]
        got, q = _records_in_windows(ctx, log, rs)
        for rid, seen in got.items():
            counts[(rid, log)] = seen[0]
        if q:
            queries[log] = q

    absent, empty, sparse = [], [], []
    for r, (logs, why) in expected:
        n = flows.get(r["rid"], 0)
        if n < min_flows:
            continue                      # too few flows for absence to prove much
        live = [lg for lg in logs if _log_present(ctx, lg)]
        records = sum(counts.get((r["rid"], lg), 0) for lg in live)
        row = {"rule_id": r["rid"], "attack": r["name"], "class": r["canonical"],
               "date": r["date"], "window": f"{r['start']}-{r['end']}",
               "flows": n, "expected_log": "/".join(logs), "basis": why,
               "records": records,
               "records_per_flow": _share(records, n)}
        if not live:
            absent.append(row)
        elif records == 0:
            empty.append(row)
        elif thin and records < thin * n:
            sparse.append(row)

    findings = []
    if absent:
        logs = sorted({row["expected_log"] for row in absent})
        findings.append(Finding(
            check_id="cor.analyzer_presence", dataset=ctx.dataset,
            severity=Severity.CRITICAL,
            title=f"{len(absent)} attack window(s) require a log the lake does not "
                  f"have ({', '.join(logs)})",
            detail="Zeek writes an application log only once it has parsed that "
                   "application. These attacks are named for a protocol whose log was "
                   "never produced anywhere in the dataset, so the traffic the schedule "
                   "calls an attack contained none of the protocol the attack is made "
                   "of. The flows are connection attempts, or the window is wrong.",
            metrics={"rules": len(absent),
                     "flows": sum(row["flows"] for row in absent),
                     "missing_logs": ", ".join(logs)},
            evidence=absent, scope={"logs": logs},
            repro=f"SELECT count(*) FROM glob('{ctx.q(logs[0].split('/')[0])}')",
        ))
    if empty:
        findings.append(Finding(
            check_id="cor.analyzer_presence", dataset=ctx.dataset,
            severity=Severity.CRITICAL,
            title=f"{len(empty)} attack window(s) contain no records in the log "
                  f"their protocol requires",
            detail="The log exists for this dataset but holds nothing inside the "
                   "window, for either endpoint of the attack. Matching here is looser "
                   "than the labelling rule -- any record touching either address "
                   "counts -- so this is not a near miss on the endpoint test.",
            metrics={"rules": len(empty), "flows": sum(row["flows"] for row in empty)},
            evidence=empty,
            repro="\n\n".join(queries.values())[:4000],
        ))
    if sparse:
        findings.append(Finding(
            check_id="cor.analyzer_presence", dataset=ctx.dataset,
            severity=Severity.MAJOR,
            title=f"{len(sparse)} attack window(s) are thin in their protocol's log",
            detail=f"Fewer than one record per {int(1 / thin)} labelled flows. The "
                   "protocol is present, so the label is not baseless, but most of the "
                   "flows carrying it produced nothing an analyser could read -- the "
                   "window is probably wider than the attack.",
            metrics={"rules": len(sparse), "records_per_flow_min": thin,
                     "flows": sum(row["flows"] for row in sparse)},
            evidence=sparse,
            repro="\n\n".join(queries.values())[:4000],
        ))
    return findings


@check(id="cor.http_payload", tier=3,
       title="web_attack windows carry SQLi/XSS payloads in http URIs",
       needs=("labelled", "http"), max_severity=Severity.MAJOR)
def http_payload(ctx):
    """The only place a web attack's *content* can be read back.

    A web_attack label says an injection was attempted. http.log records the URI
    Zeek parsed off the wire, so for GET-borne payloads the claim is directly
    testable: either the requests in that window carry injection syntax or they
    do not. 2017's WebAttack-SQLInjection is 12 flows -- this check is the
    difference between a class with twelve real examples and a class with twelve
    mislabelled ones.

    A failure means the window holds ordinary web traffic. The one honest
    caveat, reported rather than hidden: Zeek does not log request bodies, so a
    POST login brute force can be perfectly real and still leave no signature in
    any URI. Rules whose records are predominantly POST are reported as
    unobservable, not as contradicted.
    """
    web = [r for r in ctx.rules if r["canonical"] == "web_attack"]
    if not web:
        return []
    cols = _log_columns(ctx, "http")
    if "uri" not in cols:
        return [Finding(
            check_id="cor.http_payload", dataset=ctx.dataset, severity=Severity.MAJOR,
            title="http.log carries no uri column",
            detail="Without uri there is no payload to read, so web_attack labels "
                   "cannot be corroborated from this dataset's http log.",
            metrics={"http_columns": len(cols)},
        )]
    flows = _flows_by_rule(ctx)
    uri = "lower(coalesce(e.\"uri\", ''))"
    sqli = f"regexp_matches({uri}, {_lit(_alt(SQLI_SIGNATURES))})"
    xss = f"regexp_matches({uri}, {_lit(_alt(XSS_SIGNATURES))})"
    status = _ref(cols, "status_code", sqltype="BIGINT")
    method = f"upper(coalesce({_ref(cols, 'method')}, ''))"
    uid = _ref(cols, "uid")
    q = f"""
      WITH {_rules_cte(web)},
      ev AS ({_read(ctx, 'http', web, extra=('uri', 'method', 'status_code'))})
      SELECT r.rid, r.rname, count(*) AS records,
             count(DISTINCT {uid}) AS uids,
             sum(CASE WHEN {sqli} THEN 1 ELSE 0 END) AS sqli,
             sum(CASE WHEN {xss} THEN 1 ELSE 0 END) AS xss,
             count(DISTINCT CASE WHEN {sqli} OR {xss} THEN {uid} END) AS uids_sig,
             sum(CASE WHEN {method} = 'GET' THEN 1 ELSE 0 END) AS gets,
             sum(CASE WHEN {method} = 'POST' THEN 1 ELSE 0 END) AS posts,
             sum(CASE WHEN ({sqli} OR {xss}) AND {status} BETWEEN 200 AND 299
                      THEN 1 ELSE 0 END) AS sig_2xx
      FROM rules r JOIN ev e ON {_join_on(cols)}
      GROUP BY r.rid, r.rname ORDER BY r.rid"""
    seen = {int(row[0]): row for row in ctx.lake.sql(q)}

    rows, contradicted, unobservable = [], [], []
    for r in web:
        got = seen.get(r["rid"])
        n = flows.get(r["rid"], 0)
        rec, uids, n_sqli, n_xss, uids_sig, gets, posts, sig_2xx = (
            (int(got[2]), int(got[3]), int(got[4]), int(got[5]), int(got[6]),
             int(got[7]), int(got[8]), int(got[9])) if got else (0,) * 8)
        row = {"rule_id": r["rid"], "attack": r["name"], "date": r["date"],
               "window": f"{r['start']}-{r['end']}", "flows": n,
               "http_records": rec, "http_flows": uids,
               "sqli_records": n_sqli, "xss_records": n_xss,
               "flows_with_signature": uids_sig,
               "signature_share_of_flows": _share(uids_sig, n),
               "get_records": gets, "post_records": posts,
               "signature_hits_2xx": sig_2xx}
        rows.append(row)
        if uids_sig:
            continue
        # No signature. Distinguish "we could have seen it and did not" from
        # "the payload was in a body Zeek never logs".
        (unobservable if posts > gets else contradicted).append(row)

    samples = []
    if any(row["flows_with_signature"] for row in rows):
        samples = ctx.lake.sql(f"""
          WITH {_rules_cte(web)},
          ev AS ({_read(ctx, 'http', web, extra=('uri', 'method', 'status_code'))})
          SELECT r.rname, {method} AS method, {status} AS status_code,
                 substr(e."uri", 1, 120) AS uri, count(*) AS n
          FROM rules r JOIN ev e ON {_join_on(cols)}
          WHERE {sqli} OR {xss}
          GROUP BY 1, 2, 3, 4 ORDER BY n DESC LIMIT 20""")

    total_sig = sum(row["flows_with_signature"] for row in rows)
    findings = [Finding(
        check_id="cor.http_payload", dataset=ctx.dataset, severity=Severity.INFO,
        title=f"{total_sig:,} web_attack flow(s) carry an injection signature in "
              f"their URI across {len(web)} window(s)",
        detail="Independent confirmation of the web_attack class: these URIs were "
               "parsed off the wire by Zeek and match SQLi or XSS syntax that has no "
               "ordinary meaning in a web request.",
        metrics={"rules": len(web),
                 "flows": sum(row["flows"] for row in rows),
                 "flows_with_signature": total_sig,
                 "sqli_records": sum(row["sqli_records"] for row in rows),
                 "xss_records": sum(row["xss_records"] for row in rows),
                 "corroborated_share": _share(
                     total_sig, sum(row["flows"] for row in rows))},
        evidence=rows + [{"sample_uri": s[3], "attack": s[0], "method": s[1],
                          "status_code": s[2], "records": s[4]} for s in samples],
        repro=q.strip(),
    )]
    if contradicted:
        findings.append(Finding(
            check_id="cor.http_payload", dataset=ctx.dataset, severity=Severity.MAJOR,
            title=f"{len(contradicted)} web_attack window(s) contain no injection "
                  f"payload in any URI",
            detail="The window's HTTP traffic is mostly GET, so a URI-borne payload "
                   "would have been logged, and none of it matches any SQLi or XSS "
                   "signature. These flows are labelled as a web attack on the strength "
                   "of the schedule alone, and the requests Zeek parsed do not support "
                   "it.",
            metrics={"rules": len(contradicted),
                     "flows": sum(row["flows"] for row in contradicted)},
            evidence=contradicted, repro=q.strip(),
        ))
    if unobservable:
        findings.append(Finding(
            check_id="cor.http_payload", dataset=ctx.dataset, severity=Severity.INFO,
            title=f"{len(unobservable)} web_attack window(s) cannot be corroborated "
                  f"from URIs (POST-dominated)",
            detail="These windows are mostly POST requests, whose bodies Zeek does not "
                   "log. Absence of a URI signature is therefore not evidence against "
                   "the label -- it is the absence of any evidence either way, and "
                   "those flows must not be counted as confirmed.",
            metrics={"rules": len(unobservable),
                     "flows": sum(row["flows"] for row in unobservable)},
            evidence=unobservable, repro=q.strip(),
        ))
    return findings


@check(id="cor.auth_evidence", tier=3,
       title="brute_force windows carry authentication attempts",
       needs=("labelled", "ssh"), max_severity=Severity.MAJOR)
def auth_evidence(ctx):
    """A brute force is a stream of failed logins, and Zeek counts them.

    ssh.log carries auth_attempts per connection (and auth_success where the
    files have it); ftp.log carries the user, the password and the reply code,
    where 530 is a rejected login. Either one turns "the schedule says credential
    guessing" into a number taken off the wire.

    A failure means a brute_force window in which nobody tried to authenticate.
    ftp.log is used when the dataset has one -- only 2017 does -- so `needs`
    declares ssh alone; the total absence of an expected auth log is
    cor.analyzer_presence's finding, not this one's.
    """
    bf = [r for r in ctx.rules if r["canonical"] == "brute_force"]
    if not bf:
        return []
    flows = _flows_by_rule(ctx)
    rows, queries, silent = [], {}, []

    ssh_rules = [r for r in bf if r["name"].lower().startswith("ssh")]
    ssh = {}
    if ssh_rules and _log_present(ctx, "ssh"):
        cols = _log_columns(ctx, "ssh")
        attempts = _ref(cols, "auth_attempts", sqltype="BIGINT")
        success = (f'sum(CASE WHEN e."auth_success" THEN 1 ELSE 0 END)'
                   if "auth_success" in cols else "CAST(NULL AS BIGINT)")
        queries["ssh"] = f"""
          WITH {_rules_cte(ssh_rules)},
          ev AS ({_read(ctx, 'ssh', ssh_rules,
                        extra=('auth_attempts', 'auth_success', 'client'))})
          SELECT r.rid, count(*) AS records,
                 count(DISTINCT {_ref(cols, 'uid')}) AS uids,
                 sum(coalesce({attempts}, 0)) AS attempts,
                 max({attempts}) AS max_attempts, {success} AS successes,
                 count(DISTINCT e."id.orig_h") AS sources
          FROM rules r JOIN ev e ON {_join_on(cols)}
          GROUP BY r.rid ORDER BY r.rid"""
        ssh = {int(x[0]): x for x in ctx.lake.sql(queries["ssh"])}

    ftp_rules = [r for r in bf if r["name"].lower().startswith("ftp")]
    ftp = {}
    if ftp_rules and _log_present(ctx, "ftp"):
        cols = _log_columns(ctx, "ftp")
        code = _ref(cols, "reply_code", sqltype="BIGINT")
        user = _ref(cols, "user")
        queries["ftp"] = f"""
          WITH {_rules_cte(ftp_rules)},
          ev AS ({_read(ctx, 'ftp', ftp_rules,
                        extra=('user', 'password', 'command', 'reply_code'))})
          SELECT r.rid, count(*) AS records,
                 count(DISTINCT {_ref(cols, 'uid')}) AS uids,
                 count(DISTINCT {user}) AS users,
                 sum(CASE WHEN {code} = 530 THEN 1 ELSE 0 END) AS failures,
                 sum(CASE WHEN {code} = 230 THEN 1 ELSE 0 END) AS successes,
                 count(DISTINCT e."id.orig_h") AS sources
          FROM rules r JOIN ev e ON {_join_on(cols)}
          GROUP BY r.rid ORDER BY r.rid"""
        ftp = {int(x[0]): x for x in ctx.lake.sql(queries["ftp"])}

    for r in bf:
        n = flows.get(r["rid"], 0)
        row = {"rule_id": r["rid"], "attack": r["name"], "date": r["date"],
               "window": f"{r['start']}-{r['end']}", "flows": n}
        got = ssh.get(r["rid"])
        if r["rid"] in {x["rid"] for x in ssh_rules}:
            row["log"] = "ssh"
            row["log_present"] = _log_present(ctx, "ssh")
            row.update({"records": int(got[1]) if got else 0,
                        "flows_with_record": int(got[2]) if got else 0,
                        "auth_attempts": int(got[3]) if got and got[3] else 0,
                        "max_auth_attempts": int(got[4]) if got and got[4] else 0,
                        "auth_successes": int(got[5]) if got and got[5] is not None
                        else None,
                        "sources": int(got[6]) if got else 0})
        else:
            got = ftp.get(r["rid"])
            row["log"] = "ftp"
            row["log_present"] = _log_present(ctx, "ftp")
            row.update({"records": int(got[1]) if got else 0,
                        "flows_with_record": int(got[2]) if got else 0,
                        "distinct_users": int(got[3]) if got else 0,
                        "login_failures_530": int(got[4]) if got else 0,
                        "login_successes_230": int(got[5]) if got else 0,
                        "sources": int(got[6]) if got else 0})
        row["records_per_flow"] = _share(row["records"], n)
        rows.append(row)
        if row["log_present"] and n and not row["records"]:
            silent.append(row)

    findings = [Finding(
        check_id="cor.auth_evidence", dataset=ctx.dataset, severity=Severity.INFO,
        title=f"{sum(r['records'] for r in rows):,} authentication record(s) observed "
              f"across {len(bf)} brute_force window(s)",
        detail="Authentication activity Zeek parsed inside the windows the schedule "
               "calls credential guessing, set beside the number of flows those windows "
               "were labelled with. A window with far fewer records than flows is mostly "
               "connections that never reached authentication.",
        metrics={"rules": len(bf), "flows": sum(r["flows"] for r in rows),
                 "records": sum(r["records"] for r in rows),
                 "flows_with_record": sum(r["flows_with_record"] for r in rows),
                 "corroborated_share": _share(
                     sum(r["flows_with_record"] for r in rows),
                     sum(r["flows"] for r in rows))},
        evidence=rows, repro="\n\n".join(q.strip() for q in queries.values()),
    )]
    if silent:
        findings.append(Finding(
            check_id="cor.auth_evidence", dataset=ctx.dataset, severity=Severity.MAJOR,
            title=f"{len(silent)} brute_force window(s) contain no authentication "
                  f"records at all",
            detail="The auth log for this protocol exists and was read, and nothing in "
                   "the window touched either endpoint of the attack. Nobody tried to "
                   "log in, so whatever these flows are, they are not credential "
                   "guessing that got as far as an authentication exchange.",
            metrics={"rules": len(silent), "flows": sum(r["flows"] for r in silent)},
            evidence=silent,
            repro="\n\n".join(q.strip() for q in queries.values()),
        ))
    return findings


@check(id="cor.reflection_service", tier=3,
       title="2019 DDoS vectors are the protocol their name claims",
       datasets=("cic-ddos-2019",), needs=("labelled", "dns"),
       max_severity=Severity.MAJOR)
def reflection_service(ctx):
    """2019 names every window after a protocol. Do the packets agree?

    Each vector implies a service port on one end of the flow, a transport, and
    -- for the reflection vectors -- far more bytes arriving at the victim than
    it ever sent. All three come out of the labelled conn zone itself, so this is
    the one corroboration that survives the fact that Zeek wrote almost no
    application logs for the 2019 floods.

    A failure means the class `ddos` is carrying flows named for a protocol they
    do not speak: either the window is misaligned, or the raw name in the
    manifest is wrong. Both matter, because the raw names are the only reason to
    believe 2019 contains thirteen distinct attack behaviours rather than one.
    """
    vec = [(r, REFLECTION_VECTORS[r["name"]]) for r in ctx.rules
           if r["name"] in REFLECTION_VECTORS]
    if not vec or "rule_id" not in ctx.columns:
        return []
    port_min = ctx.threshold("reflection_port_match_min", 0.5)
    amp_min = ctx.threshold("amplification_ratio_min", 1.0)

    vals = ",\n        ".join(
        f"({r['rid']}, {_lit(r['name'])}, "
        f"{port if port else 'CAST(NULL AS BIGINT)'}, {_lit(proto)}, "
        f"{_list_lit(r['victims'] or [])})"
        for r, (port, proto, _) in vec)
    to_victim = ('CASE WHEN coalesce(list_contains(v.vic, l."id.resp_h"), false) '
                 "THEN coalesce(l.orig_ip_bytes, 0) ELSE coalesce(l.resp_ip_bytes, 0) END")
    from_victim = ('CASE WHEN coalesce(list_contains(v.vic, l."id.resp_h"), false) '
                   "THEN coalesce(l.resp_ip_bytes, 0) ELSE coalesce(l.orig_ip_bytes, 0) END")
    q = f"""
      WITH vec AS (SELECT * FROM (VALUES
        {vals}
      ) AS v(rid, rname, eport, eproto, vic)),
      lab AS (
        SELECT rule_id, {_projected(ctx.columns, LAB_CONN_COLS)}
        FROM read_parquet({_lit(ctx.q('labelled'))}, union_by_name=true)
        WHERE rule_id IS NOT NULL)
      SELECT v.rid, v.rname, count(*) AS flows,
             sum(CASE WHEN l.proto = v.eproto THEN 1 ELSE 0 END) AS proto_match,
             sum(CASE WHEN v.eport IS NOT NULL
                       AND (l."id.orig_p" = v.eport OR l."id.resp_p" = v.eport)
                      THEN 1 ELSE 0 END) AS port_match,
             sum(CASE WHEN l.conn_state = 'S0' THEN 1 ELSE 0 END) AS s0,
             sum({to_victim}) AS bytes_to_victim,
             sum({from_victim}) AS bytes_from_victim,
             sum(coalesce(l.orig_ip_bytes, 0)) AS orig_ip_bytes,
             sum(coalesce(l.resp_ip_bytes, 0)) AS resp_ip_bytes,
             count(DISTINCT l."id.orig_h") AS distinct_sources
      FROM lab l JOIN vec v ON l.rule_id = v.rid
      GROUP BY v.rid, v.rname ORDER BY flows DESC"""
    seen = {int(x[0]): x for x in ctx.lake.sql(q)}

    rows, wrong_proto, no_amp = [], [], []
    for r, (port, proto, amplifying) in vec:
        got = seen.get(r["rid"])
        if not got:
            continue                      # rule matched no flows: tier 1's finding
        (_, _, n, pm, portm, s0, to_v, from_v, ob, rb, srcs) = got
        n = int(n)
        amp = (to_v / from_v) if from_v else None
        row = {"rule_id": r["rid"], "vector": r["name"], "date": r["date"],
               "window": f"{r['start']}-{r['end']}", "flows": n,
               "expected_port": port, "expected_proto": proto,
               "proto_match_share": _share(int(pm), n),
               "port_match_share": _share(int(portm), n) if port else None,
               "s0_share": _share(int(s0), n),
               "bytes_to_victim": int(to_v), "bytes_from_victim": int(from_v),
               "amplification": round(amp, 3) if amp else None,
               "resp_over_orig_ip_bytes": (round(rb / ob, 3) if ob else None),
               "distinct_sources": int(srcs), "reflection": amplifying}
        rows.append(row)
        if port and row["port_match_share"] is not None \
                and row["port_match_share"] < port_min:
            wrong_proto.append(row)
        if amplifying and amp is not None and amp < amp_min:
            no_amp.append(row)

    # The DNS vector is the one 2019 attack with a matching application log, so
    # it gets the same treatment as every other protocol-specific attack.
    dns_rules = [r for r in ctx.rules if r["name"] == "DNS"]
    dns_counts, dns_q = _records_in_windows(ctx, "dns", dns_rules)
    for row in rows:
        if row["vector"] == "DNS":
            row["dns_log_records"] = dns_counts.get(row["rule_id"], (0,))[0]

    findings = [Finding(
        check_id="cor.reflection_service", dataset=ctx.dataset, severity=Severity.INFO,
        title=f"{len(rows)} DDoS vector(s) checked against the service they name",
        detail="Per vector: the share of its flows on the protocol's own port, the "
               "share with no reply at all (conn_state S0), and how many more bytes "
               "reached the victim than left it. Reflection needs a reflector, so a "
               "vector with no amplification and few sources is not the attack the "
               "manifest names.",
        metrics={"vectors": len(rows), "flows": sum(r["flows"] for r in rows),
                 "port_match_min": port_min, "amplification_ratio_min": amp_min,
                 "dns_log_records": sum(v[0] for v in dns_counts.values())},
        evidence=rows, repro=q.strip(),
    )]
    if wrong_proto:
        findings.append(Finding(
            check_id="cor.reflection_service", dataset=ctx.dataset,
            severity=Severity.MAJOR,
            title=f"{len(wrong_proto)} vector(s) mostly do not use the port their "
                  f"name implies",
            detail="Fewer than the required share of these flows have the named "
                   "service's port on either end. The window holds traffic, but not the "
                   "protocol the raw attack name claims -- so the per-vector names in "
                   "the manifest cannot be used to distinguish behaviours within the "
                   "ddos class.",
            metrics={"vectors": len(wrong_proto), "port_match_min": port_min,
                     "flows": sum(r["flows"] for r in wrong_proto)},
            evidence=wrong_proto, repro=q.strip(),
        ))
    if no_amp:
        findings.append(Finding(
            check_id="cor.reflection_service", dataset=ctx.dataset,
            severity=Severity.MAJOR,
            title=f"{len(no_amp)} reflection vector(s) show no amplification",
            detail="A reflection/amplification attack sends the victim more bytes than "
                   "it emits, by construction -- that is what the reflector is for. "
                   "These windows do not, so the flows are labelled after a mechanism "
                   "the traffic does not exhibit.",
            metrics={"vectors": len(no_amp), "amplification_ratio_min": amp_min,
                     "flows": sum(r["flows"] for r in no_amp)},
            evidence=no_amp, repro=q.strip(),
        ))
    return findings

