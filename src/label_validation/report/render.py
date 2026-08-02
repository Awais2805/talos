#!/usr/bin/env python3
"""Turn the validation JSON into standalone HTML: one page per dataset, plus an index.

A findings list that exists only as JSON gets read once, by the person who ran
it. This stage exists so the evidence survives that moment: the same page has to
be legible to a supervisor who will never run `make validate`, and it has to
open from a dissertation appendix on a machine with no network, no CDN and no
JavaScript -- which is why the findings expand with <details> rather than with a
click handler.

Everything visual is imported from the EDA renderer rather than copied. The two
report sets are one product; that only stays true if there is one stylesheet and
one chart idiom, so a colour changed in src/eda/render.py changes here too.

Rendering computes nothing. Every number on these pages was decided by a check
and written by runner.py -- if a figure looks wrong, the check is wrong, and
this file is not where to fix it.

Usage:
    python src/label_validation/report/render.py
    python src/label_validation/report/render.py --dir reports/validation
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "config.yml").is_file())
sys.path.insert(0, str(ROOT))

from src.eda.render import (CSS, bar, diverging_bars, e, fnum, hbar,   # noqa: E402
                            kpis, matrix_heat, page, pct, table)

VAL_DIR = ROOT / "reports" / "validation"

# Severity -> the EDA stylesheet's existing state classes. BLOCKER and CRITICAL
# share `bad` deliberately: the distinction between them is about consequence,
# not about whether you are allowed to ignore it.
SEV_CLASS = {"BLOCKER": "bad", "CRITICAL": "bad", "MAJOR": "warn",
             "MINOR": "mut", "INFO": "mut"}
SEV_ORDER = ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]

# Cosmetic only -- the authoritative tier numbering lives on the @check
# decorators in src/label_validation/checks/.
TIERS = {0: "T0 integrity", 1: "T1 schedule", 2: "T2 behaviour",
         3: "T3 corroboration", 4: "T4 noise", 5: "T5 probes",
         6: "T6 cross-dataset"}

# Evidence keys that mean "seconds from a scheduled boundary", in the order the
# sweep chart should prefer them. Emitted by the tier-1 window checks.
OFFSET_KEYS = {"offset_s": "declared start", "offset_after_end_s": "declared end"}

# Reports in this directory that are not per-dataset findings.
NOT_A_DATASET = ("index", "label_health")

# Planted in a cell, promoted to a class on the enclosing <tr> after the shared
# table helper has run. See the runlog below.
MARK = {"skip": "<!--did-not-run-->", "err": "<!--errored-->"}

# What a findings page needs beyond the EDA sheet: disclosure blocks, a repro
# box that wraps instead of scrolling off the page, and a runlog row for a check
# that did not run. Nothing here restyles anything an EDA page uses.
EXTRA_CSS = """
details{background:var(--card);border:1px solid var(--line);border-radius:8px;
        padding:.5rem .7rem;margin:.45rem 0}
details+details{margin-top:.45rem}
summary{cursor:pointer;font-weight:600;font-size:.9rem}
summary::marker{color:var(--mut)}
details>p{color:var(--mut);font-size:.85rem;margin:.6rem 0}
pre.mono{white-space:pre-wrap;word-break:break-word;background:var(--bg);
         border:1px solid var(--line);border-radius:6px;padding:.5rem .6rem;
         margin:.3rem 0;max-height:20rem;overflow:auto}
tr.skip td{color:var(--mut);background:repeating-linear-gradient(135deg,
           transparent 0 6px,var(--card) 6px 12px)}
tr.skip td:first-child{box-shadow:inset 3px 0 0 var(--warn)}
tr.err td:first-child{box-shadow:inset 3px 0 0 var(--bad)}
.wrap{display:inline-block;white-space:normal;max-width:34rem;vertical-align:top}
"""


def val_page(title, body, subtitle=""):
    """The EDA page shell with the findings-only rules appended to its stylesheet.

    Appending rather than shipping a second sheet is the whole reason the two
    report sets look like one product.
    """
    return page(title, body, subtitle).replace(f"{CSS}</style>",
                                               f"{CSS}{EXTRA_CSS}</style>", 1)


def sev_span(name):
    return f"<span class={SEV_CLASS.get(name, 'mut')}>{e(name)}</span>"


def sev_rank(finding):
    return finding.get("severity_rank") or (
        50 - 10 * SEV_ORDER.index(finding.get("severity", "INFO"))
        if finding.get("severity") in SEV_ORDER else 0)


def cell(v):
    """One metric or evidence value, formatted for a table cell.

    Metrics and evidence are free-form by design -- a check scopes itself
    however it needs to -- so this has to cope with ints, rates, epoch seconds,
    address lists and prose without a schema to consult.
    """
    if v is None or v == "":
        return "<span class=mut>–</span>"
    if isinstance(v, bool):
        return f"<span class={'ok' if v else 'mut'}>{v}</span>"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        # Epoch seconds and window lengths arrive as floats too, and "900.000
        # seconds" reads as false precision, so decimals follow the value.
        return fnum(v, 0 if abs(v) >= 1e5 or float(v).is_integer() else 3)
    if isinstance(v, (list, tuple)):
        return f"<span class='mono mut'>{e(', '.join(str(x) for x in v)) or '–'}</span>"
    if isinstance(v, dict):
        return f"<span class='mono mut'>{e(json.dumps(v, default=str))}</span>"
    return f"<span class=mono>{e(v)}</span>"


def kv_table(d, k_header="metric", v_header="value"):
    return table([k_header, v_header],
                 [(f"<span class=mono>{e(k)}</span>", cell(v)) for k, v in d.items()])


def rows_table(rows, limit=200):
    """Evidence rows as a table, over the union of their keys.

    Checks do not agree on a row shape and should not have to: an endpoint
    finding carries addresses, a sweep finding carries offsets. The union keeps
    every row readable without the renderer knowing what any of them mean.
    """
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    if not cols:
        return ""
    body = [[cell(r.get(c)) for c in cols] for r in rows[:limit]]
    out = table(cols, body)
    if len(rows) > limit:
        out += (f"<p class=legend>showing {limit:,} of {len(rows):,} rows carried in "
                f"the JSON</p>")
    return out


# ------------------------------------------------------------------- charts

def sweep_chart(rows, metrics=None):
    """Flows per offset bucket around a scheduled boundary, with the declared
    window drawn over them.

    This is the picture the tier-1 checks exist to produce. The bars are what
    the traffic did; the shaded band is what the manifest claims; the marker at
    zero is the boundary under test. A band whose edge sits in the middle of a
    plateau is a window in the wrong place -- and no column of edge ratios makes
    that as immediate as seeing it, which is the entire argument for the chart.
    """
    key = next((k for k in OFFSET_KEYS if any(isinstance(r.get(k), (int, float))
                                              for r in rows)), None)
    if key is None:
        return ""
    pts = sorted((r[key], r.get("flows") or 0) for r in rows
                 if isinstance(r.get(key), (int, float)))
    if len(pts) < 2:
        return ""

    m = metrics or {}
    xs = [p[0] for p in pts]
    x0, x1 = min(xs), max(xs)
    span = (x1 - x0) or 1
    top = max(p[1] for p in pts) or 1
    # padt leaves two rows of headroom: the declared boundaries label on the
    # upper one, where the traffic actually moves on the lower, so the two
    # never sit on top of each other however close together they fall.
    w, h, padl, padt, padb = 640, 176, 46, 30, 24
    base = h - padb
    plot = w - padl - 14

    def X(v):
        return padl + plot * (v - x0) / span

    out = []
    # The declared window, drawn behind the traffic so the comparison is direct.
    win = m.get("window_seconds")
    if key == "offset_s" and isinstance(win, (int, float)) and win > 0:
        bx0, bx1 = X(max(x0, 0)), X(min(x1, win))
        if bx1 > bx0:
            out.append(f"<rect x='{bx0:.1f}' y='{padt}' width='{bx1 - bx0:.1f}' "
                       f"height='{base - padt:.1f}' fill='var(--attack)' opacity='.08'>"
                       f"<title>declared window · {win:,.0f}s</title></rect>")
    for f in (1.0, 0.5):
        gy = base - (base - padt) * f
        out.append(f"<line x1='{padl}' y1='{gy:.1f}' x2='{w - 14}' y2='{gy:.1f}' "
                   f"stroke='var(--line)'/>"
                   f"<text x='{padl - 4}' y='{gy + 3:.1f}' font-size='7' "
                   f"text-anchor='end' fill='var(--mut)'>{top * f:,.0f}</text>")

    bw = max(1.0, plot / len(pts) - 0.6)
    for x, y in pts:
        bh = (base - padt) * y / top
        out.append(f"<rect x='{X(x) - bw / 2:.1f}' y='{base - bh:.1f}' width='{bw:.1f}' "
                   f"height='{bh:.1f}' fill='var(--benign)' opacity='.85'>"
                   f"<title>{x:+,.0f}s: {y:,} flow(s)</title></rect>")

    def marker(v, col, label, row=0, dash=""):
        if not isinstance(v, (int, float)) or not (x0 <= v <= x1):
            return ""
        px = X(v)
        anchor = "end" if px > w - 120 else "start"
        dx = -3 if anchor == "end" else 3
        return (f"<line x1='{px:.1f}' y1='{padt - 22}' x2='{px:.1f}' y2='{base}' "
                f"stroke='{col}' stroke-width='1.3'{dash}/>"
                f"<text x='{px + dx:.1f}' y='{padt - 22 + 8 + row * 10}' font-size='7' "
                f"fill='{col}' text-anchor='{anchor}'>{e(label)}</text>")

    out.append(marker(0, "var(--fg)", OFFSET_KEYS[key]))
    if key == "offset_s" and isinstance(win, (int, float)) and win > 0:
        out.append(marker(win, "var(--fg)", "declared end"))
    out.append(marker(m.get("detected_rise_offset_s"), "var(--ok)", "traffic rises", 1,
                      " stroke-dasharray='3 2'"))
    out.append(marker(m.get("detected_drop_offset_s"), "var(--warn)", "traffic falls", 1,
                      " stroke-dasharray='3 2'"))

    for i in range(7):
        v = x0 + span * i / 6
        px = X(v)
        out.append(f"<line x1='{px:.1f}' y1='{base}' x2='{px:.1f}' y2='{base + 3}' "
                   f"stroke='var(--line)'/>"
                   f"<text x='{px:.1f}' y='{base + 13}' font-size='7' text-anchor='middle' "
                   f"fill='var(--mut)'>{v:+,.0f}s</text>")
    out.append(f"<line x1='{padl}' y1='{base}' x2='{w - 14}' y2='{base}' "
               f"stroke='var(--line)'/>")

    legend = (f"<p class=legend><span class=sw style='background:var(--benign)'></span>"
              f"flows per bucket &nbsp;"
              f"<span class=sw style='background:var(--attack);opacity:.3'></span>"
              f"window the manifest declares &nbsp;"
              f"<span class=sw style='background:var(--ok)'></span>where traffic rises "
              f"&nbsp;<span class=sw style='background:var(--warn)'></span>where it falls"
              f"</p>")
    return f"<svg viewBox='0 0 {w} {h}' role=img>{''.join(out)}</svg>{legend}"


def tier_matrix(findings, runlog):
    """Which tier is finding what.

    A blank row means that tier reported nothing; whether it reported nothing
    because it ran clean or because it never ran is the runlog's question, not
    this chart's, and the caption says so.
    """
    tier_of = {r.get("check_id"): r.get("tier") for r in runlog}
    tiers = sorted({t for t in tier_of.values() if t is not None})
    sevs = [s for s in SEV_ORDER
            if any(f.get("severity") == s for f in findings)]
    if not tiers or not sevs:
        return ""
    vals = [[sum(1 for f in findings
                 if tier_of.get(f.get("check_id")) == t and f.get("severity") == s)
             for s in sevs] for t in tiers]
    return matrix_heat([TIERS.get(t, f"tier {t}") for t in tiers], sevs, vals)


# ------------------------------------------------------------ dataset report

def render_dataset(doc, directory=VAL_DIR):
    meta, summ = doc.get("meta", {}), doc.get("summary", {})
    ds = meta.get("dataset", "?")
    findings = sorted(doc.get("findings", []),
                      key=lambda f: (-sev_rank(f), f.get("check_id", ""), f.get("title", "")))
    runlog = sorted(doc.get("runlog", []),
                    key=lambda r: (r.get("tier", 99), r.get("check_id", "")))
    by_sev = summ.get("by_severity", {})
    n = summ.get("n_findings", len(findings))
    n_ok = summ.get("checks_run", 0)
    n_skip = summ.get("checks_skipped", 0)
    n_err = summ.get("checks_error", 0)
    worst = findings[0].get("severity") if findings else None
    logs = meta.get("logs_available") or []

    head = kpis([
        ("findings", fnum(n),
         f"worst: {worst}" if worst else "none — this is the goal state"),
        ("blocker / critical", f"{by_sev.get('BLOCKER', 0)} / {by_sev.get('CRITICAL', 0)}",
         "every later number depends on these"),
        ("major / minor / info",
         f"{by_sev.get('MAJOR', 0)} / {by_sev.get('MINOR', 0)} / {by_sev.get('INFO', 0)}",
         "judgement calls, in policy.yaml"),
        ("checks ran", fnum(n_ok),
         f"{pct(n_ok / max(n_ok + n_skip + n_err, 1), 0)} of "
         f"{n_ok + n_skip + n_err} selected"),
        ("skipped", f"<span class={'warn' if n_skip else 'mut'}>{n_skip}</span>",
         "did not run — not a pass"),
        ("errored", f"<span class={'bad' if n_err else 'mut'}>{n_err}</span>",
         "no verdict either way"),
        ("rules", fnum(meta.get("rules_total")), "schedule windows under test"),
        ("logs", str(len(logs)), ", ".join(logs) or "none found"),
    ])

    tiers = meta.get("tiers")
    head += (f"<div class=note>This report covers "
             f"{'tiers ' + ', '.join(str(t) for t in tiers) if tiers else 'the selected checks'}"
             f" against the labelled zone at <span class=mono>{e(meta.get('root', '?'))}</span>. "
             f"A finding is a defect with the numbers that establish it — never an edited "
             f"label. Fixes go to the manifest and labelling is re-run.</div>")

    # ---- findings
    if not findings:
        find_html = (f"<div class=note><b>No findings.</b> Every check that ran agreed with "
                     f"the manifest, the schedule and the labelled zone. That is the goal "
                     f"state, not an empty report — but it is a claim about the {n_ok} check"
                     f"(s) that ran and nothing else. {n_skip} did not run and {n_err} "
                     f"errored; neither of those is a pass. The runlog below says which.</div>")
        full_html = ""
        charts = ""
    else:
        idx_rows = [(sev_span(f.get("severity", "INFO")),
                     f"<span class=mono>{e(f.get('check_id', ''))}</span>",
                     f"<a href='#f{i}'><span class=wrap>{e(f.get('title', ''))}</span></a>",
                     f"<span class='mono mut'>"
                     f"{e(', '.join(f'{k}={v}' for k, v in (f.get('scope') or {}).items())) or '–'}"
                     f"</span>",
                     fnum(len(f.get("evidence") or [])))
                    for i, f in enumerate(findings)]
        find_html = (
            "<div class=note>Ordered by consequence. BLOCKER means every number derived "
            "from these labels is describing something else; MAJOR and below are "
            "judgement calls whose thresholds live in <span class=mono>"
            "src/label_validation/config/policy.yaml</span>. Open a finding for its metrics, the rows "
            "that establish it, and the query to re-run.</div>"
            + table(["severity", "check", "finding", "scope", "evidence rows"], idx_rows))

        blocks = []
        for i, f in enumerate(findings):
            sev = f.get("severity", "INFO")
            ev = f.get("evidence") or []
            parts = [f"<p>{e(f.get('detail', '')) or '—'}</p>"]
            if f.get("metrics"):
                parts.append("<h3>Metrics</h3>" + kv_table(f["metrics"]))
            if ev:
                chart = sweep_chart(ev, f.get("metrics"))
                if chart:
                    parts.append(f"<h3>Traffic around the boundary</h3>"
                                 f"<div class=chart>{chart}</div>")
                trunc = f.get("evidence_truncated") or 0
                parts.append(f"<h3>Evidence <span class=mut>· {len(ev):,} row(s)"
                             f"{f', {trunc:,} more dropped at write time' if trunc else ''}"
                             f"</span></h3>" + rows_table(ev))
            if f.get("repro"):
                parts.append(f"<h3>Reproduce</h3><pre class=mono>{e(f['repro'])}</pre>")
            # The consequential findings are open on arrival: with no JS, a
            # closed <details> is a finding nobody reads.
            blocks.append(
                f"<details id=f{i}{' open' if sev in ('BLOCKER', 'CRITICAL') else ''}>"
                f"<summary>{sev_span(sev)} <span class=mono>{e(f.get('check_id', ''))}</span> "
                f"— {e(f.get('title', ''))}</summary>{''.join(parts)}</details>")
        full_html = "".join(blocks)

        per_check = {}
        worst_of = {}
        for f in findings:
            cid = f.get("check_id", "?")
            per_check[cid] = per_check.get(cid, 0) + 1
            if sev_rank(f) >= worst_of.get(cid, (0, ""))[0]:
                worst_of[cid] = (sev_rank(f), f.get("severity", "INFO"))
        col = {"bad": "var(--attack)", "warn": "var(--warn)", "mut": "var(--rest)"}
        ranked = diverging_bars(
            [(c, v / n, col[SEV_CLASS.get(worst_of[c][1], "mut")])
             for c, v in sorted(per_check.items(), key=lambda kv: -kv[1])])
        scopes = {}
        for f in findings:
            s = (f.get("scope") or {}).get("class") or (f.get("scope") or {}).get("raw")
            if s:
                scopes[str(s)] = scopes.get(str(s), 0) + 1
        by_scope = (f"<div class=chart><div class=t>Findings by what they are about</div>"
                    f"<div class=s>canonical class where the check scoped one</div>"
                    f"{hbar(scopes, sum(scopes.values()))}</div>" if scopes else "")
        tm = tier_matrix(findings, runlog)
        charts = (f"<h2>Where the findings are</h2>"
                  f"<div class=grid style='grid-template-columns:1fr 1fr'>"
                  f"<div class=chart><div class=t>Findings by check</div>"
                  f"<div class=s>share of all {n} finding(s), coloured by worst severity"
                  f"</div>{ranked}</div>{by_scope}</div>"
                  + (f"<div class=chart style='margin-top:1rem'>"
                     f"<div class=t>Findings by tier</div><div class=s>an empty row means "
                     f"that tier reported nothing — which is not the same as that tier "
                     f"having run; the runlog below is the authority on that</div>{tm}"
                     f"</div>" if tm else ""))

    # ---- runlog
    st = {"ok": ("<span class=ok>ran</span>", ""),
          "error": ("<span class=bad>ERRORED</span>", "err"),
          "skipped": ("<span class=warn>SKIPPED</span>", "skip")}
    slowest = max((r.get("seconds") or 0 for r in runlog), default=0)
    log_rows = []
    for r in runlog:
        label, cls = st.get(r.get("status"), ("<span class=mut>?</span>", ""))
        secs = r.get("seconds")
        tier = r.get("tier")
        log_rows.append((
            (MARK[cls] if cls else "") + label,
            e(TIERS.get(tier, f"tier {tier}")) if tier is not None else "–",
            f"<span class=mono>{e(r.get('check_id', ''))}</span>",
            f"<span class=wrap>{e(r.get('title', ''))}</span>",
            (f"{secs:,.1f}s " + bar(secs, slowest, "var(--rest)", 50)
             if secs is not None else "<span class=mut>–</span>"),
            (f"<span class={'warn' if r.get('n_findings') else 'mut'}>"
             f"{r.get('n_findings', 0)}</span>"),
            f"<span class=mut>{e(r.get('reason') or r.get('error') or '')}</span>",
        ))
    log_html = table(["status", "tier", "check", "asks", "time", "findings",
                      "reason / error"], log_rows)
    # A dimmed row needs the class on the <tr>, and the shared table() helper
    # builds cells, not rows. Rather than fork it, each row that needs one plants
    # a comment in its first cell and it is promoted to a row class here.
    for cls, mark in MARK.items():
        log_html = log_html.replace(f"<tr><td>{mark}", f"<tr class={cls}><td>")

    prov = table(["field", "value"], [
        (k, f"<span class=mono>{e(v)}</span>") for k, v in [
            ("dataset", ds), ("role", meta.get("role")), ("lake root", meta.get("root")),
            ("generated (UTC)", meta.get("generated_utc")),
            ("manifest sha", meta.get("manifest_sha")),
            ("taxonomy sha", meta.get("taxonomy_sha")),
            ("policy version", meta.get("policy_version")),
            ("report version", doc.get("validate_version")),
        ]])

    eda = Path(directory).parent / "eda" / f"profile_{ds}.html"
    footer = "<p class=sub><a href='index.html'>← all validation reports</a>"
    if eda.exists():
        footer += f" · <a href='../eda/profile_{e(ds)}.html'>EDA profile for {e(ds)} →</a>"
    footer += "</p>"

    body = f"""{head}
<h2>Findings</h2>{find_html}
{charts}
{'<h2>Findings in full</h2>' + full_html if full_html else ''}
<h2>What ran, and what did not</h2>
<div class=note>A skipped check is not a passed check. Skipped and errored rows are shaded
and flagged rather than listed alongside the passes, because "did not run" and "ran and
found nothing" are different claims about this dataset, and a report that blurs them is
worth less than no report.</div>{log_html}
<h2>Provenance</h2>
<div class=note>The manifest and taxonomy hashes are the ones the checks read. If they
differ from the hashes stamped into the labelled zone, <span class=mono>int.provenance_current</span>
says so above — and nothing else on this page can be trusted until it does.</div>{prov}
{footer}"""
    return val_page(f"Label validation · {ds}", body,
                    f"{n} finding(s) · {n_ok} check(s) ran, {n_skip} skipped, "
                    f"{n_err} errored · generated {e(meta.get('generated_utc', '?'))}")


# -------------------------------------------------------------------- index

def render_index(docs):
    rows = []
    for ds in sorted(docs):
        d = docs[ds]
        m, s = d.get("meta", {}), d.get("summary", {})
        bs = s.get("by_severity", {})
        n = s.get("n_findings", 0)
        blocking = bs.get("BLOCKER", 0) + bs.get("CRITICAL", 0)
        rows.append((
            f"<b>{e(ds)}</b>", f"<span class=tag>{e(m.get('role', '?'))}</span>",
            (f"<span class={'bad' if blocking else ('warn' if n else 'ok')}>"
             f"{n:,}</span>"),
            *[(f"<span class={SEV_CLASS[k]}>{bs[k]:,}</span>" if bs.get(k) else
               "<span class=mut>–</span>") for k in SEV_ORDER],
            f"{s.get('checks_run', 0)} / "
            f"<span class={'warn' if s.get('checks_skipped') else 'mut'}>"
            f"{s.get('checks_skipped', 0)}</span> / "
            f"<span class={'bad' if s.get('checks_error') else 'mut'}>"
            f"{s.get('checks_error', 0)}</span>",
            f"<span class=mut>{e(m.get('generated_utc', ''))}</span>",
            f"<a href='{e(ds)}.html'>report</a>",
        ))

    sevs = [s for s in SEV_ORDER
            if any(d.get("summary", {}).get("by_severity", {}).get(s)
                   for d in docs.values())]
    names = sorted(docs)
    heat = ""
    if sevs and names:
        vals = [[docs[ds].get("summary", {}).get("by_severity", {}).get(s, 0)
                 for s in sevs] for ds in names]
        heat = (f"<h2>Findings across the pool</h2>"
                f"<div class=note>The same checks against every dataset. A column that "
                f"lights up for one dataset only is that dataset's problem; one that lights "
                f"up everywhere is usually the check's threshold, or a shared assumption in "
                f"the labelling stage.</div>"
                f"<div class=chart>{matrix_heat(names, sevs, vals)}</div>")

    total = sum(d.get("summary", {}).get("n_findings", 0) for d in docs.values())
    skipped = sum(d.get("summary", {}).get("checks_skipped", 0) for d in docs.values())
    body = f"""<div class=note>Talos labels flows from a published attack schedule — a
document, not a measurement. These reports establish, with evidence, whether that document
was transcribed correctly, whether the traffic agrees with it, and which flows cannot be
adjudicated either way. Nothing here edits a label; findings go to the manifests and
labelling is re-run. <span class=mono>make validate-gate</span> turns them into an exit
code.</div>
{table(['dataset', 'role', 'findings', *[s.lower() for s in SEV_ORDER],
        'ran / skipped / errored', 'generated', ''], rows)}
{heat}
<p class=sub>{total:,} finding(s) across {len(docs)} dataset(s){f' · {skipped} check(s) did not run' if skipped else ''}
{" · <a href='../eda/index.html'>EDA reports →</a>" if (VAL_DIR.parent / 'eda' / 'index.html').exists() else ''}</p>"""
    return val_page("Talos · label validation", body,
                    f"{len(docs)} dataset(s) validated")


def dataset_reports(directory):
    """Every per-dataset findings JSON in a directory, keyed by dataset.

    Discovery by glob, like the EDA stage: adding a dataset adds a row without
    anyone editing a list. Shape is checked as well as the filename, so an
    unrelated JSON dropped in this directory is ignored rather than half-rendered.
    """
    out = {}
    for path in sorted(Path(directory).glob("*.json")):
        if any(path.stem == n or path.stem.startswith(n + "_") for n in NOT_A_DATASET):
            continue
        try:
            doc = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if isinstance(doc, dict) and doc.get("meta", {}).get("dataset") \
                and isinstance(doc.get("findings"), list):
            out[doc["meta"]["dataset"]] = doc
    return out


def render_all(directory=VAL_DIR):
    """Rebuild every page from whatever JSON is on disk. Returns paths written."""
    directory = Path(directory)
    docs = dataset_reports(directory)
    written = []
    for ds, doc in sorted(docs.items()):
        out = directory / f"{ds}.html"
        out.write_text(render_dataset(doc, directory))
        written.append(out)
    if docs:
        idx = directory / "index.html"
        idx.write_text(render_index(docs))
        written.append(idx)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=str(VAL_DIR))
    a = p.parse_args()
    written = render_all(a.dir)
    if not written:
        sys.exit(f"no validation JSON in {a.dir} — run `make validate-all` first")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
