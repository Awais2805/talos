#!/usr/bin/env python3
"""Turn the EDA JSON into standalone HTML: one page per report, plus an index.

Rendering is deliberately the last and dumbest stage. It computes no statistic
that compare.py does not already own -- it imports that kernel rather than
reimplementing it, so a number can never mean one thing in the JSON and another
in the page. Charts are hand-written inline SVG: the reports have to open from
a laptop, an EC2 box or a dissertation appendix with no network and no JS.

Usage:
    python src/eda/render.py                     # rebuild every page
    python src/eda/render.py --dir reports/eda
"""

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eda.compare import (class_view, corr_matrix, js_divergence,   # noqa: E402
                             label_signal, moments, r6)

EDA_DIR = ROOT / "reports" / "eda"

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#6b7280;--line:#e5e7eb;--card:#fafafa;
      --benign:#4c78a8;--attack:#e45756;--rest:#9aa0a6;--ok:#2f7d32;--warn:#b45309;--bad:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e8eaed;--mut:#9aa0a6;
      --line:#2a2f36;--card:#171b20}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem 5rem;background:var(--bg);color:var(--fg);
     font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 .2rem} h2{font-size:1.05rem;margin:2.6rem 0 .6rem;
   padding-bottom:.35rem;border-bottom:1px solid var(--line)}
h3{font-size:.85rem;margin:1.2rem 0 .3rem;font-weight:600}
.sub{color:var(--mut);margin:0 0 1.4rem}
a{color:inherit}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.6rem;margin:1rem 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.7rem .8rem}
.kpi .v{font-size:1.25rem;font-weight:600;font-variant-numeric:tabular-nums}
.kpi .k{color:var(--mut);font-size:.72rem;text-transform:uppercase;letter-spacing:.04em}
.kpi .n{color:var(--mut);font-size:.72rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:.32rem .5rem;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:600;text-align:right;font-size:.72rem;text-transform:uppercase;
   letter-spacing:.03em;position:sticky;top:0;background:var(--bg)}
td:first-child,th:first-child{text-align:left}
tbody tr:hover{background:var(--card)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.bar{height:9px;border-radius:2px;display:inline-block;vertical-align:middle}
.tag{display:inline-block;padding:.05rem .4rem;border-radius:10px;font-size:.7rem;
     border:1px solid var(--line)}
.ok{color:var(--ok)} .warn{color:var(--warn)} .bad{color:var(--bad)} .mut{color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:1rem}
.chart{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.6rem .7rem}
.chart .t{font-size:.78rem;font-weight:600;margin-bottom:.15rem}
.chart .s{font-size:.7rem;color:var(--mut);margin-bottom:.3rem}
svg{display:block;width:100%;height:auto}
.legend{font-size:.7rem;color:var(--mut);margin:.2rem 0 0}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:.25rem}
.note{background:var(--card);border-left:3px solid var(--line);padding:.6rem .8rem;
      color:var(--mut);font-size:.82rem;margin:.8rem 0}
"""


def e(x):
    return html.escape(str(x))


def fnum(x, dp=0):
    if x is None:
        return "<span class=mut>–</span>"
    if isinstance(x, float) and abs(x) < 1 and x != 0 and dp == 0:
        return f"{x:.3f}"
    return f"{x:,.{dp}f}"


def pct(x, dp=2):
    return "<span class=mut>–</span>" if x is None else f"{100 * x:.{dp}f}%"


def page(title, body, subtitle=""):
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{e(title)}</title><style>{CSS}</style></head><body><main>"
            f"<h1>{e(title)}</h1><p class=sub>{subtitle}</p>{body}</main></body></html>")


def kpis(items):
    cells = "".join(
        f"<div class=kpi><div class=k>{e(k)}</div><div class=v>{v}</div>"
        f"<div class=n>{note}</div></div>" for k, v, note in items)
    return f"<div class=kpis>{cells}</div>"


def table(headers, rows, cls=""):
    head = "".join(f"<th>{e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f"<div class=scroll><table class='{cls}'><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def bar(value, vmax, color="var(--benign)", width=90):
    w = 0 if not vmax else max(1, round(width * value / vmax))
    return f"<span class=bar style='width:{w}px;background:{color}'></span>"


# ------------------------------------------------------------------ charts

def dual_hist(labels, a, b, a_name, b_name, a_col="var(--benign)", b_col="var(--attack)"):
    """Two normalised distributions on one axis -- shares, never raw counts.

    Raw counts would be unreadable side by side here: benign and attack differ
    by three orders of magnitude within a dataset, and the datasets differ by
    another two between each other. Shape is the question; scale is answered
    elsewhere.
    """
    n = len(labels)
    if not n:
        return ""
    sa, sb = sum(a) or 1, sum(b) or 1
    pa, pb = [v / sa for v in a], [v / sb for v in b]
    top = max(max(pa, default=0), max(pb, default=0)) or 1
    w, h, pad = 320, 92, 14
    bw = (w - 2 * pad) / n
    bars = []
    for i in range(n):
        x = pad + i * bw
        for vals, col, name, off in ((pa, a_col, a_name, 0.0), (pb, b_col, b_name, bw * 0.46)):
            bh = (h - pad - 12) * vals[i] / top
            bars.append(f"<rect x='{x + off:.1f}' y='{h - 12 - bh:.1f}' "
                        f"width='{bw * 0.42:.1f}' height='{bh:.1f}' fill='{col}'>"
                        f"<title>{e(name)} · {e(labels[i])}: {vals[i]:.4%}</title></rect>")
    ticks = "".join(
        f"<text x='{pad + (i + 0.5) * bw:.1f}' y='{h - 2}' font-size='6' fill='var(--mut)' "
        f"text-anchor='middle'>{e(labels[i].split('–')[0].lstrip('<≥'))}</text>"
        for i in range(n) if n <= 18 or i % 2 == 0)
    return (f"<svg viewBox='0 0 {w} {h}' role=img>"
            f"<line x1='{pad}' y1='{h - 12}' x2='{w - pad}' y2='{h - 12}' "
            f"stroke='var(--line)'/>{''.join(bars)}{ticks}</svg>"
            f"<p class=legend><span class=sw style='background:{a_col}'></span>{e(a_name)}"
            f" &nbsp;<span class=sw style='background:{b_col}'></span>{e(b_name)}</p>")


def _corr_color(v):
    if v is None:
        return "var(--card)"
    t = max(-1.0, min(1.0, v))
    if t >= 0:
        return f"rgba(76,120,168,{0.08 + 0.85 * t:.3f})"
    return f"rgba(228,87,86,{0.08 + 0.85 * -t:.3f})"


def heatmap(order, matrix, title="", cell=15):
    """Correlation (or Δcorrelation) as an SVG grid; hover gives the number."""
    n = len(order)
    if not n:
        return ""
    lab = 118
    w, h = lab + n * cell + 6, lab + n * cell + 6
    out = []
    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            out.append(f"<rect x='{lab + j * cell}' y='{lab + i * cell}' width='{cell - 1}' "
                       f"height='{cell - 1}' fill='{_corr_color(v)}'>"
                       f"<title>{e(order[i])} × {e(order[j])}: "
                       f"{'n/a' if v is None else f'{v:+.3f}'}</title></rect>")
    for i, name in enumerate(order):
        out.append(f"<text x='{lab - 4}' y='{lab + i * cell + cell - 4}' font-size='8' "
                   f"text-anchor='end' fill='var(--fg)'>{e(name)}</text>")
        out.append(f"<text x='{lab + i * cell + cell - 4}' y='{lab - 4}' font-size='8' "
                   f"transform='rotate(-90 {lab + i * cell + cell - 4} {lab - 4})' "
                   f"fill='var(--fg)'>{e(name)}</text>")
    return (f"{'<h3>' + e(title) + '</h3>' if title else ''}"
            f"<svg viewBox='0 0 {w} {h}' role=img>{''.join(out)}</svg>")


def scatter(points, xlab, ylab, xmax=1.0, ymax=1.0):
    """One dot per feature: how domain-specific it is vs how much it says."""
    w, h, pad = 520, 300, 40
    dots = []
    for name, x, y, col in points:
        if x is None or y is None:
            continue
        px = pad + (w - 2 * pad) * min(x / xmax, 1)
        py = h - pad - (h - 2 * pad) * min(y / ymax, 1)
        dots.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3.5' fill='{col}' "
                    f"opacity='.85'><title>{e(name)}: {x:.3f}, {y:.3f}</title></circle>"
                    f"<text x='{px + 5:.1f}' y='{py + 3:.1f}' font-size='7' "
                    f"fill='var(--mut)'>{e(name)}</text>")
    grid = "".join(
        f"<line x1='{pad}' y1='{h - pad - (h - 2 * pad) * f:.0f}' x2='{w - pad}' "
        f"y2='{h - pad - (h - 2 * pad) * f:.0f}' stroke='var(--line)'/>"
        f"<line x1='{pad + (w - 2 * pad) * f:.0f}' y1='{pad}' "
        f"x2='{pad + (w - 2 * pad) * f:.0f}' y2='{h - pad}' stroke='var(--line)'/>"
        for f in (0, 0.25, 0.5, 0.75, 1))
    return (f"<svg viewBox='0 0 {w} {h}' role=img>{grid}{''.join(dots)}"
            f"<text x='{w / 2}' y='{h - 8}' font-size='9' text-anchor='middle' "
            f"fill='var(--mut)'>{e(xlab)} →</text>"
            f"<text x='12' y='{h / 2}' font-size='9' text-anchor='middle' fill='var(--mut)' "
            f"transform='rotate(-90 12 {h / 2})'>{e(ylab)} →</text></svg>")


# --------------------------------------------------------- profile report

def render_profile(profile):
    meta = profile["meta"]
    ds = meta["dataset"]
    views = {w: class_view(profile, w) for w in ("all", "benign", "attack")}
    n_all = views["all"]["n"] or 1
    n_atk = views["attack"]["n"]
    feats = list(profile["numeric_spec"])

    head = kpis([
        ("flows", fnum(meta["rows"]), f"{meta['zone']} zone"),
        ("attack", pct(n_atk / n_all), f"{fnum(n_atk)} flows"),
        ("classes", str(len(profile["by_class"])), "incl. benign"),
        ("captures", str(len({c['capture'] for c in profile['captures']})),
         "distinct capture folders"),
        ("features", f"{meta['numeric_features']}+{meta['categorical_features']}",
         "numeric + categorical"),
        ("scan", f"{meta['scan_seconds']}s",
         f"spec v{meta['spec_version']}·{meta['spec_sha']}"),
    ])
    if meta.get("partial"):
        head += (f"<div class=note><b>Partial profile — not a result.</b> Built from "
                 f"{e(meta['partial'])}. Shares are indicative, absolute counts are not, "
                 f"rare classes may be missing entirely, and a row limit takes the "
                 f"earliest flows rather than a representative draw.</div>")
    if meta["skipped_features"]:
        miss = "; ".join(f"<b>{e(k)}</b> needs {e(', '.join(v))}"
                         for k, v in meta["skipped_features"].items())
        head += f"<div class=note>Not measurable on this source: {miss}.</div>"

    # ---- class distribution
    order = sorted(profile["by_class"].items(), key=lambda kv: -kv[1]["n"])
    top = order[0][1]["n"] if order else 1
    rows = [(f"<span class=mono>{e(c)}</span>", fnum(b["n"]), pct(b["n"] / n_all),
             bar(b["n"], top, "var(--attack)" if c != profile["benign_class"]
                 else "var(--benign)"))
            for c, b in order]
    cls_html = table(["class", "flows", "share", ""], rows)

    # ---- capture / domain sub-structure
    caps = {}
    for row in profile["captures"]:
        c = caps.setdefault(row["capture"], {"n": 0, "atk": 0, "t0": None, "t1": None})
        c["n"] += row["n"]
        c["atk"] += row["n"] if row["class"] != profile["benign_class"] else 0
        c["t0"] = row["ts_min"] if c["t0"] is None else min(c["t0"], row["ts_min"])
        c["t1"] = row["ts_max"] if c["t1"] is None else max(c["t1"], row["ts_max"])
    cap_rows = [(f"<span class=mono>{e(k)}</span>", fnum(v["n"]), pct(v["n"] / n_all),
                 pct(v["atk"] / v["n"] if v["n"] else 0),
                 f"{(v['t1'] - v['t0']) / 3600:.1f} h" if v["t0"] and v["t1"] else "–",
                 bar(v["n"], max(c["n"] for c in caps.values())))
                for k, v in sorted(caps.items(), key=lambda kv: -kv[1]["n"])]
    cap_html = (table(["capture", "flows", "share of dataset", "attack ratio", "span", ""],
                      cap_rows) if cap_rows else
                "<p class=mut>no capture column on this source</p>")

    # ---- numeric feature table
    qs = profile["quantiles"]
    frows = []
    for f in feats:
        blk_all = views["all"]["numeric"][f]
        blk_b, blk_a = views["benign"]["numeric"][f], views["attack"]["numeric"][f]
        mean, var = moments(blk_all, n_all)
        u = (label_signal(blk_b["hist"], blk_a["hist"], balanced=True)
             if blk_b["hist"] and blk_a["hist"] else None)
        # Quantiles are per class in the profile; the dataset-wide median is
        # taken from the largest class rather than faked by averaging t-digests.
        big = max(profile["by_class"], key=lambda c: profile["by_class"][c]["n"])
        q = profile["by_class"][big]["numeric"][f]["q"]
        frows.append((
            f"<span class=mono>{e(f)}</span>",
            e(profile["numeric_spec"][f]["transform"]),
            fnum(blk_all["min"], 3), fnum(q.get(qs[len(qs) // 2]), 3),
            fnum(q.get(qs[-1]), 3), fnum(blk_all["max"], 3),
            fnum(mean, 3), fnum(var ** 0.5 if var is not None else None, 3),
            pct(blk_all["n_zero"] / n_all, 1), pct(blk_all["n_undefined"] / n_all, 1),
            (f"<span class={'ok' if u and u > .1 else 'mut'}>{u:.3f}</span>"
             if u is not None else "<span class=mut>–</span>"),
        ))
    num_html = table(["feature", "space", "min", "p50*", "p99*", "max", "mean†", "sd†",
                      "zeros", "undef", "label signal"], frows)

    # ---- per-feature benign vs attack shape
    charts = []
    for f in feats:
        if f == "label_binary":
            continue
        spec = profile["numeric_spec"][f]
        b, a = views["benign"]["numeric"][f]["hist"], views["attack"]["numeric"][f]["hist"]
        if not b or not a:
            continue
        j = js_divergence(b, a)
        charts.append(
            f"<div class=chart><div class=t>{e(f)}</div>"
            f"<div class=s>benign vs attack · JS {j:.3f}</div>"
            f"{dual_hist(spec['bin_labels'], b, a, 'benign', 'attack')}</div>")
    shape_html = f"<div class=grid>{''.join(charts)}</div>"

    # ---- correlation
    cm = corr_matrix(views["all"], feats)
    corr_html = heatmap(feats, [[r6(v) for v in row] for row in cm])

    # ---- categorical
    cat_blocks = []
    for c in profile["categorical_spec"]:
        blk_all = views["all"]["categorical"].get(c)
        if not blk_all:
            continue
        blk_b = views["benign"]["categorical"].get(c, {"counts": {}})
        blk_a = views["attack"]["categorical"].get(c, {"counts": {}})
        tot = sum(blk_all["counts"].values()) + blk_all["n_other"]
        top_vals = sorted(blk_all["counts"].items(), key=lambda kv: -kv[1])[:12]
        rows = [(f"<span class=mono>{e(v)}</span>", fnum(n), pct(n / tot, 2),
                 pct(blk_b["counts"].get(v, 0) / max(views["benign"]["n"], 1), 2),
                 pct(blk_a["counts"].get(v, 0) / max(views["attack"]["n"], 1), 2),
                 bar(n, top_vals[0][1]))
                for v, n in top_vals]
        cat_blocks.append(
            f"<h3>{e(c)} <span class=mut>· {blk_all['n_distinct']:,} distinct"
            f"{', tail ' + pct(blk_all['n_other'] / tot, 2) if blk_all['n_other'] else ''}"
            f"</span></h3>"
            + table(["value", "flows", "share", "of benign", "of attack", ""], rows))
    cat_html = "".join(cat_blocks)

    # ---- data quality
    nulls = {}
    dist = {}
    for blk in profile["by_class"].values():
        for col, n in blk.get("nulls", {}).items():
            nulls[col] = nulls.get(col, 0) + n
        for col, n in blk.get("distinct", {}).items():
            dist[col] = max(dist.get(col, 0), n)
    q_rows = [(f"<span class=mono>{e(c)}</span>", fnum(n), pct(n / n_all, 3))
              for c, n in sorted(nulls.items(), key=lambda kv: -kv[1])]
    d_rows = [(f"<span class=mono>{e(c)}</span>", fnum(n), "approx_count_distinct")
              for c, n in sorted(dist.items(), key=lambda kv: -kv[1])]

    body = f"""{head}
<h2>Class balance</h2>{cls_html}
<h2>Capture sub-domains</h2>
<div class=note>A dataset is not one environment: each capture day has its own host mix and
attack schedule. Divergence between captures here is a lower bound on the divergence you
should expect between datasets.</div>{cap_html}
<h2>Numeric features</h2>
<div class=note>* p50/p99 are from the largest class ({e(max(profile['by_class'], key=lambda c: profile['by_class'][c]['n']))}),
since t-digests do not merge across classes. † mean and sd are in the
<i>transformed</i> space named in the space column (log1p for the heavy-tailed counters),
which is also the space the correlations use. Label signal is balanced normalised mutual
information with the attack/benign label: 0 = the feature says nothing, 1 = it decides
the label on its own.</div>{num_html}
<h2>Distribution shape: benign vs attack</h2>
<div class=note>Fixed bin edges from the EDA spec, shown as within-class shares so the
two classes are comparable despite the imbalance. JS is the Jensen–Shannon divergence in
bits between the two classes: high means this feature separates attack from benign
<i>in this dataset</i>.</div>{shape_html}
<h2>Feature correlation</h2>
<div class=note>Pearson r over all flows in the transformed space, computed exactly from
summed moments. <span class=mono>label_binary</span> is included, so its row is the
point-biserial correlation between each feature and the attack label.</div>{corr_html}
<h2>Categorical features</h2>{cat_html}
<h2>Data quality</h2><h3>Nulls in the raw columns</h3>{table(['column', 'nulls', 'share'], q_rows)}
<h3>Identity cardinality</h3>
<div class=note>Counted, never used as features — an address is a fact about the testbed.</div>
{table(['column', 'distinct (max over classes)', 'method'], d_rows)}
<h2>Provenance</h2>
{table(['field', 'value'], [(k, f"<span class=mono>{e(v)}</span>") for k, v in
    [('source', meta['source']), ('generated (UTC)', meta['generated_utc']),
     ('spec', f"v{meta['spec_version']} sha {meta['spec_sha']}"),
     ('duckdb', meta['duckdb']), ('role', meta['role'])]])}
<p class=sub><a href="index.html">← all EDA reports</a> ·
<a href="compare_{e(ds)}.html">one-vs-rest comparison →</a></p>"""
    return page(f"EDA · {ds}",
                body,
                f"Individual profile · {fnum(meta['rows'])} flows · "
                f"generated {e(meta['generated_utc'])}")


# ------------------------------------------------------ comparison report

def render_compare(doc):
    m, o = doc["meta"], doc["overview"]
    ds = m["dataset"]
    peers = m["compared_against"]
    if not peers:
        return page(f"EDA · {ds} vs rest",
                    "<div class=note>This is the only dataset profiled so far, so there is "
                    "nothing to compare it against. Profile another dataset and this page "
                    "regenerates automatically.</div>", "Nothing in the pool yet")

    head = kpis([
        ("domain fingerprint", fnum(o["domain_fingerprint"]),
         "mean benign-vs-benign JS"),
        ("concept shift", fnum(o["concept_shift"]), "mean attack-vs-attack JS"),
        ("class mix JS", fnum(o["class_js"]), "label distribution"),
        ("Δcorrelation", fnum(o["corr_delta_mean_abs"]),
         f"mean |Δr| · max {fnum(o['corr_delta_max_abs'])}"),
        ("domain-biased", str(o["n_domain_biased"]),
         f"of {m['features_compared']} features"),
        ("transferable", str(o["n_transfers"]), "stable and informative"),
    ])

    interp = ("<div class=note><b>How to read the headline.</b> "
              "<i>Domain fingerprint</i> compares only benign traffic on each side, so it "
              "isolates how much of the difference is the testbed rather than the attacks. "
              "A high value means a classifier can tell which capture a flow came from "
              "without ever seeing an attack — which is exactly the shortcut a cross-domain "
              "model will take if you let it. <i>Concept shift</i> is the same measure on "
              "attack traffic: high means the same attack class looks different here.</div>")

    # ---- overview / class mix
    ov = table(["", ds, "rest of pool (" + ", ".join(peers) + ")"], [
        ("flows", fnum(o["flows_self"]), fnum(o["flows_rest"])),
        ("attack ratio", pct(o["attack_ratio_self"]), pct(o["attack_ratio_rest"])),
        ("classes present", str(sum(1 for r in doc["class_distribution"] if r["n_self"])),
         str(sum(1 for r in doc["class_distribution"] if r["n_rest"]))),
        ("classes unique to it",
         f"<span class=mono>{e(', '.join(o['classes_only_self']) or '—')}</span>",
         f"<span class=mono>{e(', '.join(o['classes_only_rest']) or '—')}</span>"),
    ])
    crows = [(f"<span class=mono>{e(r['class'])}</span>",
              fnum(r["n_self"]), pct(r["share_self"]),
              bar(r["share_self"] or 0, 1, "var(--benign)"),
              fnum(r["n_rest"]), pct(r["share_rest"]),
              bar(r["share_rest"] or 0, 1, "var(--rest)"),
              "<span class=warn>only here</span>" if r["only_self"] else
              ("<span class=mut>absent here</span>" if r["only_rest"] else ""))
             for r in sorted(doc["class_distribution"], key=lambda r: -r["n_self"])]
    cls_html = table(["class", "flows", "share", "", "flows (rest)", "share (rest)", "", ""],
                     crows)

    # ---- numeric drift
    tags = {"domain-biased": "bad", "transfers": "ok", "dead": "mut",
            "uninformative": "mut", "ok": ""}
    nrows = [(f"<span class=mono>{e(r['feature'])}</span>",
              fnum(r["js_domain"]) + " " + bar(r["js_domain"] or 0, 1, "var(--attack)", 60),
              fnum(r["js_concept"]), fnum(r["js_all"]),
              fnum(r["cohens_d"]), fnum(r["var_ratio"]),
              fnum(r["label_signal_self"]), fnum(r["label_signal_rest"]),
              pct(r["undefined_self"], 1),
              f"<span class='tag {tags.get(r['verdict'], '')}' title='{e(r['verdict_reason'])}'>"
              f"{e(r['verdict'])}</span>")
             for r in doc["numeric"]]
    num_html = table(["feature", "domain JS (benign↔benign)", "concept JS", "overall JS",
                      "Cohen's d", "var ratio", "signal here", "signal rest", "undef",
                      "verdict"], nrows)

    pts = [(r["feature"], r["js_domain"], r["label_signal_self"],
            {"domain-biased": "var(--attack)", "transfers": "var(--ok)"}
            .get(r["verdict"], "var(--rest)"))
           for r in doc["numeric"] if r["feature"] != "label_binary"]
    scat = scatter(pts, "domain fingerprint (benign-vs-benign JS)",
                   "label signal in this dataset")

    # ---- correlation
    order = doc["correlation"]["order"]
    delta = [[None if a is None or b is None else a - b
              for a, b in zip(ra, rb)]
             for ra, rb in zip(doc["correlation"]["self"], doc["correlation"]["rest"])]
    heat = (f"<div class=grid style='grid-template-columns:1fr 1fr'>"
            f"<div class=chart><div class=t>{e(ds)}</div>"
            f"{heatmap(order, doc['correlation']['self'])}</div>"
            f"<div class=chart><div class=t>rest of pool</div>"
            f"{heatmap(order, doc['correlation']['rest'])}</div></div>"
            f"<h3>difference (this − rest)</h3>{heatmap(order, delta)}")
    drift_html = table(["feature A", "feature B", "r here", "r in rest", "Δ"],
                       [(f"<span class=mono>{e(d['a'])}</span>",
                         f"<span class=mono>{e(d['b'])}</span>",
                         fnum(d["rho_self"]), fnum(d["rho_rest"]),
                         f"<span class={'bad' if abs(d['delta']) > .4 else ''}>"
                         f"{d['delta']:+.3f}</span>")
                        for d in doc["correlation"]["top_drift"]])

    lc_html = table(["feature", "r with label here", "r with label in rest", ""],
                    [(f"<span class=mono>{e(r['feature'])}</span>",
                      fnum(r["rho_self"]), fnum(r["rho_rest"]),
                      "<span class=bad>sign flips</span>" if r["flips_sign"] else "")
                     for r in doc["label_correlation"]])

    # ---- categorical
    cat_html = table(["feature", "domain JS", "overall JS", "distinct here", "distinct rest",
                      "shared", "traffic on values the pool never sees", "examples"],
                     [(f"<span class=mono>{e(r['feature'])}</span>",
                       fnum(r["js_domain"]), fnum(r["js_all"]),
                       fnum(r["n_distinct_self"]), fnum(r["n_distinct_rest"]),
                       fnum(r["shared_values"]), pct(r["unseen_mass"], 2),
                       f"<span class='mono mut'>{e(', '.join(r['only_self'][:6]) or '—')}</span>")
                      for r in doc["categorical"]])

    sampled = ("<div class=note><b>Partial profiles in this pool.</b> " +
               "; ".join(f"{e(k)}: {e(v)}" for k, v in m.get("partial", {}).items()) +
               " — the comparison is on shares so it still holds, but absolute counts "
               "and rare classes do not.</div>"
               if m.get("partial") else "")

    body = f"""{head}{interp}{sampled}
<h2>At a glance</h2>{ov}
<h2>Class distribution</h2>{cls_html}
<h2>Feature-level drift</h2>
<div class=note>Domain JS compares this dataset's <b>benign</b> flows with the pool's benign
flows on the same fixed bins; concept JS does the same for attack flows. Cohen's d and the
variance ratio are in the transformed space. "Signal" is balanced normalised mutual
information with the label — computed with a 50/50 prior so a 98%-attack capture and a
25%-attack capture can be put side by side.</div>{num_html}
<h2>Domain specificity vs label signal</h2>
<div class=note>The bottom-right corner is where cross-domain generalisation goes to die:
features that identify the capture but say little about the label. The top-left is what you
want to keep.</div><div class=chart>{scat}</div>
<h2>Correlation structure</h2>
<div class=note>Two datasets can agree on every marginal distribution and still disagree on
how features move together — that difference is what a model trained on one and tested on
the other actually trips over.</div>{heat}
<h3>Largest correlation shifts</h3>{drift_html}
<h3>Relation to the label</h3>
<div class=note>Point-biserial correlation with the attack label. A sign flip means the
feature argues for "attack" here and for "benign" in the rest of the pool.</div>{lc_html}
<h2>Categorical drift</h2>{cat_html}
<h2>Provenance</h2>
<div class=note>Built only from the per-dataset profiles listed below — no lake access.
Comparability is enforced by the spec hash <span class=mono>{e(m['spec_sha'])}</span>.</div>
{table(['profile', 'generated (UTC)'],
       [(f"<span class=mono>{e(k)}</span>", e(v)) for k, v in m['profiles_used'].items()])}
<p class=sub><a href="index.html">← all EDA reports</a> ·
<a href="profile_{e(ds)}.html">individual profile →</a></p>"""
    return page(f"EDA · {ds} vs the rest", body,
                f"One-against-all · pool: {e(', '.join([ds] + peers))} · "
                f"generated {e(m['generated_utc'])}")


# ---------------------------------------------------------------- index

def render_index(profiles, compares):
    rows = []
    for ds in sorted(profiles):
        p, c = profiles[ds], compares.get(ds)
        o = c["overview"] if c else {}
        rows.append((
            f"<b>{e(ds)}</b>", f"<span class=tag>{e(p['meta']['role'])}</span>",
            fnum(p["meta"]["rows"]),
            pct(o.get("attack_ratio_self")) if c else "–",
            str(len(p["by_class"])),
            fnum(o.get("domain_fingerprint")) if c else "–",
            fnum(o.get("concept_shift")) if c else "–",
            f"{o.get('n_domain_biased', '–')} / {o.get('n_transfers', '–')}",
            f"<a href='profile_{e(ds)}.html'>profile</a> · "
            f"<a href='compare_{e(ds)}.html'>vs rest</a>",
        ))
    body = f"""<div class=note>Each dataset has an individual profile and a one-against-all
comparison. The comparisons are derived from the profiles alone, so adding a dataset
re-derives every comparison in seconds without re-reading the lake.</div>
{table(['dataset', 'role', 'flows', 'attack', 'classes', 'domain fingerprint',
        'concept shift', 'biased / transferable', 'reports'], rows)}"""
    return page("Talos · EDA reports", body,
                f"{len(profiles)} dataset(s) in the pool")


def render_all(directory=EDA_DIR):
    """Rebuild every page from whatever JSON is on disk. Returns paths written."""
    directory = Path(directory)
    written = []
    profiles, compares = {}, {}
    for path in sorted(directory.glob("profile_*.json")):
        doc = json.loads(path.read_text())
        ds = doc["meta"]["dataset"]
        profiles[ds] = doc
        out = directory / f"profile_{ds}.html"
        out.write_text(render_profile(doc))
        written.append(out)
    for path in sorted(directory.glob("compare_*.json")):
        doc = json.loads(path.read_text())
        ds = doc["meta"]["dataset"]
        compares[ds] = doc
        out = directory / f"compare_{ds}.html"
        out.write_text(render_compare(doc))
        written.append(out)
    if profiles:
        idx = directory / "index.html"
        idx.write_text(render_index(profiles, compares))
        written.append(idx)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=str(EDA_DIR))
    a = p.parse_args()
    written = render_all(a.dir)
    if not written:
        sys.exit(f"no EDA JSON in {a.dir} — run `make eda-all` first")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
