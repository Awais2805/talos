#!/usr/bin/env python3
"""Turn adjudicated flows into a measured per-class precision, with an interval.

This is the file that lets the project state an accuracy figure instead of
asserting one. Everything upstream reasons from the attack schedule the labels
were derived from, so nothing upstream can be evidence that the labels are
right -- it would be the document checking itself. A person looking at flows and
saying yes or no is outside that loop, and this reads what they said.

What is measured is PRECISION, per class: of the flows we labelled C, the share
an adjudicator confirms really are C. Not recall -- a sample of what we labelled
says nothing about what we missed, and the flows we called benign by closed-world
default are the ones most likely to hide missed attacks. The benign stratum in
the sample is what carries information about that, and it is reported as its own
class rather than folded into an overall figure.

Four decisions here are what make the number defensible rather than flattering:

  * Wilson score intervals, not the normal approximation. At p = 1.0 with n = 400
    the normal interval is [1.0, 1.0] -- it claims perfection with no uncertainty,
    which is exactly the region a label-quality audit lives in. Wilson gives
    [0.9905, 1.0]: the lower bound n/(n+z^2) is the honest statement, and it is
    the one that supports or refuses a "99%" claim.
  * Cohen's kappa with the observed and expected agreement reported beside it.
    On a dataset where one class is 46% of the flows, a rater who always said
    "correct" would score high raw agreement. Kappa discounts that, and printing
    p_o and p_e separately shows how much of the kappa is the discount.
  * `uncertain` is excluded from precision and reported separately. Folding an
    undecidable flow in as either a hit or a miss invents a measurement that was
    not taken; excluding it silently hides how much of the sample was
    undecidable. These are the flows the quarantine concept exists for.
  * Coverage is measured and gates the run. A half-finished sample scored as if
    complete is the single easiest way to produce a wrong number here, so the
    unadjudicated rows are counted, printed, and made to fail the exit code.

Extrapolation to the dataset is reweighted by the true class prior. The sample
is stratified with equal quotas -- 400 benign out of 226,200 and 198 portscan
out of 198 -- so the unweighted mean over sampled flows is a statement about a
population that does not exist. Every population figure below is
sum_c (N_c / N) * p_c, and the interval on it comes from the stratified variance
with a finite-population correction, which is what makes a class adjudicated in
full contribute no sampling error.

Usage:
    python src/label_validation/certify/score.py --dataset cic-ids-2017
    python src/label_validation/certify/score.py --dataset toy --sample /tmp/toy.csv
    python src/label_validation/certify/score.py --dataset toy --lake-root /tmp/fixture --no-s3
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "config.yml").is_file())
sys.path.insert(0, str(ROOT))

from src.common.lake import Lake                                    # noqa: E402
from src.label_validation.core import paths as _paths  # noqa: E402
from src.label_validation.runner import load_config, probe_sources          # noqa: E402

REPORTS = ROOT / "reports" / "validation"
SAMPLES = REPORTS / "samples"
POLICY = _paths.POLICY

# Two-sided normal quantiles. A table rather than an inverse-CDF because scipy is
# not a dependency of this stage and three levels is every level anyone asks for;
# an unlisted level is refused rather than silently rounded to 95%.
Z = {0.80: 1.2815515655446004, 0.90: 1.6448536269514722,
     0.95: 1.959963984540054, 0.99: 2.5758293035489004}

# The adjudication vocabulary. Kept deliberately forgiving on spelling and case,
# because it is typed into a spreadsheet by hand -- but NOT open-ended: anything
# unrecognised is counted and reported, never quietly treated as a miss or a
# pass. A verdict nobody can interpret is a hole in the sample, and holes have to
# be visible.
CONFIRM = {"correct", "c", "yes", "y", "ok", "true", "t", "confirm", "confirmed",
           "right", "1"}
REJECT = {"wrong", "w", "no", "n", "false", "f", "incorrect", "mislabelled",
          "mislabeled", "bad", "0"}
UNCERTAIN = {"uncertain", "unsure", "unknown", "?", "ambiguous", "maybe",
             "undecidable", "skip", "tbd", "quarantine"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--sample", default=None,
                   help="adjudicated CSV (default: reports/validation/samples/<dataset>.csv)")
    p.add_argument("--frame", default=None,
                   help="sampling frame JSON (default: the sample's .frame.json sidecar)")
    p.add_argument("--confidence", type=float, default=None,
                   help=f"interval level; one of {sorted(Z)} (default: policy, or 0.95)")
    p.add_argument("--out", default=None,
                   help="default: reports/validation/certification_<dataset>.json")
    p.add_argument("--config", default=str(ROOT / "config.yml"))
    p.add_argument("--policy", default=str(POLICY))
    # Only used when the frame sidecar is missing or carries no populations: the
    # class prior has to come from somewhere, and a guess is not an option.
    p.add_argument("--lake-root", default=None,
                   help="read parquet from this root instead of s3://<bucket>")
    p.add_argument("--no-s3", action="store_true")
    p.add_argument("--memory-limit", default="8GB")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--temp-dir", default=None)
    p.add_argument("--max-temp", default="20GB")
    return p.parse_args()


def threshold(policy, key, default=None):
    """policy.yaml read directly -- this is a CLI, not a registered check."""
    return (policy.get("thresholds") or {}).get(key, default)


# ------------------------------------------------------------------ the maths

def wilson(k, n, z):
    """Wilson (1927) score interval for a binomial proportion.

    Solves |p_hat - p| = z * sqrt(p(1-p)/n) for p, i.e. it inverts the score
    test rather than assuming the estimate is normally distributed around
    itself:

        centre = (p_hat + z^2/2n) / (1 + z^2/n)
        half   = z/(1 + z^2/n) * sqrt( p_hat(1-p_hat)/n + z^2/(4n^2) )

    The normal (Wald) interval p_hat +/- z*sqrt(p_hat(1-p_hat)/n) is wrong in
    exactly the regime this file operates in: at p_hat = 1 its width is zero, so
    400 clean adjudications would "prove" 100% precision, and near p_hat = 1 its
    actual coverage is far below the nominal 95%. Wilson stays inside [0, 1],
    never collapses, and at k = n reduces to a lower bound of n/(n + z^2) --
    99.05% for n = 400, which is the honest form of "we saw no errors".

    Returns (point estimate, low, high), or (None, None, None) for n = 0.
    """
    if n <= 0:
        return None, None, None
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def cohen_kappa(ours, theirs):
    """Agreement between our label and the adjudicated class, chance discounted.

        p_o    = share of flows where the two agree
        p_e    = sum_i P(ours = i) * P(theirs = i), the agreement two raters
                 with these marginals would reach by chance alone
        kappa  = (p_o - p_e) / (1 - p_e)

    Both p_o and p_e are returned, not just kappa, because they are the only way
    to see what a given kappa is made of. A stratified sample flattens the class
    prior, which drives p_e down and kappa up relative to the same adjudication
    on a population sample -- so kappa here is a statement about the sample's
    class mix, and p_e is the number that says by how much.

    Only flows where the adjudicator NAMED a class can appear: a bare "wrong"
    says our label is not C without saying what is, and a rater who has not
    named a category cannot be cross-tabulated against one.
    """
    labels = sorted(set(ours) | set(theirs))
    if not labels or len(ours) == 0:
        return None
    idx = {c: i for i, c in enumerate(labels)}
    m = len(labels)
    cm = np.zeros((m, m), dtype=np.int64)
    for a, b in zip(ours, theirs):
        cm[idx[a], idx[b]] += 1
    n = cm.sum()
    p_o = float(np.trace(cm)) / n
    p_e = float((cm.sum(axis=1) / n) @ (cm.sum(axis=0) / n))
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else None
    return {"n": int(n), "labels": labels,
            "observed_agreement": round(p_o, 6),
            "expected_agreement": round(p_e, 6),
            "kappa": None if kappa is None else round(kappa, 6),
            "confusion": {a: {b: int(cm[i, j]) for j, b in enumerate(labels) if cm[i, j]}
                          for i, a in enumerate(labels)}}


def stratified(per_class, populations, z):
    """Combine per-class precisions into one population figure, correctly.

    The draw gives every class the same quota, so the sample's class mix is not
    the dataset's: taking a plain mean over sampled flows would weight 198
    portscan flows as heavily as 226,200 benign ones. The population estimate is
    therefore sum_c w_c * p_c with w_c = N_c / N, and its variance is the
    stratified one,

        Var = sum_c w_c^2 * p_c(1-p_c)/n_c * (N_c - n_c)/(N_c - 1)

    including the finite-population correction, which is what makes a class
    adjudicated in full (n_c = N_c) contribute exactly zero sampling error
    rather than a fictitious amount.

    Two intervals come back. `normal` is z*sqrt(Var) around the point estimate --
    Wilson has no stratified form, so this is the estimate's own interval and is
    labelled as the approximation it is. `envelope` is sum_c w_c * [lo_c, hi_c],
    the per-class Wilson bounds carried through the same weighting: wider,
    conservative, and it does not collapse when a class measured p = 1. Where
    they disagree, believe the envelope.
    """
    usable = [r for r in per_class if r["precision"] is not None
              and populations.get(r["class"], 0) > 0]
    total = sum(populations.get(r["class"], 0) for r in usable)
    if not usable or not total:
        return None
    point = var = lo = hi = 0.0
    for r in usable:
        w = populations[r["class"]] / total
        n, N, p = r["decided"], populations[r["class"]], r["precision"]
        fpc = max(0.0, (N - n) / (N - 1)) if N > 1 else 0.0
        point += w * p
        var += w * w * (p * (1 - p) / n) * fpc if n else 0.0
        lo += w * r["ci_low"]
        hi += w * r["ci_high"]
    se = float(np.sqrt(var))
    return {
        "classes": [r["class"] for r in usable],
        "population_covered": total,
        "precision": round(point, 6),
        "standard_error": round(se, 6),
        "normal_low": round(max(0.0, point - z * se), 6),
        "normal_high": round(min(1.0, point + z * se), 6),
        "envelope_low": round(lo, 6), "envelope_high": round(hi, 6),
        "finite_population_correction": True,
    }


# ------------------------------------------------------------- reading inputs

def read_sample(path):
    """The adjudicated CSV, straight from the spreadsheet it was edited in.

    utf-8-sig, because Excel writes a BOM and would otherwise leave the first
    header as '\\ufeffverdict' -- every verdict in the file would read as blank
    and the sample would score as entirely unadjudicated.
    """
    p = Path(path)
    if not p.exists():
        sys.exit(f"no adjudicated sample at {p} — run src/label_validation/certify/sample.py first")
    with p.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = reader.fieldnames or []
    missing = [c for c in ("uid", "label_class", "verdict") if c not in fields]
    if missing:
        sys.exit(f"{p} is missing required column(s): {', '.join(missing)}")
    return rows, fields


def vocabulary(rows, populations):
    """The class names a verdict may name, lowercase -> canonical spelling.

    Drawn from the sample AND the population, so an adjudicator may reassign a
    flow to a class that happens not to appear in the sample's own label column.
    benign is always admissible: it is the verdict that matters most, being the
    one that says a scheduled attack window caught ordinary traffic.
    """
    names = ({str(r["label_class"]).strip() for r in rows}
             | set(populations) | {"benign"})
    return {c.lower(): c for c in names if c}


def resolve(verdict, label_class, known):
    """One typed verdict -> (state, the class the adjudicator says this really is).

    States:
      blank        nobody has looked at this flow yet. Counts against coverage,
                   never against precision.
      confirmed    the label is right; true class = label_class.
      rejected     the label is wrong, but no class was named. Counts as an
                   error for precision; cannot enter the kappa table, since
                   "not C" is not a category.
      reassigned   a class name was typed; true class = that class. This is the
                   most useful verdict, because it says where the label went.
      uncertain    undecidable from the evidence. Excluded from precision and
                   reported on its own -- these are quarantine candidates.
      unrecognised something else was typed. Reported loudly rather than
                   guessed at.
    """
    v = (verdict or "").strip().lower()
    if not v:
        return "blank", None
    if v in CONFIRM:
        return "confirmed", label_class
    if v in UNCERTAIN:
        return "uncertain", None
    if v in known:
        return ("confirmed" if v == str(label_class).strip().lower()
                else "reassigned", known[v])
    if v in REJECT:
        return "rejected", None
    return "unrecognised", None


def populations_from_lake(a, cfg):
    """Class prior straight from the labelled zone, when no frame sidecar exists.

    The frame is the right source -- it records the population as it was at draw
    time, which is what the sample is a sample OF. Reading the lake now is the
    fallback, and it is recorded as such in the output, because a lake that has
    been re-labelled since the draw would silently reweight the result.
    """
    bucket, region = cfg["aws"]["bucket"], cfg["aws"]["region"]
    root = a.lake_root.rstrip("/") if a.lake_root else f"s3://{bucket}"
    require_s3 = not a.no_s3 and root.startswith("s3://")
    lake = Lake(region, a.memory_limit, a.temp_dir, a.threads, a.max_temp,
                require_s3=require_s3)
    sources, available = probe_sources(lake, root, a.dataset, cfg, require_s3)
    if "labelled" not in available:
        sys.exit(f"no sampling frame and no labelled zone under {root} — "
                 f"cannot weight the sample to the population")
    rows = lake.sql(f"SELECT label_class, count(*) FROM "
                    f"read_parquet('{sources['labelled']}', union_by_name=true) "
                    f"GROUP BY 1")
    return {str(r[0]): int(r[1]) for r in rows}, root


# ----------------------------------------------------------------- the report

def score(rows, populations, z):
    """Per-class counts, precision and interval, plus the kappa inputs.

    Every flow is counted twice: once for the class as a whole, and once more
    only if the labelling stage did NOT quarantine it. `label_quality` is
    written by label_flows.py -- 'uncertain' where a flow falls in a declared
    quarantine region, 'certain' otherwise -- and the second figure is the one
    that answers "how good are the labels we intend to build on", as opposed to
    "how good are the labels including the ones we already disowned". Where the
    column is absent (a sample drawn before quarantine existed) the second
    figure is simply not produced, rather than silently equalling the first.
    """
    known = vocabulary(rows, populations)
    classes = sorted({str(r["label_class"]).strip() for r in rows})
    fresh = {"sampled": 0, "blank": 0, "confirmed": 0, "rejected": 0,
             "reassigned": 0, "uncertain": 0, "unrecognised": 0}
    buckets = {c: dict(fresh, reassigned_to={}, unrecognised_verdicts={},
                       quarantined=0, certain=dict(fresh),
                       quarantine_reasons={}) for c in classes}
    ours, theirs, unrecognised_rows = [], [], []
    has_quality = any("label_quality" in r for r in rows[:1])

    for r in rows:
        cls = str(r["label_class"]).strip()
        state, truth = resolve(r.get("verdict"), cls, known)
        b = buckets[cls]
        b["sampled"] += 1
        b[state] += 1
        quarantined = str(r.get("label_quality") or "").strip().lower() == "uncertain"
        if quarantined:
            b["quarantined"] += 1
            why = (r.get("quarantine_reason") or "").strip() or "(unnamed region)"
            b["quarantine_reasons"][why] = b["quarantine_reasons"].get(why, 0) + 1
        else:
            b["certain"]["sampled"] += 1
            b["certain"][state] += 1
        if state == "reassigned":
            b["reassigned_to"][truth] = b["reassigned_to"].get(truth, 0) + 1
        if state == "unrecognised":
            v = (r.get("verdict") or "").strip()
            b["unrecognised_verdicts"][v] = b["unrecognised_verdicts"].get(v, 0) + 1
            unrecognised_rows.append({"uid": r.get("uid"), "label_class": cls,
                                      "verdict": v})
        if truth is not None:
            ours.append(cls)
            theirs.append(truth)

    per_class = []
    for c in classes:
        b = buckets[c]
        # Decided = adjudicated, minus the ones that produced no decision.
        # `rejected` counts here as a miss even though it names no replacement:
        # the adjudicator said this is not C, which is all precision asks.
        adjudicated = b["sampled"] - b["blank"]
        decided = b["confirmed"] + b["rejected"] + b["reassigned"]
        p, lo, hi = wilson(b["confirmed"], decided, z)
        N = populations.get(c, 0)
        rec = {
            "class": c, "sampled": b["sampled"], "adjudicated": adjudicated,
            "blank": b["blank"], "uncertain": b["uncertain"],
            "unrecognised": b["unrecognised"], "decided": decided,
            "confirmed": b["confirmed"], "rejected": b["rejected"],
            "reassigned": b["reassigned"], "reassigned_to":
                dict(sorted(b["reassigned_to"].items(), key=lambda kv: -kv[1])),
            "unrecognised_verdicts": b["unrecognised_verdicts"],
            "coverage": round(adjudicated / b["sampled"], 6) if b["sampled"] else None,
            "uncertain_share": round(b["uncertain"] / adjudicated, 6) if adjudicated else None,
            "precision": None if p is None else round(p, 6),
            "ci_low": None if lo is None else round(lo, 6),
            "ci_high": None if hi is None else round(hi, 6),
            "population": N,
            "population_share": round(N / sum(populations.values()), 6)
            if populations and sum(populations.values()) else None,
            "census": bool(N and b["sampled"] >= N),
        }
        if p is not None and N:
            # The figure a reader actually wants: not "97% precise" but "between
            # this many and this many flows in this class carry the wrong label".
            rec["mislabelled_estimate"] = int(round((1 - p) * N))
            rec["mislabelled_low"] = int(round((1 - hi) * N))
            rec["mislabelled_high"] = int(round((1 - lo) * N))
        if has_quality:
            q = b["certain"]
            q_decided = q["confirmed"] + q["rejected"] + q["reassigned"]
            qp, qlo, qhi = wilson(q["confirmed"], q_decided, z)
            rec["quarantined"] = b["quarantined"]
            rec["quarantine_reasons"] = dict(
                sorted(b["quarantine_reasons"].items(), key=lambda kv: -kv[1]))
            rec["excluding_quarantined"] = {
                "sampled": q["sampled"], "decided": q_decided,
                "confirmed": q["confirmed"], "uncertain": q["uncertain"],
                "precision": None if qp is None else round(qp, 6),
                "ci_low": None if qlo is None else round(qlo, 6),
                "ci_high": None if qhi is None else round(qhi, 6),
            }
        per_class.append(rec)

    return per_class, cohen_kappa(ours, theirs), unrecognised_rows


def pct(v, width=8):
    return f"{v:>{width}.2%}" if v is not None else f"{'-':>{width}}"


def main():
    a = parse_args()
    cfg = load_config(a.config)
    policy = yaml.safe_load(Path(a.policy).read_text()) or {}

    conf = a.confidence or float(threshold(policy, "certification_confidence", 0.95))
    if conf not in Z:
        sys.exit(f"--confidence {conf} not tabulated; use one of {sorted(Z)}")
    z = Z[conf]
    min_cov = float(threshold(policy, "certification_min_coverage", 0.95))
    min_prec = float(threshold(policy, "certification_min_precision", 0.99))
    max_unc = float(threshold(policy, "certification_max_uncertain_share", 0.10))

    sample_path = Path(a.sample) if a.sample else SAMPLES / f"{a.dataset}.csv"
    frame_path = Path(a.frame) if a.frame else sample_path.with_suffix(".frame.json")
    rows, _fields = read_sample(sample_path)

    frame = json.loads(frame_path.read_text()) if frame_path.exists() else None
    populations = (frame or {}).get("population") or {}
    prior_source = str(frame_path)
    lake_root = None
    if not populations:
        populations, lake_root = populations_from_lake(a, cfg)
        prior_source = f"lake ({lake_root}) — no frame sidecar at {frame_path}"

    per_class, kappa, unrecognised = score(rows, populations, z)

    sampled_classes = {r["class"] for r in per_class}
    # A class in the dataset that the sample never reached cannot be certified,
    # and its flows must not be swept into the population figure as if it had.
    unsampled = {c: n for c, n in populations.items() if c not in sampled_classes}
    unknown_pop = [r["class"] for r in per_class if not r["population"]]

    pop_total = sum(populations.values())
    strat = stratified(per_class, populations, z)

    total_sampled = sum(r["sampled"] for r in per_class)
    total_adjud = sum(r["adjudicated"] for r in per_class)
    total_decided = sum(r["decided"] for r in per_class)
    total_confirmed = sum(r["confirmed"] for r in per_class)
    total_uncertain = sum(r["uncertain"] for r in per_class)
    total_blank = sum(r["blank"] for r in per_class)
    coverage = total_adjud / total_sampled if total_sampled else 0.0
    unc_share = total_uncertain / total_adjud if total_adjud else 0.0
    pooled = wilson(total_confirmed, total_decided, z)

    # The labelling stage's own uncertainty, measured against the adjudicator's.
    # Two different numbers wear the word "uncertain" here and conflating them
    # would be the easiest mistake in this file: `quarantined` is what the
    # manifests declared in advance, `uncertain` is what a person could not
    # decide from the evidence. Whether they pick out the same flows is the
    # question that says if the quarantine regions are drawn in the right place.
    quarantine = None
    if any("quarantined" in r for r in per_class):
        declared = sum(r.get("quarantined", 0) for r in per_class)
        both = sum(min(r.get("quarantined", 0), r["uncertain"]) for r in per_class)
        quarantine = {
            "sampled_quarantined": declared,
            "share_of_sample": round(declared / total_sampled, 6) if total_sampled else None,
            "adjudicator_uncertain": total_uncertain,
            "reasons": {k: v for r in per_class
                        for k, v in (r.get("quarantine_reasons") or {}).items()},
            "note": "`quarantined` is declared by the manifest before anyone "
                    "looked; `uncertain` is what the adjudicator could not "
                    "decide. The per-class `excluding_quarantined` block is the "
                    "figure to publish -- precision over the flows the pipeline "
                    "still stands behind. Its population cannot be reweighted "
                    "here: the frame records each class's total, not how much "
                    "of it a quarantine region covers.",
            "upper_bound_overlap": both,
        }

    certified = [r["class"] for r in per_class
                 if r["ci_low"] is not None and r["ci_low"] >= min_prec]
    not_certified = [r["class"] for r in per_class
                     if r["ci_low"] is not None and r["ci_low"] < min_prec]
    # "Valid" is about the measurement, not the result: a sample can be complete,
    # decisive and unambiguous and still report a low precision. Refusing to
    # build on a low figure is gate.py's job; refusing to publish a figure drawn
    # from an unfinished sample is this one's.
    valid = (coverage >= min_cov and not unrecognised and unc_share <= max_unc)

    report = {
        "certification_version": 1,
        "meta": {
            "dataset": a.dataset,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sample": str(sample_path), "frame": str(frame_path) if frame else None,
            "prior_source": prior_source,
            "manifest_sha": ((frame or {}).get("meta") or {}).get("manifest_sha"),
            "taxonomy_sha": ((frame or {}).get("meta") or {}).get("taxonomy_sha"),
            "sample_drawn_utc": ((frame or {}).get("meta") or {}).get("generated_utc"),
            "sampling": (frame or {}).get("sampling"),
            "policy_version": policy.get("version"),
            "confidence": conf, "z": z,
            "interval_method": "Wilson score interval (per class); stratified "
                               "normal interval with finite-population "
                               "correction (population figure)",
        },
        "coverage": {
            "sampled": total_sampled, "adjudicated": total_adjud,
            "blank": total_blank, "coverage": round(coverage, 6),
            "minimum_required": min_cov,
            "complete": total_blank == 0,
            "sufficient": coverage >= min_cov,
        },
        "exclusions": {
            "uncertain": total_uncertain,
            "uncertain_share_of_adjudicated": round(unc_share, 6),
            "uncertain_share_max": max_unc,
            "unrecognised": len(unrecognised),
            "unrecognised_rows": unrecognised[:50],
            "note": "flows marked uncertain are excluded from every precision "
                    "figure and from kappa; they are undecidable from the "
                    "evidence exported, not evidence of a correct label, and "
                    "are the population a quarantine region would cover",
        },
        "per_class": per_class,
        "sample_pooled": {
            "decided": total_decided, "confirmed": total_confirmed,
            "precision": None if pooled[0] is None else round(pooled[0], 6),
            "ci_low": None if pooled[1] is None else round(pooled[1], 6),
            "ci_high": None if pooled[2] is None else round(pooled[2], 6),
            "note": "unweighted over the stratified sample -- NOT a statement "
                    "about the dataset; the sample deliberately over-represents "
                    "small classes. Use `population` for that.",
        },
        "population": strat,
        "population_total": pop_total,
        "quarantine": quarantine,
        "unsampled_classes": unsampled,
        "classes_without_population": unknown_pop,
        "agreement": kappa,
        "certification": {
            "precision_floor": min_prec,
            "certified": certified, "not_certified": not_certified,
            "undecided": [r["class"] for r in per_class if r["ci_low"] is None],
            "measurement_valid": valid,
        },
    }

    out = Path(a.out) if a.out else REPORTS / f"certification_{a.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")

    # ---- console -----------------------------------------------------------
    print(f"dataset     {a.dataset}")
    print(f"sample      {sample_path}   {total_sampled:,} row(s)")
    print(f"prior       {prior_source}")
    print(f"confidence  {conf:.0%}   z = {z:.6f}   Wilson score interval")
    if frame:
        print(f"drawn       {report['meta']['sample_drawn_utc']}   manifest "
              f"{report['meta']['manifest_sha']}")

    head = (f"{'class':<14}{'sampled':>8}{'adjud':>7}{'cover':>8}{'uncert':>7}"
            f"{'decided':>8}{'correct':>8}{'precision':>11}"
            f"{'  ' + str(int(conf * 100)) + '% Wilson CI':<22}"
            f"{'population':>13}{'prior':>8}   mislabelled (est)")
    print(f"\n{head}")
    print("-" * len(head))
    for r in sorted(per_class, key=lambda r: -(r["population"] or 0)):
        ci = (f"  [{r['ci_low']:.2%}, {r['ci_high']:.2%}]"
              if r["ci_low"] is not None else "  (nothing decided)")
        est = (f"{r['mislabelled_estimate']:,} [{r['mislabelled_low']:,}, "
               f"{r['mislabelled_high']:,}]" if "mislabelled_estimate" in r else "-")
        print(f"{r['class']:<14}{r['sampled']:>8,}{r['adjudicated']:>7,}"
              f"{pct(r['coverage'], 8)}{r['uncertain']:>7,}{r['decided']:>8,}"
              f"{r['confirmed']:>8,}{pct(r['precision'], 11)}{ci:<22}"
              f"{r['population']:>13,}{pct(r['population_share'], 8)}   {est}"
              + ("   census" if r["census"] else ""))

    if pooled[0] is not None:
        print(f"\n{'sample pooled':<14}{total_sampled:>8,}{total_adjud:>7,}"
              f"{pct(coverage, 8)}{total_uncertain:>7,}{total_decided:>8,}"
              f"{total_confirmed:>8,}{pct(pooled[0], 11)}"
              f"  [{pooled[1]:.2%}, {pooled[2]:.2%}]"
              "   <- unweighted; not the dataset figure")
    if strat:
        print(f"\npopulation-weighted precision   {strat['precision']:.4%}"
              f"   normal [{strat['normal_low']:.4%}, {strat['normal_high']:.4%}]"
              f"   SE {strat['standard_error']:.5f}")
        print(f"  conservative Wilson envelope   "
              f"[{strat['envelope_low']:.4%}, {strat['envelope_high']:.4%}]"
              f"   over {strat['population_covered']:,} of {pop_total:,} flow(s)")
        print("  reweighted by the true class prior: the draw is stratified, so "
              "the pooled\n  sample figure above describes a population that "
              "does not exist.")

    if kappa:
        print(f"\nagreement (our label vs the adjudicated class, over the "
              f"{kappa['n']:,} flow(s)\n"
              f"           where the adjudicator named a class)")
        print(f"  observed agreement  {kappa['observed_agreement']:.4f}")
        print(f"  expected by chance  {kappa['expected_agreement']:.4f}"
              f"   (from the two raters' marginals)")
        k = kappa["kappa"]
        print(f"  Cohen's kappa       {k:.4f}" if k is not None
              else "  Cohen's kappa       undefined (chance agreement is 1)")
    unnamed = sum(r["rejected"] for r in per_class)
    if unnamed:
        print(f"  excluded from kappa {unnamed:,} flow(s) marked wrong without "
              f"naming a class\n                      (counted as errors in "
              f"precision, where naming is not needed)")

    print(f"\ncoverage    {total_adjud:,} of {total_sampled:,} adjudicated "
          f"({coverage:.1%}); {total_blank:,} left blank"
          + ("" if coverage >= min_cov else
             f"   << BELOW THE {min_cov:.0%} FLOOR"))
    print(f"excluded    {total_uncertain:,} uncertain ({unc_share:.1%} of "
          f"adjudicated), {len(unrecognised)} unrecognised verdict(s)")
    if quarantine:
        print(f"quarantine  {quarantine['sampled_quarantined']:,} sampled flow(s) "
              f"were already quarantined by the manifests")
        for r in sorted(per_class, key=lambda r: -(r["population"] or 0)):
            q = r.get("excluding_quarantined")
            if not q or q["precision"] is None:
                continue
            print(f"            {r['class']:<12} excluding them: "
                  f"{q['precision']:.2%} over {q['decided']:,} decided "
                  f"[{q['ci_low']:.2%}, {q['ci_high']:.2%}]")
    if unrecognised:
        seen = sorted({r["verdict"] for r in unrecognised})[:8]
        print(f"            !! not scored: {', '.join(repr(s) for s in seen)}")
    if unsampled:
        print(f"unsampled   {len(unsampled)} class(es) absent from the sample "
              f"and excluded from the population figure: "
              f"{', '.join(sorted(unsampled))}")

    print(f"\ncertified at a {min_prec:.0%} precision floor (Wilson lower bound "
          f">= {min_prec:.0%}):")
    print(f"  yes   {', '.join(certified) or '(none)'}")
    print(f"  no    {', '.join(not_certified) or '(none)'}")
    print(f"\nreport -> {out}")
    if not valid:
        print("\n!! the sample does not support a published figure: "
              + "; ".join(filter(None, [
                  f"coverage {coverage:.1%} < {min_cov:.0%}" if coverage < min_cov else "",
                  f"{len(unrecognised)} unrecognised verdict(s)" if unrecognised else "",
                  f"uncertain {unc_share:.1%} > {max_unc:.0%}"
                  if unc_share > max_unc else ""])), file=sys.stderr)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
