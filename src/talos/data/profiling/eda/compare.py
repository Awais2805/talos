#!/usr/bin/env python3
"""One-against-all comparison, computed from the per-dataset profiles alone.

This module never opens a parquet file, never contacts S3, and does not import
duckdb. Everything it needs -- pooled means, variances, correlation matrices
and distributions of "every OTHER dataset in the pool" -- is reconstructed
exactly from the additive statistics profile_dataset.py already wrote:

    n, Σx, Σx², Σ(x·y), and fixed-grid histograms

Sums add, so the pool of the rest is the elementwise sum of their profiles; the
correlation of the pool follows from those sums exactly, not approximately.
That is the inverse dependency the pipeline is built around: adding a dataset
regenerates N comparison reports in seconds, from N JSON files, instead of
re-scanning hundreds of millions of flows.

The one thing that would break it is two profiles measured on different rulers,
so every profile carries the spec hash and this stage refuses to mix hashes.

Usage:
    python -m talos.eda.compare                    # regenerate all comparisons
    python -m talos.eda.compare --dataset cic-ids-2017
    python -m talos.eda.compare --dir reports/eda
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from talos.common.paths import eda_dir
from talos.data.profiling.eda.spec import Spec

EDA_DIR = eda_dir()
COMPARE_VERSION = 2
LABEL_COL = "label_binary"
EPS = 1e-12


# --------------------------------------------------------------- loading

def load_profiles(directory):
    """Every profile in the pool, keyed by dataset, with the ruler checked.

    A stale profile is a silent wrong answer -- its histograms would be on
    different bin edges -- so a spec mismatch stops the run and names the files
    that need re-profiling.
    """
    profiles = {}
    for path in sorted(Path(directory).glob("profile_*.json")):
        doc = json.loads(path.read_text())
        profiles[doc["meta"]["dataset"]] = doc
    if not profiles:
        return profiles

    # Comparability is checked against the ruler each profile CARRIES -- its own
    # transforms and bin edges -- not against a hash of the spec file. A hash
    # also moves when a comment changes or a new optional statistic is added,
    # which would force a re-scan of 137M flows to answer a question that the
    # profiles can answer themselves. This only objects when a bin edge or a
    # transform actually differs, and says which feature.
    spec, bad = Spec(), []
    want = spec.contract()
    for ds, p in profiles.items():
        got = spec.contract(p["numeric_spec"])
        for f, cfg in got.items():
            if f in want and want[f] != cfg:
                bad.append(f"{ds}: {f} was measured as {cfg[0]} on {len(cfg[1])} edges, "
                           f"spec.yaml now says {want[f][0]} on {len(want[f][1])}")
    if bad:
        sys.exit("these profiles were measured on a different ruler:\n  " +
                 "\n  ".join(bad) + "\nre-run `make eda-all` — comparing across bin "
                 "edges is meaningless, so this is refused rather than approximated.")
    return profiles


# --------------------------------------------------- additive class views

def _blank(feature_names, pair_names):
    return {"n": 0,
            "numeric": {f: {"n_undefined": 0, "n_zero": 0, "min": None, "max": None,
                            "sum_t": 0.0, "sumsq_t": 0.0,
                            "hist": None} for f in feature_names},
            "categorical": {},
            # Pearson rides on the co-moment sums; the rank matrix rides on the
            # joint bin counts. Both are additive, so both survive pooling.
            "pairs": {p: 0.0 for p in pair_names},
            "joint": {p: {} for p in pair_names}}


def _add_numeric(dst, src):
    dst["n_undefined"] += src["n_undefined"]
    dst["n_zero"] += src["n_zero"]
    dst["sum_t"] += src["sum_t"]
    dst["sumsq_t"] += src["sumsq_t"]
    dst["hist"] = ([int(x) for x in src["hist"]] if dst["hist"] is None
                   else [a + b for a, b in zip(dst["hist"], src["hist"])])
    for key, op in (("min", min), ("max", max)):
        if src[key] is not None:
            dst[key] = src[key] if dst[key] is None else op(dst[key], src[key])


def class_view(profile, which):
    """Collapse a profile's per-class statistics into one additive block.

    `which` is 'all', 'benign', or 'attack'. Because every heavy statistic was
    stored per label_class, these three views cost a sum over ~10 dicts -- and
    they are what separates DOMAIN shift (benign here vs benign there) from
    CONCEPT shift (attack here vs attack there), which a single overall
    histogram silently mixes together.
    """
    benign = profile["benign_class"]
    wanted = [c for c in profile["by_class"]
              if which == "all" or (c == benign) == (which == "benign")]
    feats = list(profile["numeric_spec"])
    pairs = [tuple(p) for p in profile["pair_order"]]
    view = _blank(feats, pairs)
    for cls in wanted:
        blk = profile["by_class"][cls]
        view["n"] += blk["n"]
        for f in feats:
            if f in blk["numeric"]:
                _add_numeric(view["numeric"][f], blk["numeric"][f])
        for i, p in enumerate(pairs):
            if i < len(blk["pairs_xy"]):
                view["pairs"][p] += blk["pairs_xy"][i]
            if i < len(blk.get("pairs_joint", [])):
                tgt = view["joint"][p]
                for cell, cnt in blk["pairs_joint"][i].items():
                    tgt[cell] = tgt.get(cell, 0) + cnt
        for cat, cblk in blk["categorical"].items():
            tgt = view["categorical"].setdefault(cat, {"counts": {}, "n_other": 0,
                                                       "n_distinct": 0})
            for val, cnt in cblk["counts"].items():
                tgt["counts"][val] = tgt["counts"].get(val, 0) + cnt
            tgt["n_other"] += cblk["n_other"]
            tgt["n_distinct"] = max(tgt["n_distinct"], cblk["n_distinct"])
    return view


def pool(views):
    """Sum several datasets' views into the 'rest of the pool' view."""
    if not views:
        return None
    feats = sorted(set().union(*(set(v["numeric"]) for v in views)))
    pairs = sorted(set().union(*(set(v["pairs"]) for v in views)))
    out = _blank(feats, pairs)
    for v in views:
        out["n"] += v["n"]
        for f in feats:
            if f in v["numeric"] and v["numeric"][f]["hist"] is not None:
                _add_numeric(out["numeric"][f], v["numeric"][f])
        for p in pairs:
            out["pairs"][p] += v["pairs"].get(p, 0.0)
            tgt = out["joint"].setdefault(p, {})
            for cell, cnt in v.get("joint", {}).get(p, {}).items():
                tgt[cell] = tgt.get(cell, 0) + cnt
        for cat, cblk in v["categorical"].items():
            tgt = out["categorical"].setdefault(cat, {"counts": {}, "n_other": 0,
                                                      "n_distinct": 0})
            for val, cnt in cblk["counts"].items():
                tgt["counts"][val] = tgt["counts"].get(val, 0) + cnt
            tgt["n_other"] += cblk["n_other"]
            tgt["n_distinct"] = max(tgt["n_distinct"], cblk["n_distinct"])
    return out


# ------------------------------------------------------------- statistics

def js_divergence(p, q):
    """Jensen-Shannon divergence in bits: 0 = identical, 1 = disjoint support.

    Symmetric and bounded, unlike KL -- which matters here because one dataset
    routinely has empty bins another fills, and KL would simply return infinity
    for the most interesting features.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() <= 0 or q.sum() <= 0:
        return None
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], EPS))))
    return max(0.0, min(1.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def moments(block, n):
    """(mean, variance) in the transformed space from n, Σx, Σx²."""
    if not n:
        return None, None
    mean = block["sum_t"] / n
    return mean, max(0.0, block["sumsq_t"] / n - mean * mean)


def has_rank_space(view):
    """v1 profiles predate the joint histograms; the rank matrix is skipped, not faked."""
    return any(view.get("joint", {}).values())


def midrank_map(hist):
    """Bin -> its mid-quantile under this distribution: the binned equivalent of
    a rank. A bin holding the 10th to 30th percentile maps to 0.20, so a
    variable becomes approximately uniform on [0, 1] exactly as ranking makes
    it. Doing this at COMPARE time is what keeps the pipeline additive: the map
    is derived from the merged histogram of whichever pool is being described,
    while the stored joint counts stay pool-independent."""
    total = sum(hist) or 1
    out, cum = [], 0
    for c in hist:
        out.append((cum + c / 2) / total)
        cum += c
    return out


def rank_corr_matrix(view, feats):
    """Spearman-style correlation over binned mid-ranks, from the joint counts.

    Equal to Spearman's rho up to the resolution of the bin grid: within a bin
    the ordering is unknown, so ties are broken as the bin's average rank --
    the same convention Spearman itself uses for genuine ties.
    """
    n = view["n"]
    if n < 2:
        return np.full((len(feats), len(feats)), np.nan)
    qmap = {f: midrank_map(view["numeric"][f]["hist"] or []) for f in feats}
    mean, var = {}, {}
    for f in feats:
        h = np.asarray(view["numeric"][f]["hist"] or [], float)
        q = np.asarray(qmap[f])
        if h.sum() <= 0:
            mean[f], var[f] = np.nan, np.nan
            continue
        p = h / h.sum()
        mean[f] = float(p @ q)
        var[f] = float(p @ (q * q) - mean[f] ** 2)

    out = np.full((len(feats), len(feats)), np.nan)
    for i, a in enumerate(feats):
        out[i, i] = 1.0 if var[a] and var[a] > 0 else np.nan
        for j, b in enumerate(feats[i + 1:], start=i + 1):
            cells = view["joint"].get((a, b)) or view["joint"].get((b, a))
            if not cells or not var[a] or not var[b]:
                continue
            flip = (a, b) not in view["joint"]
            tot = exy = 0
            qa, qb = qmap[a], qmap[b]
            for cell, cnt in cells.items():
                x, y = cell.split(",")
                x, y = int(x), int(y)
                if flip:
                    x, y = y, x
                if x < len(qa) and y < len(qb):
                    exy += cnt * qa[x] * qb[y]
                    tot += cnt
            if tot < 2:
                continue
            cov = exy / tot - mean[a] * mean[b]
            out[i, j] = out[j, i] = cov / np.sqrt(var[a] * var[b])
    return np.clip(out, -1, 1)


def corr_matrix(view, feats, space="t"):
    """Pearson correlation of the pool, exactly, from its summed moments."""
    n = view["n"]
    key_s, key_ss = f"sum_{space}", f"sumsq_{space}"
    pairs = view["pairs"]
    if n < 2:
        return np.full((len(feats), len(feats)), np.nan)
    s = np.array([view["numeric"][f][key_s] for f in feats])
    ss = np.array([view["numeric"][f][key_ss] for f in feats])
    sxy = np.zeros((len(feats), len(feats)))
    for i, a in enumerate(feats):
        sxy[i, i] = ss[i]
        for j, b in enumerate(feats[i + 1:], start=i + 1):
            v = pairs.get((a, b), pairs.get((b, a)))
            sxy[i, j] = sxy[j, i] = 0.0 if v is None else v
    cov = sxy - np.outer(s, s) / n
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = cov / np.outer(sd, sd)
    out[~np.isfinite(out)] = np.nan
    return np.clip(out, -1, 1)


def label_signal(benign_hist, attack_hist, balanced=False):
    """Normalised mutual information between a feature's bins and the label.

    Returns MI(bin; attack?) / H(label), i.e. the fraction of the label's
    uncertainty this one feature resolves. `balanced` recomputes it under a
    50/50 prior: 2019 is 98% attack and 2017 is 25%, so raw MI is dominated by
    how skewed the capture happens to be, and only the balanced figure answers
    "does this feature MEAN the same thing in both domains".
    """
    b, a = np.asarray(benign_hist, float), np.asarray(attack_hist, float)
    if b.sum() <= 0 or a.sum() <= 0:
        return None
    pb, pa = b / b.sum(), a / a.sum()
    prior_b, prior_a = (0.5, 0.5) if balanced else (
        b.sum() / (a.sum() + b.sum()), a.sum() / (a.sum() + b.sum()))
    joint = np.vstack([prior_b * pb, prior_a * pa])
    px = joint.sum(axis=0)
    py = np.array([prior_b, prior_a])
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log2(
        joint[mask] / np.maximum((np.outer(py, px))[mask], EPS))))
    h_y = float(-np.sum(py[py > 0] * np.log2(py[py > 0])))
    return max(0.0, min(1.0, mi / h_y)) if h_y > 0 else None


def cat_distribution(block):
    """Value -> count, with the truncated tail kept as an explicit category.

    Folding the tail into '(other)' rather than dropping it keeps the shares
    honest: a dataset with 900 rare services and one with 3 should not look
    identical just because the top 40 match.
    """
    if not block:
        return {}
    d = dict(block["counts"])
    if block.get("n_other"):
        d["(other)"] = block["n_other"]
    return d


def cat_js(self_block, rest_block):
    a, b = cat_distribution(self_block), cat_distribution(rest_block)
    keys = sorted(set(a) | set(b))
    if not keys:
        return None, [], []
    js = js_divergence([a.get(k, 0) for k in keys], [b.get(k, 0) for k in keys])
    only_self = sorted(k for k in a if k not in b and k != "(other)")
    only_rest = sorted(k for k in b if k not in a and k != "(other)")
    return js, only_self, only_rest


def r6(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 6)


# ------------------------------------------------------------- comparison

def compare_one(dataset, profiles, spec):
    """This dataset against the pooled rest of the directory."""
    me = profiles[dataset]
    others = [p for ds, p in profiles.items() if ds != dataset]
    th = spec.thresholds

    views_self = {w: class_view(me, w) for w in ("all", "benign", "attack")}
    views_rest = {w: pool([class_view(p, w) for p in others]) for w in ("all", "benign", "attack")}

    # Compare only what both sides actually measured, and say so when they differ.
    feats_self = set(me["numeric_spec"])
    feats_rest = set().union(*(set(p["numeric_spec"]) for p in others)) if others else set()
    feats = ([f for f in me["numeric_spec"] if f in feats_rest] if others
             else list(me["numeric_spec"]))

    # ---- class distribution ----------------------------------------------
    n_self = me["meta"]["rows"]
    n_rest = sum(p["meta"]["rows"] for p in others)
    classes = set(me["by_class"])
    for p in others:
        classes |= set(p["by_class"])
    classes = sorted(classes)
    class_rows = []
    for c in classes:
        a = me["by_class"].get(c, {}).get("n", 0)
        b = sum(p["by_class"].get(c, {}).get("n", 0) for p in others)
        class_rows.append({
            "class": c, "n_self": a, "n_rest": b,
            "share_self": r6(a / n_self if n_self else 0),
            "share_rest": r6(b / n_rest if n_rest else 0),
            "only_self": bool(a and not b), "only_rest": bool(b and not a),
        })
    class_js = js_divergence([r["n_self"] for r in class_rows],
                             [r["n_rest"] for r in class_rows]) if others else None

    # ---- per-feature drift -------------------------------------------------
    # LABEL_COL rides in `feats` so that the correlation matrices carry the
    # point-biserial row, but it is not a feature and must never appear in a
    # feature table or be counted as one that transfers.
    numeric_rows = []
    for f in [x for x in feats if x != LABEL_COL]:
        s_all, s_ben, s_atk = (views_self[w]["numeric"].get(f) for w in
                               ("all", "benign", "attack"))
        r_all, r_ben, r_atk = ((views_rest[w]["numeric"].get(f) if views_rest[w] else None)
                               for w in ("all", "benign", "attack"))
        m_s, v_s = moments(s_all, views_self["all"]["n"])
        m_r, v_r = (moments(r_all, views_rest["all"]["n"]) if r_all else (None, None))
        pooled_sd = (np.sqrt((v_s + v_r) / 2) if v_s is not None and v_r is not None else None)

        # Domain shift is measured benign-against-benign on purpose: any gap
        # there is the testbed talking, not the attacks, and a feature with a
        # large one is a feature a cross-domain model will overfit to.
        def hist(block):
            return block["hist"] if block and block["hist"] else None

        js_domain = (js_divergence(hist(s_ben), hist(r_ben))
                     if hist(s_ben) and hist(r_ben) else None)
        js_concept = (js_divergence(hist(s_atk), hist(r_atk))
                      if hist(s_atk) and hist(r_atk) else None)
        js_all = (js_divergence(hist(s_all), hist(r_all))
                  if hist(s_all) and hist(r_all) else None)

        u_self = (label_signal(hist(s_ben), hist(s_atk), balanced=True)
                  if hist(s_ben) and hist(s_atk) else None)
        u_rest = (label_signal(hist(r_ben), hist(r_atk), balanced=True)
                  if hist(r_ben) and hist(r_atk) else None)

        undef_share = s_all["n_undefined"] / max(views_self["all"]["n"], 1)
        zero_share = s_all["n_zero"] / max(views_self["all"]["n"], 1)

        # Order matters: every disqualifying condition is tested before any
        # recommendation, because the failure mode that actually bit was a
        # feature undefined for 96% of rows being recommended on the strength of
        # the other 4%.
        verdict, why = "ok", ""
        if undef_share >= th["undefined_high"]:
            verdict, why = ("mostly undefined",
                            f"undefined for {100 * undef_share:.1f}% of flows here — "
                            f"every statistic below is computed on the remainder")
        elif v_s is not None and v_s < th["near_constant_var"]:
            verdict, why = "dead", "near-constant in this dataset"
        elif zero_share >= th["zero_high"] and (u_self or 0) < th["utility_useful"]:
            verdict, why = ("dead", f"{100 * zero_share:.1f}% zeros and almost no "
                                    f"label signal — nothing to carry")
        elif js_domain is not None and js_domain >= th["js_domain_high"]:
            verdict, why = ("domain-biased",
                            "benign traffic alone separates this dataset from the pool")
        elif u_self is not None and u_self < th["utility_low"] and (
                u_rest is None or u_rest < th["utility_low"]):
            verdict, why = "uninformative", "carries almost no label signal anywhere"
        elif (js_domain is not None and js_domain <= th["js_domain_low"]
                and u_self is not None and u_self >= th["utility_useful"]):
            verdict, why = "transfers", "stable across domains and informative in both"

        numeric_rows.append({
            "feature": f,
            "js_all": r6(js_all), "js_domain": r6(js_domain), "js_concept": r6(js_concept),
            "mean_self": r6(m_s), "mean_rest": r6(m_r),
            "sd_self": r6(np.sqrt(v_s) if v_s is not None else None),
            "sd_rest": r6(np.sqrt(v_r) if v_r is not None else None),
            "cohens_d": r6((m_s - m_r) / pooled_sd
                           if pooled_sd not in (None, 0) and m_s is not None
                           and m_r is not None else None),
            "var_ratio": r6(v_s / v_r if v_r else None),
            "undefined_self": r6(s_all["n_undefined"] / max(views_self["all"]["n"], 1)),
            "undefined_rest": r6(r_all["n_undefined"] / max(views_rest["all"]["n"], 1)
                                 if r_all else None),
            "zero_self": r6(s_all["n_zero"] / max(views_self["all"]["n"], 1)),
            "zero_rest": r6(r_all["n_zero"] / max(views_rest["all"]["n"], 1)
                            if r_all else None),
            "label_signal_self": r6(u_self), "label_signal_rest": r6(u_rest),
            "label_signal_gap": r6(abs(u_self - u_rest)
                                   if u_self is not None and u_rest is not None else None),
            "verdict": verdict, "verdict_reason": why,
            # The benign-vs-benign histograms travel with the row so the page can
            # draw the shift it is reporting, without reopening the profiles.
            "bin_labels": me["numeric_spec"][f]["bin_labels"],
            "hist_self": hist(s_ben), "hist_rest": hist(r_ben),
        })

    # ---- categorical drift -------------------------------------------------
    cat_rows = []
    cats = [c for c in me["categorical_spec"]
            if not others or any(c in p["categorical_spec"] for p in others)]
    for c in cats:
        s_blk = views_self["all"]["categorical"].get(c)
        r_blk = views_rest["all"]["categorical"].get(c) if views_rest["all"] else None
        js_a, only_self, only_rest = cat_js(s_blk, r_blk)
        js_dom, _, _ = cat_js(views_self["benign"]["categorical"].get(c),
                              views_rest["benign"]["categorical"].get(c)
                              if views_rest["benign"] else None)
        shared = (set(cat_distribution(s_blk)) & set(cat_distribution(r_blk))
                  if r_blk else set())
        tot = sum(cat_distribution(s_blk).values()) or 1
        cat_rows.append({
            "feature": c, "js_all": r6(js_a), "js_domain": r6(js_dom),
            "n_distinct_self": s_blk["n_distinct"] if s_blk else 0,
            "n_distinct_rest": r_blk["n_distinct"] if r_blk else 0,
            # Share of this dataset's traffic that uses a value the pool has
            # never seen -- the honest version of "novel categories": a model
            # trained elsewhere has no embedding for any of it.
            "unseen_mass": r6(sum(cat_distribution(s_blk).get(v, 0)
                                  for v in only_self) / tot),
            "only_self": only_self[:25], "only_rest": only_rest[:25],
            "shared_values": len(shared),
            "top_self": dict(sorted(cat_distribution(s_blk).items(),
                                    key=lambda kv: -kv[1])[:10]) if s_blk else {},
            "top_rest": dict(sorted(cat_distribution(r_blk).items(),
                                    key=lambda kv: -kv[1])[:10]) if r_blk else {},
        })

    # ---- correlation structure --------------------------------------------
    c_self = corr_matrix(views_self["all"], feats)
    c_rest = (corr_matrix(views_rest["all"], feats)
              if views_rest["all"] and views_rest["all"]["n"] else
              np.full_like(c_self, np.nan))
    delta = c_self - c_rest
    iu = np.triu_indices(len(feats), k=1)
    finite = np.isfinite(delta[iu])
    drift = sorted(
        ({"a": feats[i], "b": feats[j], "rho_self": r6(c_self[i, j]),
          "rho_rest": r6(c_rest[i, j]), "delta": r6(delta[i, j])}
         for i, j in zip(*iu) if np.isfinite(delta[i, j])),
        key=lambda d: -abs(d["delta"]))

    # Correlation with the 0/1 label is the point-biserial label relation; a
    # sign flip between domains is the clearest possible statement that a
    # feature means something different here than it does in the pool.
    label_corr = []
    if LABEL_COL in feats:
        k = feats.index(LABEL_COL)
        for i, f in enumerate(feats):
            if f == LABEL_COL or not np.isfinite(c_self[i, k]):
                continue
            # A "sign flip" only means anything if there is a relation to flip:
            # +0.004 here and -0.001 there is noise, not a contradiction.
            flip = bool(np.isfinite(c_rest[i, k]) and c_self[i, k] * c_rest[i, k] < 0
                        and min(abs(c_self[i, k]),
                                abs(c_rest[i, k])) >= th["rho_meaningful"])
            label_corr.append({"feature": f, "rho_self": r6(c_self[i, k]),
                               "rho_rest": r6(c_rest[i, k]), "flips_sign": flip})
        label_corr.sort(key=lambda d: -abs(d["rho_self"] or 0))

    # ---- the rank-style (Spearman) matrices -------------------------------
    rank = None
    if has_rank_space(views_self["all"]) and (
            views_rest["all"] and has_rank_space(views_rest["all"])):
        g_self = rank_corr_matrix(views_self["all"], feats)
        g_rest = rank_corr_matrix(views_rest["all"], feats)
        g_delta = g_self - g_rest
        g_drift = sorted(
            ({"a": feats[i], "b": feats[j], "rho_self": r6(g_self[i, j]),
              "rho_rest": r6(g_rest[i, j]), "delta": r6(g_delta[i, j])}
             for i, j in zip(*iu) if np.isfinite(g_delta[i, j])),
            key=lambda d: -abs(d["delta"]))
        # Where Pearson and the rank matrix disagree inside one dataset, the
        # Pearson value is being driven by the tail rather than by the bulk of
        # the flows -- worth knowing before a linear model is fitted to it.
        gap = sorted(({"a": feats[i], "b": feats[j], "pearson": r6(c_self[i, j]),
                       "rank": r6(g_self[i, j]), "gap": r6(c_self[i, j] - g_self[i, j])}
                      for i, j in zip(*iu)
                      if np.isfinite(c_self[i, j]) and np.isfinite(g_self[i, j])),
                     key=lambda d: -abs(d["gap"]))
        rank = {"self": [[r6(v) for v in row] for row in g_self],
                "rest": [[r6(v) for v in row] for row in g_rest],
                "top_drift": g_drift[:25], "pearson_gap": gap[:15]}

    # ---- one-vs-EACH, not just one-vs-pool --------------------------------
    # The pool is flow-weighted, so a 2.1M-flow dataset is nearly invisible
    # beside a 72M-flow one and "vs rest" quietly becomes "vs the biggest peer".
    # The per-peer breakdown and the macro (equal-weight) average say what the
    # pooled number cannot.
    per_peer = []
    for p in others:
        pv = {w: class_view(p, w) for w in ("all", "benign", "attack")}
        row = {"dataset": p["meta"]["dataset"], "flows": p["meta"]["rows"],
               "share_of_rest": r6(p["meta"]["rows"] / n_rest if n_rest else None),
               "features": {}}
        dom, con = [], []
        for f in [x for x in feats if x != LABEL_COL]:
            jd = js_divergence(views_self["benign"]["numeric"][f]["hist"],
                               pv["benign"]["numeric"][f]["hist"]) \
                if pv["benign"]["numeric"].get(f, {}).get("hist") else None
            jc = js_divergence(views_self["attack"]["numeric"][f]["hist"],
                               pv["attack"]["numeric"][f]["hist"]) \
                if pv["attack"]["numeric"].get(f, {}).get("hist") else None
            row["features"][f] = {"js_domain": r6(jd), "js_concept": r6(jc)}
            if jd is not None:
                dom.append(jd)
            if jc is not None:
                con.append(jc)
        row["domain_fingerprint"] = r6(np.mean(dom) if dom else None)
        row["concept_shift"] = r6(np.mean(con) if con else None)
        per_peer.append(row)
    per_peer.sort(key=lambda r: -(r["domain_fingerprint"] or 0))

    js_domain_vals = [r["js_domain"] for r in numeric_rows if r["js_domain"] is not None]
    js_concept_vals = [r["js_concept"] for r in numeric_rows if r["js_concept"] is not None]
    macro_domain = [r["domain_fingerprint"] for r in per_peer
                    if r["domain_fingerprint"] is not None]
    macro_concept = [r["concept_shift"] for r in per_peer if r["concept_shift"] is not None]
    benign_self = me["by_class"].get(me["benign_class"], {}).get("n", 0)

    return {
        "compare_version": COMPARE_VERSION,
        "meta": {
            "dataset": dataset, "role": me["meta"]["role"],
            "compared_against": sorted(p["meta"]["dataset"] for p in others),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "spec_sha": me["meta"]["spec_sha"], "spec_version": me["meta"]["spec_version"],
            "profiles_used": {p["meta"]["dataset"]: p["meta"]["generated_utc"]
                              for p in [me] + others},
            # Pooling a full profile with a smoke-test one is legitimate -- every
            # comparison is on shares, not counts -- but the reader has to be
            # told, so it travels with the report rather than being inferred.
            "partial": {p["meta"]["dataset"]: p["meta"].get("partial")
                        for p in [me] + others if p["meta"].get("partial")},
            "features_compared": len(feats),
            "features_only_self": sorted(feats_self - feats_rest) if others else [],
            "features_only_rest": sorted(feats_rest - feats_self) if others else [],
        },
        "overview": {
            "flows_self": n_self, "flows_rest": n_rest,
            "attack_ratio_self": r6(1 - benign_self / n_self if n_self else None),
            "attack_ratio_rest": r6(1 - sum(p["by_class"].get(p["benign_class"], {}).get("n", 0)
                                            for p in others) / n_rest if n_rest else None),
            "class_js": r6(class_js),
            "classes_only_self": [r["class"] for r in class_rows if r["only_self"]],
            "classes_only_rest": [r["class"] for r in class_rows if r["only_rest"]],
            # Headline numbers. domain_fingerprint is the mean benign-vs-benign
            # divergence: how loudly this capture announces itself before any
            # attack is involved. concept_shift is the same for attack traffic.
            "domain_fingerprint": r6(np.mean(js_domain_vals) if js_domain_vals else None),
            "concept_shift": r6(np.mean(js_concept_vals) if js_concept_vals else None),
            # ...and the same two numbers with every peer counted once, whatever
            # its flow count. Where these disagree, the pooled figure is really a
            # statement about the largest peer.
            "domain_fingerprint_macro": r6(np.mean(macro_domain) if macro_domain else None),
            "concept_shift_macro": r6(np.mean(macro_concept) if macro_concept else None),
            "rest_dominated_by": (per_peer and
                                  max(per_peer, key=lambda r: r["flows"])["dataset"] or None),
            "rest_dominance": r6(max((r["share_of_rest"] or 0) for r in per_peer)
                                 if per_peer else None),
            "corr_delta_mean_abs": r6(np.mean(np.abs(delta[iu][finite])) if finite.any() else None),
            "corr_delta_max_abs": r6(np.max(np.abs(delta[iu][finite])) if finite.any() else None),
            "n_features": len(numeric_rows),
            "n_domain_biased": sum(1 for r in numeric_rows if r["verdict"] == "domain-biased"),
            "n_transfers": sum(1 for r in numeric_rows if r["verdict"] == "transfers"),
            "n_unusable": sum(1 for r in numeric_rows
                              if r["verdict"] in ("dead", "mostly undefined")),
        },
        "class_distribution": class_rows,
        "numeric": sorted(numeric_rows, key=lambda r: -(r["js_domain"] or -1)),
        "categorical": sorted(cat_rows, key=lambda r: -(r["js_domain"] or -1)),
        "per_peer": per_peer,
        "rank_correlation": rank,
        "correlation": {
            "order": feats,
            "self": [[r6(v) for v in row] for row in c_self],
            "rest": [[r6(v) for v in row] for row in c_rest],
            "top_drift": drift[:25],
        },
        "label_correlation": label_corr,
    }


def regenerate(directory=EDA_DIR, only=None):
    """Rewrite every comparison in the pool. Returns the paths written.

    Every dataset's report is rewritten, not just the new one: "the rest" has a
    different meaning for each of them now, so leaving the others in place
    would leave stale reports that still look current.
    """
    directory = Path(directory)
    profiles = load_profiles(directory)
    if not profiles:
        return []
    spec = Spec()
    written = []
    for ds in sorted(profiles):
        if only and ds != only:
            continue
        out = directory / f"compare_{ds}.json"
        out.write_text(json.dumps(compare_one(ds, profiles, spec), indent=1) + "\n")
        written.append(out)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=str(EDA_DIR), help="directory holding profile_*.json")
    p.add_argument("--dataset", default=None,
                   help="regenerate only this dataset's comparison (default: all)")
    p.add_argument("--render", action="store_true", help="also rebuild the HTML")
    a = p.parse_args()

    profiles = load_profiles(a.dir)
    if not profiles:
        sys.exit(f"no profile_*.json in {a.dir} — run `make eda-all` first")
    print(f"pool: {', '.join(sorted(profiles))}")
    written = regenerate(a.dir, a.dataset)
    for w in written:
        doc = json.loads(w.read_text())
        o = doc["overview"]
        print(f"  {doc['meta']['dataset']:<16} vs {len(doc['meta']['compared_against'])} "
              f"other(s): domain fingerprint {o['domain_fingerprint']}, "
              f"{o['n_domain_biased']} domain-biased / {o['n_transfers']} transferable "
              f"features -> {w.name}")
    if a.render:
        from talos.data.profiling.eda import render
        print(f"rendered {len(render.render_all(a.dir))} page(s)")


if __name__ == "__main__":
    main()
