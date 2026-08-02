#!/usr/bin/env python3
"""Tier 6 -- does a canonical class mean the same thing in every dataset?

Talos exists to test whether a detector trained on one testbed transfers to
another. That question is only meaningful if `dos` in 2017 and `dos` in 2018
name the same phenomenon. The taxonomy asserts they do. Nothing has ever checked
it.

The test used here is comparative rather than absolute, because there is no
threshold at which two distributions "are the same attack". A canonical class
should resemble ITSELF across datasets more than it resembles OTHER classes
within a dataset. When it does not, the label is a name shared by two different
things, and a transfer result about it measures the rename, not the detector.

Everything in this tier is computed from the EDA profile JSONs already on disk:
no lake access, no S3, instant. It reuses `js_divergence` from the comparison
kernel rather than reimplementing it, so a divergence quoted here and one quoted
in an EDA report are the same number computed the same way.
"""

import json
from itertools import combinations
from pathlib import Path

from src.eda.compare import js_divergence
from src.label_validation.core.finding import Finding, Severity
from src.label_validation.core import paths as _paths
from src.label_validation.core.registry import check

ROOT = _paths.ROOT
EDA_DIR = ROOT / "reports" / "eda"

# Classes that exist to describe absence of attack, or that name a scenario
# rather than a behaviour, are excluded from the coherence test: benign is not
# an attack, and comparing it across datasets is what xds.benign_baseline does.
NON_ATTACK = {"benign"}


# ------------------------------------------------------------------ loading

def _profiles(exclude_spec_mismatch=True):
    """Every EDA profile on disk, keyed by dataset.

    Profiles measured against different spec versions are not comparable -- the
    histogram bin edges are the ruler, and two rulers make a divergence
    meaningless -- so mismatches are dropped and reported rather than silently
    averaged in.
    """
    found, shas = {}, {}
    for p in sorted(EDA_DIR.glob("profile_*.json")):
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue
        ds = doc.get("meta", {}).get("dataset") or p.stem[len("profile_"):]
        found[ds] = doc
        shas[ds] = doc.get("meta", {}).get("spec_sha")
    if exclude_spec_mismatch and len(set(shas.values())) > 1:
        winner = max(set(shas.values()), key=list(shas.values()).count)
        found = {d: doc for d, doc in found.items() if shas[d] == winner}
    return found, shas


def _hist(profile, cls, feat):
    block = profile.get("by_class", {}).get(cls, {}).get("numeric", {}).get(feat)
    return block.get("hist") if block else None


def _features(profile):
    return sorted(profile.get("numeric_spec", {}))


def _mean_js(pa, ca, pb, cb, feats):
    """Mean Jensen-Shannon divergence between two (dataset, class) populations.

    Averaged over every numeric feature both profiles measured, so one feature
    with a quirk cannot carry the verdict.
    """
    vals = []
    for f in feats:
        ha, hb = _hist(pa, ca, f), _hist(pb, cb, f)
        if not ha or not hb or sum(ha) == 0 or sum(hb) == 0:
            continue
        # label_binary is a restatement of the class, not a description of the
        # traffic; including it would make every same-class pair look identical.
        if f == "label_binary":
            continue
        vals.append(js_divergence(ha, hb))
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def _no_profiles(check_id, ctx, shas):
    return Finding(
        check_id=check_id, dataset=ctx.dataset, severity=Severity.INFO,
        title="check did not run: no comparable EDA profiles on disk",
        detail="This tier is computed from the EDA profiles rather than the lake. "
               "Run `make eda-all` first. Recorded as a finding so it cannot be read "
               "as a clean result.",
        metrics={"profiles_found": len(shas), "spec_shas": sorted(set(filter(None, shas.values())))},
        repro="ls reports/eda/profile_*.json",
    )


# ------------------------------------------------------------------- checks

@check(id="xds.class_coheres", tier=6,
       title="a canonical class resembles itself across datasets more than it "
             "resembles other classes",
       max_severity=Severity.CRITICAL)
def class_coheres(ctx):
    """The load-bearing assumption of the whole project, stated as a test.

    For each canonical class present in two or more datasets, compare how far it
    sits from its own counterpart elsewhere against how far it sits from the
    nearest DIFFERENT class inside one dataset. If the same label spans a wider
    gap than two different labels do, then the taxonomy is grouping unlike things
    and a cross-domain claim about that class is measuring the name.

    Known instance this must reproduce: `dos` 2017 vs `dos` 2018 sits at mean JS
    0.331 while `dos` vs `ddos` inside 2018 sits at 0.138 -- the same label is
    2.4x further apart than two different labels are.
    """
    profiles, shas = _profiles()
    if len(profiles) < 2:
        return [_no_profiles("xds.class_coheres", ctx, shas)]

    feats = sorted(set.intersection(*(set(_features(p)) for p in profiles.values())))
    classes = {}
    for ds, prof in profiles.items():
        for cls in prof.get("by_class", {}):
            if cls not in NON_ATTACK:
                classes.setdefault(cls, []).append(ds)

    # Within each dataset, how far apart are its two most SIMILAR distinct
    # classes? That is the tightest "different things look different" baseline
    # the dataset itself provides.
    nearest_other = {}
    for ds, prof in profiles.items():
        present = [c for c in prof.get("by_class", {}) if c not in NON_ATTACK]
        best = None
        for a, b in combinations(sorted(present), 2):
            m, n = _mean_js(prof, a, prof, b, feats)
            if m is None:
                continue
            if best is None or m < best[0]:
                best = (m, a, b)
        if best:
            nearest_other[ds] = best

    findings, table = [], []
    for cls, datasets in sorted(classes.items()):
        if len(datasets) < 2:
            continue
        for da, db in combinations(sorted(datasets), 2):
            same, n = _mean_js(profiles[da], cls, profiles[db], cls, feats)
            if same is None:
                continue
            row = {"class": cls, "dataset_a": da, "dataset_b": db,
                   "same_class_js": round(same, 4), "features_compared": n}
            # Compare against whichever dataset provides the tighter baseline;
            # using the looser one would make it too easy to pass.
            baselines = [nearest_other[d] for d in (da, db) if d in nearest_other]
            if baselines:
                base = min(baselines, key=lambda b: b[0])
                row.update({"nearest_different_class_js": round(base[0], 4),
                            "nearest_pair": f"{base[1]} vs {base[2]}",
                            "ratio": round(same / base[0], 2) if base[0] > 0 else None})
            table.append(row)

            if baselines and same > base[0]:
                findings.append(Finding(
                    check_id="xds.class_coheres", dataset=ctx.dataset,
                    severity=Severity.CRITICAL if same > 2 * base[0] else Severity.MAJOR,
                    title=f"'{cls}' differs more between {da} and {db} (JS {same:.3f}) "
                          f"than '{base[1]}' and '{base[2]}' differ inside one dataset "
                          f"(JS {base[0]:.3f})",
                    detail=f"A canonical class should look more like itself in another "
                           f"testbed than like a different attack in the same testbed. "
                           f"'{cls}' does not: it spans {same / base[0]:.1f}x the distance "
                           f"that separates two genuinely different classes. Whatever a "
                           f"cross-domain result about '{cls}' measures, it is not one "
                           f"phenomenon.",
                    metrics=row, scope={"class": cls},
                    evidence=[{"comparison": f"{cls}: {da} vs {db}", "mean_js": round(same, 4)},
                              {"comparison": f"{base[1]} vs {base[2]} within a dataset",
                               "mean_js": round(base[0], 4)}],
                    repro="python -c \"from src.eda.compare import js_divergence\"  "
                          "# histograms from reports/eda/profile_<ds>.json by_class",
                ))
    if table and not findings:
        findings.append(Finding(
            check_id="xds.class_coheres", dataset=ctx.dataset, severity=Severity.INFO,
            title=f"every shared class is more self-similar across datasets than "
                  f"different classes are within one",
            detail="The taxonomy's grouping survives the comparison for every class "
                   "present in two or more datasets.",
            metrics={"comparisons": len(table)}, evidence=table,
        ))
    return findings


@check(id="xds.class_support_matrix", tier=6,
       title="every canonical class has enough evidence in enough datasets",
       max_severity=Severity.MAJOR)
def class_support_matrix(ctx):
    """A class carried by one dataset cannot demonstrate transfer.

    Counting, not modelling. The point is to state plainly which classes are
    able to support the project's claim before any model is trained on them --
    `heartbleed` is one flow in one dataset, `infiltration` is 25 and 8.
    """
    profiles, shas = _profiles()
    if not profiles:
        return [_no_profiles("xds.class_support_matrix", ctx, shas)]

    floor = int(ctx.threshold("class_support_min", 1000))
    need = int(ctx.threshold("class_support_datasets_min", 2))

    matrix = {}
    for ds, prof in profiles.items():
        for cls, block in prof.get("by_class", {}).items():
            matrix.setdefault(cls, {})[ds] = int(block.get("n", 0))

    rows, weak = [], []
    for cls in sorted(matrix):
        counts = matrix[cls]
        supported = sum(1 for n in counts.values() if n >= floor)
        row = {"class": cls, "total": sum(counts.values()),
               "datasets_with_support": supported,
               **{ds: counts.get(ds, 0) for ds in sorted(profiles)}}
        rows.append(row)
        if cls not in NON_ATTACK and supported < need:
            weak.append(row)
    if not weak:
        return [Finding(
            check_id="xds.class_support_matrix", dataset=ctx.dataset, severity=Severity.INFO,
            title="every attack class has support in enough datasets",
            metrics={"classes": len(rows), "support_floor": floor}, evidence=rows,
        )]
    return [Finding(
        check_id="xds.class_support_matrix", dataset=ctx.dataset, severity=Severity.MAJOR,
        title=f"{len(weak)} class(es) lack support in {need}+ datasets and cannot carry "
              f"a cross-domain claim",
        detail=f"A class needs at least {floor:,} flows in at least {need} datasets before "
               f"a transfer result about it means anything. These do not have it, so they "
               f"should be dropped, folded into a broader class, or reported as "
               f"single-domain observations -- not evaluated as transfer.",
        metrics={"classes_below_floor": len(weak), "support_floor": floor,
                 "datasets_required": need},
        evidence=rows,
    )]


@check(id="xds.taxonomy_audit", tier=6,
       title="the taxonomy maps every raw attack name and no canonical class is empty",
       max_severity=Severity.MAJOR)
def taxonomy_audit(ctx):
    """Does the mapping from raw attack names to canonical classes hold up?

    Three ways it can fail quietly: a raw name in a manifest with no mapping
    (labelling would have refused, so this is a guard against future edits), a
    canonical class declared but never produced by any rule, and a class whose
    every rule comes from a single dataset while being treated as shared.
    """
    tax = ctx.taxonomy or {}
    mapping = tax.get("raw_to_canonical", {})
    declared = set(tax.get("canonical_classes", {}))

    unmapped = sorted({r["name"] for r in ctx.rules if r["name"] not in mapping})
    produced = {r["canonical"] for r in ctx.rules}
    orphan_targets = sorted(set(mapping.values()) - declared)

    profiles, _ = _profiles()
    seen_anywhere = set()
    for prof in profiles.values():
        seen_anywhere |= set(prof.get("by_class", {}))
    never_produced = sorted(declared - seen_anywhere - {"benign"}) if profiles else []

    problems = []
    if unmapped:
        problems.append(("raw attack names with no canonical mapping", unmapped))
    if orphan_targets:
        problems.append(("canonical targets not declared in canonical_classes", orphan_targets))
    if never_produced:
        problems.append(("declared classes no dataset ever produced", never_produced))
    if not problems:
        return [Finding(
            check_id="xds.taxonomy_audit", dataset=ctx.dataset, severity=Severity.INFO,
            title="taxonomy is complete and every declared class is populated",
            metrics={"raw_names_mapped": len(mapping), "canonical_classes": len(declared),
                     "classes_produced_here": len(produced)},
        )]
    return [Finding(
        check_id="xds.taxonomy_audit", dataset=ctx.dataset, severity=Severity.MAJOR,
        title=f"{len(problems)} taxonomy inconsistency/ies",
        detail="Cross-domain evaluation is only meaningful when the classes mean the same "
               "thing everywhere, which starts with the mapping being complete and every "
               "declared class actually existing in the data.",
        metrics={"issue_count": len(problems)},
        evidence=[{"issue": name, "items": items} for name, items in problems],
        repro="src/preprocess/manifests/taxonomy.yaml",
    )]


@check(id="xds.benign_baseline", tier=6,
       title="benign traffic is more alike across datasets than benign is to attack",
       max_severity=Severity.CRITICAL)
def benign_baseline(ctx):
    """Benign is the reference against which domain shift is measured.

    If one dataset's benign class sits closer to its own attack class than to
    another dataset's benign, that benign class is contaminated -- it is not
    describing normal traffic at all, and using it as a false-positive baseline
    would measure something else entirely.

    Known instance this must reproduce: 2019's benign sits at JS 0.064 from 2019
    attack on conn_state and 0.743 from 2017 benign.
    """
    profiles, shas = _profiles()
    if len(profiles) < 2:
        return [_no_profiles("xds.benign_baseline", ctx, shas)]
    feats = sorted(set.intersection(*(set(_features(p)) for p in profiles.values())))

    findings, rows = [], []
    for ds, prof in profiles.items():
        by_class = prof.get("by_class", {})
        if "benign" not in by_class:
            continue
        attack = [c for c in by_class if c not in NON_ATTACK]
        # Distance from this dataset's benign to its own attack traffic.
        own = [(_mean_js(prof, "benign", prof, c, feats)[0], c) for c in attack]
        own = [(m, c) for m, c in own if m is not None]
        # ...against the distance to every other dataset's benign.
        peer = []
        for other, oprof in profiles.items():
            if other == ds or "benign" not in oprof.get("by_class", {}):
                continue
            m, _ = _mean_js(prof, "benign", oprof, "benign", feats)
            if m is not None:
                peer.append((m, other))
        if not own or not peer:
            continue
        nearest_attack = min(own)
        nearest_peer = min(peer)
        row = {"dataset": ds,
               "benign_to_own_attack_js": round(nearest_attack[0], 4),
               "nearest_own_attack_class": nearest_attack[1],
               "benign_to_peer_benign_js": round(nearest_peer[0], 4),
               "nearest_peer": nearest_peer[1],
               "contaminated": nearest_attack[0] < nearest_peer[0]}
        rows.append(row)
        if row["contaminated"]:
            findings.append(Finding(
                check_id="xds.benign_baseline", dataset=ctx.dataset,
                severity=Severity.CRITICAL,
                title=f"{ds}'s benign class is closer to its own '{nearest_attack[1]}' "
                      f"(JS {nearest_attack[0]:.3f}) than to {nearest_peer[1]}'s benign "
                      f"(JS {nearest_peer[0]:.3f})",
                detail=f"Benign traffic from two testbeds should resemble each other more "
                       f"than either resembles an attack. {ds}'s does not, which means its "
                       f"benign class is largely unlabelled attack traffic. It cannot serve "
                       f"as a false-positive baseline, and any accuracy computed against it "
                       f"is measuring contamination.",
                metrics=row, scope={"dataset": ds},
                evidence=[{"comparison": f"{ds} benign vs {ds} {c}", "mean_js": round(m, 4)}
                          for m, c in sorted(own)]
                         + [{"comparison": f"{ds} benign vs {o} benign", "mean_js": round(m, 4)}
                            for m, o in sorted(peer)],
            ))
    if rows and not findings:
        findings.append(Finding(
            check_id="xds.benign_baseline", dataset=ctx.dataset, severity=Severity.INFO,
            title="every benign class is nearer to its peers than to its own attacks",
            metrics={"datasets_compared": len(rows)}, evidence=rows,
        ))
    return findings
