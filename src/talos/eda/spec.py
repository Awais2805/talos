#!/usr/bin/env python3
"""The EDA contract: spec.yaml loaded, hashed, and turned into SQL fragments.

Nothing here reads data. It exists so that exactly one place decides what a
feature IS -- its expression, its transform, its bin edges -- and so that both
the profiler and the comparison stage can prove they are talking about the same
thing by comparing a hash instead of trusting a version number nobody bumped.

`resolve()` is the other half of the modularity story: it intersects the spec
with the columns a source actually has, so the same spec profiles the labelled
zone now and the canonical `mapped` zone later, recording what it could not
measure instead of crashing or silently omitting it.
"""

import hashlib
from pathlib import Path

import yaml

SPEC_PATH = Path(__file__).with_name("spec.yaml")


def _q(col):
    """Quote an identifier -- Zeek's 'id.resp_p' is not a struct reference."""
    return '"' + col.replace('"', '""') + '"'


class Feature:
    """One measured column: how to compute it, and on what ruler."""

    def __init__(self, name, cfg, kind):
        self.name = name
        self.kind = kind                        # numeric | categorical
        self.expr = cfg["expr"].strip()
        self.requires = list(cfg.get("requires", []))
        self.transform = cfg.get("transform", "none")
        self.edges = [float(e) for e in cfg.get("edges", [])]

    # ---- SQL --------------------------------------------------------------
    @property
    def raw(self):
        return f"({self.expr})"

    @property
    def trans(self):
        """Moment/correlation space.

        log1p on the counters because a Pearson r on raw byte counts is really
        a statement about the handful of largest flows; greatest(...,0) guards
        the ln domain. Undefined values (e.g. bytes-per-packet of a 0-packet
        flow) impute to 0 here -- the same value a model would receive after
        imputation -- while the histogram keeps them out and counts them as
        undefined, so nothing is quietly invented.
        """
        if self.transform == "log1p":
            return f"coalesce(ln(1 + greatest({self.raw}, 0)), 0)"
        if self.transform == "none":
            return f"coalesce({self.raw}, 0)"
        raise ValueError(f"{self.name}: unknown transform {self.transform!r}")

    @property
    def nbins(self):
        """n edges -> n+1 bins: one underflow, n-1 interior, one overflow."""
        return len(self.edges) + 1

    def bin_counts(self, col=None):
        """One COUNT expression per bin. Together they partition the defined rows.

        Written out as explicit CASEs rather than DuckDB's histogram(x, bins):
        the bin boundaries are then visible in the query plan, identical on
        every DuckDB version, and the underflow/overflow bins are explicit
        instead of implied -- which matters because a bin that silently drops
        rows would show up as a distribution shift that isn't there.
        """
        r, e = (col or self.raw), self.edges
        out = [f"count(*) FILTER ({r} < {e[0]})"]
        out += [f"count(*) FILTER ({r} >= {lo} AND {r} < {hi})"
                for lo, hi in zip(e, e[1:])]
        out.append(f"count(*) FILTER ({r} >= {e[-1]})")
        return out

    def bin_index(self, col=None):
        """The bin number, 0..nbins-1, as a single monotone expression.

        Two things ride on this. Pearson r over bin indices is a rank-style
        correlation -- Spearman's robustness to the heavy tails, without needing
        a global sort -- and because the grid is fixed by the spec rather than
        by the data, those co-moments stay additive, so the pooled rest keeps
        working. True ranks would not: they depend on the whole dataset, so no
        sum of them could be merged across datasets.

        It is also the key for the 2-D joint histograms: two bin indices pack
        into one integer, turning a density surface into one MAP aggregate.
        """
        r, e = (col or self.raw), self.edges
        whens = " ".join(f"WHEN {r} < {v} THEN {i}" for i, v in enumerate(e))
        return f"CASE {whens} ELSE {len(e)} END"

    def bin_labels(self):
        """Human-readable bin names, in the same order as bin_counts()."""
        def fmt(v):
            return f"{v:g}"
        e = self.edges
        return ([f"<{fmt(e[0])}"] +
                [f"{fmt(lo)}–{fmt(hi)}" for lo, hi in zip(e, e[1:])] +
                [f"≥{fmt(e[-1])}"])


class Spec:
    """spec.yaml, hashed, with the feature set resolved against a real source."""

    def __init__(self, path=SPEC_PATH):
        self.path = Path(path)
        text = self.path.read_bytes()
        self.sha = hashlib.sha256(text).hexdigest()[:12]
        doc = yaml.safe_load(text)
        self.version = doc["version"]
        self.group_by = doc["group_by"]
        self.benign_class = doc["benign_class"]
        self.numeric = [Feature(n, c, "numeric") for n, c in doc["numeric"].items()]
        self.categorical = [Feature(n, c, "categorical")
                            for n, c in doc["categorical"].items()]
        self.identity = list(doc.get("identity", []))
        self.null_checks = list(doc.get("null_checks", []))
        self.quantiles = list(doc["quantiles"])
        self.top_k = int(doc["top_k"])
        self.thresholds = dict(doc["thresholds"])
        self.joint_pairs = [tuple(p) for p in doc.get("joint_pairs", [])]

    def contract(self, numeric_spec=None):
        """The part of the spec two profiles must agree on to be comparable.

        Measured, not hashed from the file: a profile carries its own edges and
        transforms, so comparability is checked against what was actually used
        rather than against a hash that also moves when a comment or a new
        optional measurement is added. Adding statistics stays backward
        compatible; changing a bin edge does not, and says which feature.
        """
        if numeric_spec is not None:
            return {n: (c["transform"], tuple(c["edges"]), c["expr"])
                    for n, c in numeric_spec.items()}
        return {f.name: (f.transform, tuple(f.edges), f.expr) for f in self.numeric}

    def resolve(self, columns):
        """Drop what this source cannot supply; report it rather than hide it.

        Returns (numeric, categorical, identity, null_checks, skipped).
        """
        have = set(columns)
        skipped = {}

        def keep(feats):
            out = []
            for f in feats:
                missing = [c for c in f.requires if c not in have]
                if missing:
                    skipped[f.name] = missing
                else:
                    out.append(f)
            return out

        num, cat = keep(self.numeric), keep(self.categorical)
        ident = [c for c in self.identity if c in have]
        nulls = [c for c in self.null_checks if c in have]
        return num, cat, ident, nulls, skipped

    @staticmethod
    def pairs(numeric):
        """Strictly-upper-triangle index pairs, in a fixed documented order.

        The profile stores Σ(x·y) for these pairs; the diagonal is the moment
        table's sumsq. Correlation is then exactly reconstructable -- for this
        dataset, for any other, and for any POOL of others -- from sums alone,
        which is the whole reason the comparison stage never re-reads the lake.
        """
        n = len(numeric)
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
