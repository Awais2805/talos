"""Label-dependent selection: task relevance against the domain probe.

Mutual information per feature, the per-feature domain AUC, and the vault's
cross-reference — low domain AUC with high relevance keeps, high with low drops,
high with high is HELD for the post-training stage to settle. Runs after
labelling because MI needs a label; the label-free half is `screen`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from talos.data.preprocess.stats import quoted

#: Equal-frequency bins for a continuous feature. Quantiles, not widths: one
#: flood burst puts almost every row in bin 0 of an equal-width grid.
MI_BINS = 20

#: The vault's threshold. Above it a feature mostly encodes WHERE traffic came
#: from rather than what it did.
DOMAIN_AUC_HIGH = 0.75

KEEP, DROP, HOLD = "keep", "drop", "hold"

#: A verdict nothing can be said about, because one domain cannot be probed.
UNMEASURED = "unmeasured"


class RelevanceError(Exception):
    pass


def _mi_bits(table: np.ndarray) -> tuple[float, float]:
    """Mutual information and its share of the label's entropy.

    Raw MI is in bits and grows with the bin count, so it is not comparable
    across features of different cardinality; MI / H(Y) is, and is in [0, 1].
    """
    total = table.sum()
    if not total:
        return 0.0, 0.0
    joint = table / total
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = joint * np.log2(joint / (px * py))
    # MI is non-negative by definition; summing near-cancelling logs can land a
    # few ulps below zero, and a negative information score reads as a defect.
    mi = max(0.0, float(np.nansum(terms)))
    entropy = float(-np.nansum(py * np.log2(np.where(py > 0, py, 1.0))))
    return mi, (mi / entropy if entropy > 0 else 0.0)


@dataclass(frozen=True)
class FeatureRelevance:
    """One feature: how much it says about the label, and about the domain."""

    name: str
    mi: float
    mi_normalised: float
    domain_auc: float | None
    verdict: str
    reason: str
    per_domain: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"feature": self.name, "mi_bits": self.mi,
                "relevance": self.mi_normalised, "domain_auc": self.domain_auc,
                "verdict": self.verdict, "reason": self.reason,
                "domain_auc_per_domain": self.per_domain}


@dataclass
class RelevanceReport:
    """Schema v1's verdict, and the two measurements behind it."""

    dataset: str
    datasets: tuple[str, ...] = ()
    rows: int = 0
    features: tuple[FeatureRelevance, ...] = ()
    joint_domain_auc: float | None = None
    joint_note: str = ""
    relevance_cut: float = 0.0
    thresholds: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def of(self, verdict: str) -> tuple[str, ...]:
        return tuple(f.name for f in self.features if f.verdict == verdict)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset, "datasets": list(self.datasets),
            "rows": self.rows,
            "thresholds": {**self.thresholds, "relevance_cut": self.relevance_cut},
            "joint_domain_auc": self.joint_domain_auc,
            "joint_domain_auc_note": self.joint_note,
            "keep": list(self.of(KEEP)), "drop": list(self.of(DROP)),
            "hold": list(self.of(HOLD)), "unmeasured": list(self.of(UNMEASURED)),
            "features": [f.to_dict() for f in self.features],
            "policy": ("high domain AUC with high relevance is HELD, not "
                       "decided: only measured performance in the post-training "
                       "stage can settle whether it is signal or leak"),
            **self.extra,
        }

    def table(self) -> str:
        lines = [f"{'feature':<24}{'MI(bits)':>10}{'relevance':>11}"
                 f"{'domainAUC':>11}  verdict"]
        for f in sorted(self.features, key=lambda f: -f.mi_normalised):
            auc = "-" if f.domain_auc is None else f"{f.domain_auc:.4f}"
            lines.append(f"{f.name:<24}{f.mi:>10.4f}{f.mi_normalised:>11.4f}"
                         f"{auc:>11}  {f.verdict.upper()}")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [f"rows          {self.rows:,}",
                 f"domains       {', '.join(self.datasets) or '(one)'}",
                 f"relevance cut {self.relevance_cut:.4f}  (median of measured)",
                 f"keep {len(self.of(KEEP))} · drop {len(self.of(DROP))} · "
                 f"hold {len(self.of(HOLD))} · unmeasured {len(self.of(UNMEASURED))}"]
        if self.joint_domain_auc is not None:
            lines.append(f"joint probe   {self.joint_domain_auc:.4f} — how "
                         f"separable the domains are from the whole vector")
        elif self.joint_note:
            lines.append(f"joint probe   not run: {self.joint_note}")
        if self.of(HOLD):
            lines.append(f"HELD (decide post-training): {', '.join(self.of(HOLD))}")
        return "\n".join(lines)


class RelevanceAnalyst:
    """Task relevance and domain leakage, per feature."""

    def __init__(self, duck, bins: int = MI_BINS,
                 domain_high: float = DOMAIN_AUC_HIGH):
        self.duck = duck
        self.bins = int(bins)
        self.domain_high = float(domain_high)

    # ------------------------------------------------------------------ MI

    def mutual_information(self, source: str, feature: str, label: str,
                           numeric: bool) -> tuple[float, float]:
        """MI between one feature and the label, over equal-frequency bins."""
        col, y = quoted(feature), quoted(label)
        if numeric:
            # NULL is its own bin: absence is a fact about the flow, and
            # dropping those rows would score the feature on a different pool.
            binned = (f"SELECT ntile({self.bins}) OVER (ORDER BY {col}) AS b, "
                      f"{y} AS y FROM {source} WHERE {col} IS NOT NULL "
                      f"UNION ALL SELECT 0 AS b, {y} AS y FROM {source} "
                      f"WHERE {col} IS NULL")
        else:
            binned = (f"SELECT coalesce(CAST({col} AS VARCHAR), '__null') AS b, "
                      f"{y} AS y FROM {source}")
        rows = self.duck.sql(
            f"SELECT b, y, count(*) FROM ({binned}) GROUP BY 1, 2")
        return _mi_bits(self._contingency(rows))

    @staticmethod
    def _contingency(rows: Sequence) -> np.ndarray:
        bins = {b: i for i, b in enumerate(sorted({str(r[0]) for r in rows}))}
        labels = {y: i for i, y in enumerate(sorted({str(r[1]) for r in rows}))}
        table = np.zeros((len(bins), len(labels)))
        for b, y, n in rows:
            table[bins[str(b)], labels[str(y)]] = n
        return table

    # ---------------------------------------------------------- domain AUC

    def domain_auc(self, source: str, feature: str, domain: str,
                   positive: str, numeric: bool) -> float | None:
        """One-vs-rest AUC for one feature against one domain.

        A univariate logistic regression is monotone in its single input, so
        its AUC is exactly the AUC of the feature used directly as a ranker —
        the Mann-Whitney U statistic. Computed in SQL with midranks rather than
        by fitting a model: the same number, exactly, at any row count.

        A categorical has no order, so it is scored by the domain rate of its
        own value — target encoding, which is what a univariate model on its
        one-hot expansion would learn anyway.
        """
        col, dom = quoted(feature), quoted(domain)
        pos = f"({dom} = '{positive}')"
        score = col if numeric else (
            f"avg(CASE WHEN {pos} THEN 1.0 ELSE 0.0 END) OVER (PARTITION BY {col})")
        row = self.duck.one(f"""
            WITH scored AS (
                SELECT {score} AS s, {pos} AS pos FROM {source}
                WHERE {col} IS NOT NULL),
            ranked AS (
                SELECT pos, rank() OVER (ORDER BY s)
                     + (count(*) OVER (PARTITION BY s) - 1) / 2.0 AS r
                FROM scored)
            SELECT sum(CASE WHEN pos THEN r ELSE 0 END),
                   count(*) FILTER (WHERE pos),
                   count(*) FILTER (WHERE NOT pos)
            FROM ranked""")
        if row is None:
            return None
        rank_sum, n_pos, n_neg = row
        if not n_pos or not n_neg:
            return None
        auc = (float(rank_sum) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        # Direction carries no information about leakage: a feature that is
        # perfectly LOW in one domain identifies it exactly as well.
        return max(auc, 1.0 - auc)

    def joint_domain_auc(self, source: str, features: Sequence[str],
                         domain: str) -> tuple[float | None, str]:
        """How separable the domains are from the whole vector at once.

        The per-feature scores answer "which feature leaks"; this answers
        "is cross-domain training viable at all", which no per-feature number
        does. Needs a real multivariate fit, so it degrades loudly.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return None, ("scikit-learn is not installed; install the [model] "
                          "extra to measure joint domain separability")
        columns = ", ".join(
            f"coalesce(CAST({quoted(f)} AS DOUBLE), 0.0)" for f in features)
        rows = self.duck.sql(
            f"SELECT {columns}, {quoted(domain)} FROM {source} USING SAMPLE 200000")
        if len(rows) < 100:
            return None, f"only {len(rows)} rows sampled; too few to fit a probe"
        matrix = np.array([r[:-1] for r in rows], dtype=np.float64)
        labels = np.array([str(r[-1]) for r in rows])
        if len(set(labels)) < 2:
            return None, "one domain only; a probe needs at least two"
        model = LogisticRegression(max_iter=1000, random_state=0)
        scaled = StandardScaler().fit_transform(matrix)
        model.fit(scaled, labels)
        probabilities = model.predict_proba(scaled)
        score = (roc_auc_score(labels, probabilities[:, 1])
                 if len(set(labels)) == 2
                 else roc_auc_score(labels, probabilities, multi_class="ovr"))
        return float(score), "fitted on a 200k sample, in-sample"

    # ------------------------------------------------------------- verdict

    def decide(self, auc: float | None, relevance: float, cut: float) -> tuple:
        """The vault's cross-reference, and nothing more than it says."""
        if auc is None:
            return UNMEASURED, ("domain AUC could not be measured, so the "
                                "cross-reference cannot be applied")
        high_auc, high_rel = auc > self.domain_high, relevance >= cut
        if not high_auc and high_rel:
            return KEEP, (f"domain AUC {auc:.4f} is at or below "
                          f"{self.domain_high}, and relevance is above the cut")
        if high_auc and not high_rel:
            return DROP, (f"domain AUC {auc:.4f} exceeds {self.domain_high} "
                          f"while relevance is below the cut: it encodes where "
                          f"the traffic came from, not what it did")
        if high_auc and high_rel:
            return HOLD, (f"domain AUC {auc:.4f} and relevance are both high. "
                          f"Held, not decided: only measured performance under "
                          f"leave-one-domain-out can say which it is")
        return KEEP, (f"domain AUC {auc:.4f} is low; relevance is below the cut "
                      f"but a feature that does not identify its domain costs "
                      f"nothing to carry into post-training selection")

    # -------------------------------------------------------------- the run

    def analyse(self, dataset: str, source: str, numeric: Sequence[str],
                categorical: Sequence[str], label: str = "label_class",
                domain: str = "dataset") -> RelevanceReport:
        report = RelevanceReport(
            dataset=dataset,
            thresholds={"mi_bins": self.bins, "domain_auc_high": self.domain_high})
        report.rows = self.duck.one(f"SELECT count(*) FROM {source}")[0]
        domains = [str(r[0]) for r in self.duck.sql(
            f"SELECT DISTINCT {quoted(domain)} FROM {source} ORDER BY 1")]
        report.datasets = tuple(domains)

        measured: list[tuple[str, float, float, float | None, dict]] = []
        for name, is_numeric in ([(n, True) for n in numeric]
                                 + [(c, False) for c in categorical]):
            mi, relevance = self.mutual_information(source, name, label, is_numeric)
            per_domain: dict[str, float] = {}
            if len(domains) > 1:
                for one in domains:
                    auc = self.domain_auc(source, name, domain, one, is_numeric)
                    if auc is not None:
                        per_domain[one] = auc
            # The most identifiable domain is the leak that matters: a feature
            # that gives one domain away is unusable however well it hides the rest.
            worst = max(per_domain.values()) if per_domain else None
            measured.append((name, mi, relevance, worst, per_domain))

        scored = [m[2] for m in measured if m[3] is not None]
        report.relevance_cut = float(np.median(scored)) if scored else 0.0
        report.features = tuple(
            FeatureRelevance(name=name, mi=mi, mi_normalised=relevance,
                             domain_auc=auc, per_domain=per_domain,
                             **dict(zip(("verdict", "reason"),
                                        self.decide(auc, relevance,
                                                    report.relevance_cut))))
            for name, mi, relevance, auc, per_domain in measured)

        if len(domains) > 1:
            report.joint_domain_auc, report.joint_note = self.joint_domain_auc(
                source, list(numeric), domain)
        else:
            report.joint_note = (f"one domain ({domains[0] if domains else 'none'}); "
                                 f"a probe needs at least two")
        return report
