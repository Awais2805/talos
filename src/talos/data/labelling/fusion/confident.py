"""Traffic-adapted confident learning: weigh every pseudo-label, discard none.

Eslami & Hamouda §IV-E, Eq. 18-30. Pure numpy, taking probabilities as input,
so every formula is testable without a model. Two confident joints are
implemented and they are different quantities — see `ConfidentJoint`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Paper values, Table VII (self-generated / ISCX).
DEFAULT_Q = 0.70            # Eq. 20 quantile
DEFAULT_GAMMA = 4.0         # Eq. 24 slope
DEFAULT_W_MIN = 0.20        # Eq. 26 floor
DEFAULT_FOLDS = 5           # Eq. 18

#: Eq. 23 and Eq. 29 both guard a division; neither value is published.
EPSILON = 1e-6


class ConfidentLearningError(Exception):
    pass


def self_confidence(probabilities, labels):
    """Eq. 19: the probability the model gives to the label a row already has."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    return probabilities[np.arange(len(labels)), labels]


def class_thresholds(scores, labels, n_classes: int, q: float = DEFAULT_Q):
    """Eq. 20: the q-quantile of each class's own self-confidences.

    Replaces canonical confident learning's per-class MEAN threshold. A mean is
    dragged by the same over-confident errors the threshold is meant to exclude;
    a quantile is not.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    out = np.zeros(n_classes)
    for cls in range(n_classes):
        own = np.sort(scores[labels == cls])
        if not len(own):
            continue
        # k = ceil(q * n) on the sorted order statistics, as published.
        out[cls] = own[min(int(np.ceil(q * len(own))) - 1, len(own) - 1)]
    return out


def class_scales(scores, labels, n_classes: int, epsilon: float = EPSILON):
    """Eq. 21-23: median absolute deviation per class, floored.

    Raw MAD — the paper applies no 1.4826 consistency constant, so neither does
    this. It is a scale for the logistic, not an estimate of a normal sigma.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    out = np.full(n_classes, epsilon)
    for cls in range(n_classes):
        own = scores[labels == cls]
        if not len(own):
            continue
        out[cls] = max(float(np.median(np.abs(own - np.median(own)))), epsilon)
    return out


def logistic_weights(scores, labels, thresholds, scales,
                     gamma: float = DEFAULT_GAMMA, w_min: float = DEFAULT_W_MIN):
    """Eq. 24-26: a smooth weight in [w_min, 1], centred on the class threshold.

    A row exactly at its class's threshold gets the midpoint, not a coin flip —
    which is what a hard cut at the threshold would amount to.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    z = (scores - thresholds[labels]) / (gamma * scales[labels])
    return w_min + (1.0 - w_min) / (1.0 + np.exp(-z))


class ConfidentJoint:
    """Two estimators of the class-confusion matrix, kept apart deliberately.

    `soft` is the paper's Eq. 27 — a column-wise sum of predicted probability
    over rows carrying each pseudo-label. `northcutt` is the estimator from
    Confident Learning (Northcutt et al. 2021): a thresholded HARD count, which
    is what `cleanlab` computes. They are different quantities and the paper
    cites the second while defining the first, so both are here: the pipeline
    uses `soft`, and `northcutt` exists to be checked against cleanlab.
    """

    @staticmethod
    def soft(probabilities, labels, n_classes: int):
        """Eq. 27: Q[k, j] = sum of p_i[k] over rows pseudo-labelled j."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels)
        joint = np.zeros((n_classes, n_classes))
        for cls in range(n_classes):
            rows = probabilities[labels == cls]
            if len(rows):
                joint[:, cls] = rows.sum(axis=0)
        return joint

    @staticmethod
    def northcutt(probabilities, labels, n_classes: int):
        """The published confident joint: thresholded counts, argmax on ties.

        Threshold per class is the MEAN predicted probability for that class
        among rows already carrying it. A row counts towards `[k, j]` only when
        p[k] clears class k's threshold; where several clear it, the largest wins.
        """
        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels)
        thresholds = np.array([
            probabilities[labels == cls, cls].mean() if (labels == cls).any() else 1.0
            for cls in range(n_classes)])

        joint = np.zeros((n_classes, n_classes), dtype=np.int64)
        above = probabilities >= thresholds
        for row in range(len(labels)):
            candidates = np.flatnonzero(above[row])
            if not len(candidates):
                continue
            best = candidates[np.argmax(probabilities[row, candidates])]
            joint[best, labels[row]] += 1
        return joint


def balanced_retention(weights, labels, joint, n_classes: int,
                       w_min: float = DEFAULT_W_MIN, epsilon: float = EPSILON):
    """Eq. 28-30: rescale each class's weight mass to its estimated clean count.

    WATCH OUT. This lifts any class whose retained mass falls short of its
    estimated clean count. In the paper's task the minority classes are coherent
    applications; here the minority class is attack traffic, which is exactly
    where pseudo-labels are least reliable — and `rho` is estimated from the same
    model whose errors concentrate there. The per-class factors are returned so a
    lift is visible rather than silent.
    """
    weights, labels = np.asarray(weights, dtype=np.float64), np.asarray(labels)
    rho = np.zeros(n_classes)
    factors = np.ones(n_classes)
    out = weights.copy()

    for cls in range(n_classes):
        column = joint[:, cls].sum()
        rho[cls] = joint[cls, cls] / column if column else 0.0
        rows = labels == cls
        if not rows.any():
            continue
        target = rho[cls] * rows.sum()
        mass = weights[rows].sum()
        factors[cls] = target / max(epsilon, mass)
        out[rows] = np.clip(weights[rows] * factors[cls], w_min, 1.0)
    return out, rho, factors


def symmetric_cross_entropy(probabilities, labels, weights, n_classes: int,
                            alpha: float = 0.3, beta: float = 2.0,
                            epsilon: float = 0.01, xp=np):
    """Eq. 31-33: per-sample weighted SCE, `w'_i [alpha*CE + beta*RCE]`.

    Supplied here because it is paper mathematics, not a design choice, and
    whichever layer becomes `C_final` needs it. Nothing in the labelling stage
    calls it: Eq. 31 trains the FINAL classifier, which Talos places in the ML
    pipeline. Defaults are Table VII's alpha and beta.
    """
    probabilities = xp.asarray(probabilities, dtype=np.float64) if xp is np \
        else probabilities
    rows = len(labels)
    # Eq. 32: the label smoothed off a hard one-hot, so RCE's log is finite.
    target = xp.full((rows, n_classes), epsilon / (n_classes - 1))
    target[xp.arange(rows), labels] = 1.0 - epsilon

    safe = xp.clip(probabilities, 1e-12, 1.0)
    cross = -(target * xp.log(safe)).sum(axis=1)
    reverse = -(probabilities * xp.log(target)).sum(axis=1)
    return xp.asarray(weights) * (alpha * cross + beta * reverse)


@dataclass(frozen=True)
class NoiseReport:
    """What the noise estimate found, per class."""

    n_classes: int
    classes: tuple[str, ...]
    counts: tuple[int, ...]
    thresholds: tuple[float, ...]
    scales: tuple[float, ...]
    rho: tuple[float, ...]
    factors: tuple[float, ...]
    mean_weight: float
    min_weight: float
    balanced: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def lifted(self) -> tuple[str, ...]:
        """Classes balanced retention scaled UP — the ones to look at first."""
        return tuple(name for name, factor in zip(self.classes, self.factors)
                     if factor > 1.0)

    def to_dict(self) -> dict:
        return {
            "per_class": [
                {"class": c, "rows": n, "threshold": t, "mad": s, "clean_share": r,
                 "retention_factor": f}
                for c, n, t, s, r, f in zip(self.classes, self.counts,
                                            self.thresholds, self.scales,
                                            self.rho, self.factors)],
            "mean_weight": self.mean_weight, "min_weight": self.min_weight,
            "balanced_retention": self.balanced, "lifted_classes": list(self.lifted),
            **self.extra,
        }

    def describe(self) -> str:
        head = (f"{'class':<14}{'rows':>9}{'thresh':>9}{'mad':>8}"
                f"{'clean':>8}{'factor':>9}")
        lines = [head]
        for c, n, t, s, r, f in zip(self.classes, self.counts, self.thresholds,
                                    self.scales, self.rho, self.factors):
            flag = "  <-- lifted" if f > 1.0 else ""
            lines.append(f"{c:<14}{n:>9,}{t:>9.4f}{s:>8.4f}{r:>8.3f}{f:>9.3f}{flag}")
        lines.append(f"\nweights: mean {self.mean_weight:.4f}, "
                     f"min {self.min_weight:.4f}, nothing discarded")
        return "\n".join(lines)


class ConfidentLearning:
    """Probabilities and pseudo-labels in, per-sample weights out."""

    def __init__(self, space, q: float = DEFAULT_Q, gamma: float = DEFAULT_GAMMA,
                 w_min: float = DEFAULT_W_MIN, balanced: bool = True,
                 epsilon: float = EPSILON):
        self.space = space
        self.q, self.gamma, self.w_min = float(q), float(gamma), float(w_min)
        self.balanced = bool(balanced)
        self.epsilon = float(epsilon)

    def search(self, probabilities, labels, score, grid=None) -> tuple:
        """Alg. 4 lines 3-10: pick `(q, w_min, gamma)` by a supplied objective.

        The objective is INJECTED because the paper's is `C_final`'s macro-F1
        (line 11 sits inside this loop), and `C_final` lives in the ML pipeline.
        Supplying a scorer here rather than choosing one keeps the search
        faithful: whoever owns the final classifier owns the criterion.

        `score(weights) -> float`, higher is better. Returns the best triple and
        the full table, so a selection is always reported with what it beat.
        """
        grid = grid or {"q": (0.6, 0.7, 0.8),
                        "w_min": (0.1, 0.2, 0.3),
                        "gamma": (4.0, 6.0, 8.0)}
        table = []
        for q in grid["q"]:
            for w_min in grid["w_min"]:
                for gamma in grid["gamma"]:
                    trial = ConfidentLearning(
                        self.space, q=q, gamma=gamma, w_min=w_min,
                        balanced=self.balanced, epsilon=self.epsilon)
                    values, _noise = trial.weights(probabilities, labels)
                    table.append({"q": q, "w_min": w_min, "gamma": gamma,
                                  "score": float(score(values))})
        best = max(table, key=lambda row: row["score"])
        return (best["q"], best["w_min"], best["gamma"]), tuple(table)

    def weights(self, probabilities, labels) -> tuple:
        """`(weights, NoiseReport)`. Nothing is removed — only attenuated."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        k = self.space.n_classes
        if probabilities.ndim != 2 or probabilities.shape[1] != k:
            raise ConfidentLearningError(
                f"probabilities are {probabilities.shape}, but label space "
                f"{self.space.name!r} has {k} classes. A column order mismatch "
                f"here would mislabel every row and look like poor accuracy.")

        scores = self_confidence(probabilities, labels)
        thresholds = class_thresholds(scores, labels, k, self.q)
        scales = class_scales(scores, labels, k, self.epsilon)
        weights = logistic_weights(scores, labels, thresholds, scales,
                                   self.gamma, self.w_min)

        joint = ConfidentJoint.soft(probabilities, labels, k)
        if self.balanced:
            weights, rho, factors = balanced_retention(
                weights, labels, joint, k, self.w_min, self.epsilon)
        else:
            column = joint.sum(axis=0)
            rho = np.divide(np.diag(joint), column, out=np.zeros(k),
                            where=column != 0)
            factors = np.ones(k)

        report = NoiseReport(
            n_classes=k, classes=tuple(self.space.classes),
            counts=tuple(int((labels == c).sum()) for c in range(k)),
            thresholds=tuple(thresholds), scales=tuple(scales),
            rho=tuple(rho), factors=tuple(factors),
            mean_weight=float(weights.mean()), min_weight=float(weights.min()),
            balanced=self.balanced,
            extra={"q": self.q, "gamma": self.gamma, "w_min": self.w_min})
        return weights, report
