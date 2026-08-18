"""Gate 13 — confident learning, checked against an independent implementation.

Pure numpy taking probabilities as input, so every formula is exercised without
a model. `cleanlab` is the outside reference: rebuild rule 5, independence beats
coverage, turned on our own code.
"""

from __future__ import annotations

import numpy as np
import pytest

from talos.data.labelling.fusion.confident import (
    ConfidentJoint, ConfidentLearning, ConfidentLearningError, balanced_retention,
    class_scales, class_thresholds, logistic_weights, self_confidence,
)
from talos.data.labelling.space import LabelSpaceLoader


@pytest.fixture
def space():
    return LabelSpaceLoader().load("core-5")


def noisy(n=600, k=5, noise=0.2, seed=0):
    """Pseudo-labels with a known, class-correlated error rate."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, k, n)
    labels = truth.copy()
    flip = rng.random(n) < noise
    labels[flip] = (labels[flip] + 1) % k

    # Probabilities that mostly know the truth — what an OOS model would give.
    probabilities = rng.dirichlet(np.ones(k) * 0.5, n)
    probabilities[np.arange(n), truth] += 3.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, labels, truth


# ------------------------------------------------------------- the formulas

def test_self_confidence_reads_the_labels_own_column(space):
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert self_confidence(p, [0, 2]).tolist() == [0.7, 0.8]


def test_threshold_is_the_quantile_order_statistic_not_the_mean():
    """Eq. 20 replaces canonical CL's per-class MEAN.

    A mean is dragged by exactly the over-confident errors the threshold exists
    to exclude, so the two must not coincide on a skewed class.
    """
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.99])
    labels = np.zeros(5, dtype=int)
    t = class_thresholds(scores, labels, 1, q=0.70)[0]

    assert t == 0.4                       # ceil(0.7 * 5) = 4th order statistic
    assert t != scores.mean()


def test_mad_carries_no_consistency_constant():
    """The paper uses raw MAD; 1.4826 would scale every logistic differently."""
    scores = np.array([0.2, 0.4, 0.6, 0.8])
    labels = np.zeros(4, dtype=int)
    assert class_scales(scores, labels, 1)[0] == pytest.approx(0.2)


def test_a_row_at_its_threshold_gets_the_midpoint_weight():
    """Eq. 24-26 at z = 0. A hard cut here would be a coin flip on a tie."""
    w_min = 0.2
    w = logistic_weights(np.array([0.5]), np.array([0]), np.array([0.5]),
                         np.array([0.1]), gamma=4.0, w_min=w_min)
    assert w[0] == pytest.approx(w_min + 0.5 * (1 - w_min))


def test_weights_are_bounded_and_nothing_is_ever_discarded(space):
    probabilities, labels, _ = noisy()
    weights, report = ConfidentLearning(space).weights(probabilities, labels)

    assert len(weights) == len(labels)          # no row removed
    assert weights.min() >= 0.20 - 1e-12
    assert weights.max() <= 1.0 + 1e-12
    assert report.min_weight > 0                # attenuated, never zeroed


def test_a_wrong_label_is_weighed_below_a_right_one(space):
    """The whole point: the estimate has to separate clean from noisy."""
    probabilities, labels, truth = noisy(noise=0.3)
    weights, _ = ConfidentLearning(space).weights(probabilities, labels)

    clean, dirty = weights[labels == truth], weights[labels != truth]
    assert dirty.mean() < clean.mean()


def test_shape_mismatch_is_refused_by_name(space):
    with pytest.raises(ConfidentLearningError, match="column order mismatch"):
        ConfidentLearning(space).weights(np.ones((10, 3)) / 3, np.zeros(10, int))


# --------------------------------------------------------- balanced retention

def test_balanced_retention_lifts_a_starved_class_and_says_so(space):
    """Eq. 28-30, and the reason the factors are reported rather than applied silently."""
    probabilities, labels, _ = noisy()
    weights, report = ConfidentLearning(space, balanced=True).weights(
        probabilities, labels)
    plain, unbalanced = ConfidentLearning(space, balanced=False).weights(
        probabilities, labels)

    assert any(f != 1.0 for f in report.factors)
    assert all(f == 1.0 for f in unbalanced.factors)
    lifted = [c for c, f in zip(report.classes, report.factors) if f > 1.0]
    assert set(report.lifted) == set(lifted)
    assert not np.allclose(weights, plain)


def test_retention_mass_approaches_the_estimated_clean_count(space):
    """The constraint Eq. 30 is meant to enforce, within the clip."""
    probabilities, labels, _ = noisy()
    k = space.n_classes
    scores = self_confidence(probabilities, labels)
    weights = logistic_weights(scores, labels,
                               class_thresholds(scores, labels, k),
                               class_scales(scores, labels, k))
    joint = ConfidentJoint.soft(probabilities, labels, k)
    scaled, rho, _factors = balanced_retention(weights, labels, joint, k)

    for cls in range(k):
        rows = labels == cls
        if not rows.any():
            continue
        target = rho[cls] * rows.sum()
        assert scaled[rows].sum() == pytest.approx(target, rel=0.35)


# ------------------------------------------------------------ the two joints

def test_the_papers_joint_and_northcutts_are_different_quantities(space):
    """Eq. 27 is a soft column sum; the published confident joint is a hard count.

    The paper cites Northcutt while defining something else. Both are
    implemented so the pipeline can use the paper's and still be checked.
    """
    probabilities, labels, _ = noisy()
    k = space.n_classes
    soft = ConfidentJoint.soft(probabilities, labels, k)
    hard = ConfidentJoint.northcutt(probabilities, labels, k)

    assert soft.sum() == pytest.approx(len(labels))     # probabilities sum to 1
    assert hard.dtype == np.int64
    assert not np.allclose(soft, hard)


def test_our_northcutt_joint_matches_cleanlab(space):
    """The independent check. If these differ, one of us has the estimator wrong."""
    cleanlab = pytest.importorskip("cleanlab", reason="the [cl] extra is not installed")
    from cleanlab.count import compute_confident_joint

    probabilities, labels, _ = noisy(n=1500, seed=3)
    ours = ConfidentJoint.northcutt(probabilities, labels, space.n_classes)
    theirs = compute_confident_joint(labels=labels, pred_probs=probabilities,
                                     calibrate=False)

    # cleanlab indexes [given_label, true_label]; ours is [true, given].
    assert np.array_equal(ours, np.asarray(theirs).T)


def test_the_soft_joint_is_column_normalised_by_construction(space):
    """Each column is the total probability mass of rows carrying that label."""
    probabilities, labels, _ = noisy()
    joint = ConfidentJoint.soft(probabilities, labels, space.n_classes)
    for cls in range(space.n_classes):
        assert joint[:, cls].sum() == pytest.approx(int((labels == cls).sum()))


# ------------------------------------------------------------- the reporting

def test_report_names_every_class_and_publishes_no_single_score(space):
    probabilities, labels, _ = noisy()
    _weights, report = ConfidentLearning(space).weights(probabilities, labels)
    payload = report.to_dict()

    assert len(payload["per_class"]) == space.n_classes
    for key in ("accuracy", "f1", "score", "agreement"):
        assert key not in payload
    assert "lifted_classes" in payload
    assert "retention_factor" in payload["per_class"][0]
