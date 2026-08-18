"""Alg. 4 lines 3-10 and Eq. 31-33 — the two pieces C_final needs.

Neither is called by the labelling stage. Eq. 31 trains the FINAL classifier,
which Talos places in the ML pipeline, and the grid's objective is that
classifier's macro-F1 — so the search takes its scorer as an argument rather
than choosing one here.
"""

from __future__ import annotations

import numpy as np
import pytest

from talos.data.labelling.fusion.confident import (
    ConfidentLearning, symmetric_cross_entropy,
)


class Space:
    """The two attributes ConfidentLearning reads off a label space."""

    name = "toy"
    classes = ("benign", "attack")
    n_classes = 2

    def index(self, value):
        return self.classes.index(value)


@pytest.fixture
def pool():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=400)
    confident = rng.uniform(0.8, 1.0, size=400)
    probabilities = np.zeros((400, 2))
    probabilities[np.arange(400), labels] = confident
    probabilities[np.arange(400), 1 - labels] = 1.0 - confident
    return probabilities, labels


# ------------------------------------------------------------ the search

def test_the_search_returns_the_argmax_of_what_it_reports(pool):
    """The objective is injected: whoever owns C_final owns the criterion.

    Asserted against the table rather than against a guessed winner --
    balanced retention rescales each class's mass to its estimated clean count,
    so the mean weight barely moves with w_min and there is no monotone corner
    to predict. What must hold is that the returned triple IS the table's best.
    """
    probabilities, labels = pool
    best, table = ConfidentLearning(Space()).search(
        probabilities, labels, score=lambda w: float(np.mean(w)))

    top = max(table, key=lambda row: row["score"])
    assert best == (top["q"], top["w_min"], top["gamma"])
    assert len(table) == 27


def test_every_grid_point_is_reported_with_its_score(pool):
    """A selection has to be reportable alongside what it beat."""
    probabilities, labels = pool
    _best, table = ConfidentLearning(Space()).search(
        probabilities, labels, score=lambda w: float(np.mean(w)))
    for row in table:
        assert {"q", "w_min", "gamma", "score"} == set(row)
    assert len({(r["q"], r["w_min"], r["gamma"]) for r in table}) == len(table)


def test_a_custom_grid_is_honoured(pool):
    """Table VII's chosen values must be searchable as a one-point grid."""
    probabilities, labels = pool
    best, table = ConfidentLearning(Space()).search(
        probabilities, labels, score=lambda w: 1.0,
        grid={"q": (0.70,), "w_min": (0.20,), "gamma": (4.0,)})
    assert best == (0.70, 0.20, 4.0) and len(table) == 1


# --------------------------------------------------------------- Eq. 31-33

def test_sce_scales_with_the_per_sample_weight(pool):
    """Eq. 31 multiplies the whole bracket by w'_i -- that is CL's only effect."""
    probabilities, labels = pool
    full = symmetric_cross_entropy(probabilities, labels, np.ones(400), 2)
    half = symmetric_cross_entropy(probabilities, labels, np.full(400, 0.5), 2)
    assert np.allclose(half, full * 0.5)


def test_sce_is_finite_for_a_confident_wrong_prediction():
    """Eq. 32's smoothing exists so RCE's log is never log(0)."""
    probabilities = np.array([[1.0, 0.0]])
    labels = np.array([1])                      # confidently, entirely wrong
    loss = symmetric_cross_entropy(probabilities, labels, np.ones(1), 2)
    assert np.isfinite(loss).all()


def test_sce_punishes_a_wrong_prediction_more_than_a_right_one():
    """Otherwise it is not a loss."""
    right = symmetric_cross_entropy(
        np.array([[0.9, 0.1]]), np.array([0]), np.ones(1), 2)
    wrong = symmetric_cross_entropy(
        np.array([[0.9, 0.1]]), np.array([1]), np.ones(1), 2)
    assert wrong[0] > right[0]


def test_sce_defaults_are_table_vii():
    """alpha 0.3 and beta 2.0 are published; a silent change would be invisible."""
    import inspect

    signature = inspect.signature(symmetric_cross_entropy)
    assert signature.parameters["alpha"].default == 0.3
    assert signature.parameters["beta"].default == 2.0


def test_nothing_in_labelling_calls_the_final_loss():
    """Eq. 31 trains C_final, which is the ML pipeline's, not labelling's."""
    import subprocess
    import sys

    found = subprocess.run(
        [sys.executable, "-c",
         "import pathlib,re;"
         "root=pathlib.Path('src/talos/data/labelling');"
         "hits=[p for p in root.rglob('*.py')"
         " if 'symmetric_cross_entropy(' in p.read_text()"
         " and p.name != 'confident.py'];"
         "print(len(hits))"],
        capture_output=True, text=True)
    assert found.stdout.strip() == "0"
