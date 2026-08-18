"""Algorithm 2 and Eq. 2-6 fidelity, on the parts that need no torch.

The augmentation, the reservoir and the constraint projection are pure numpy,
so the defects that mattered most are all reachable without the [torch] extra:
a one-hot block split across donors, a recency-biased marginal, and a
projection that rewrote rows nothing had touched.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from talos.common.plugins import ConfigPlugins
from talos.data.feature.featureset import FeatureError, FeatureSetLoader
from talos.data.feature.vectorise import build as build_vectoriser
from talos.data.labelling.behavioural.tabcl import (
    ClassMarginals, feature_units, project_constraints, view_indices,
)

SET = textwrap.dedent("""\
    name: probe
    numeric:
      a:
        transform: log1p
        indicator: true
      b:
        transform: log1p
    derived:
      ratio:
        expr: "a / nullif(b, 0)"
        transform: log1p
    categorical:
      proto:
        values: [tcp, udp, icmp]
    constraints:
      - name: ratio_definition
        form: ratio
        target: ratio
        operands: [a, b]
        weight: 0.25
""")


@pytest.fixture
def spec(tmp_path):
    (tmp_path / "probe.yaml").write_text(SET)
    features = FeatureSetLoader(ConfigPlugins(
        "feature set", built_in=tmp_path, error=FeatureError,
        addressable=False)).load("probe")
    return build_vectoriser("passthrough", features).columns()


# ------------------------------------------------------------- the units

def test_a_one_hot_block_is_one_unit(spec):
    """Eq. 4's `d` counts FEATURES; a block replaced column-wise stops being one."""
    units = feature_units(spec)
    block = next(u for u in units if len(u) == spec.categorical[0].width)
    assert tuple(block) == tuple(spec.categorical[0].indices)


def test_a_numeric_carries_its_indicator(spec):
    """Replacing a value but not its "was absent" flag states two different things."""
    units = feature_units(spec)
    a, flag = spec.index("a"), spec.indicators[spec.index("a")]
    assert (a, flag) in units


def test_every_column_belongs_to_exactly_one_unit(spec):
    """A column in two units gets replaced twice; in none, never."""
    seen = [i for unit in feature_units(spec) for i in unit]
    assert sorted(seen) == list(range(spec.width))


# ------------------------------------------------------------ the views

def test_the_two_index_sets_are_disjoint_when_they_can_be():
    """Eq. 4 asks for minimal overlap; with 2k <= d that means none at all.

    Two views that replaced the same feature are more alike, which is exactly
    the similarity the contrastive loss is trying to earn rather than be given.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        first, second = view_indices(rng, n_units=44, k=7)
        assert len(first) == len(second) == 7
        assert not set(first) & set(second)


def test_each_index_set_has_no_repeats():
    """A repeated index would replace one feature twice and another never."""
    rng = np.random.default_rng(1)
    first, second = view_indices(rng, n_units=44, k=7)
    assert len(set(first)) == 7 and len(set(second)) == 7


def test_it_falls_back_when_no_disjoint_pair_exists():
    """At a high replacement rate 2k exceeds d and disjointness is impossible."""
    rng = np.random.default_rng(2)
    first, second = view_indices(rng, n_units=10, k=8)
    assert len(first) == len(second) == 8
    assert set(first) & set(second)          # forced by the pigeonhole


# --------------------------------------------------------- the reservoir

def test_the_reservoir_is_uniform_not_recency_biased():
    """Subsampling old-union-new each batch decays survival as (S/(S+B))^t.

    Batches arrive in table order, so that skews the marginal toward the tail
    of the table -- and the marginal is what every augmented view is drawn from.
    """
    marginals = ClassMarginals(1, 1, size=100, seed=0)
    for start in range(0, 10000, 100):
        rows = np.arange(start, start + 100, dtype=float).reshape(-1, 1)
        marginals.observe(rows, np.zeros(100, dtype=int))
    kept = marginals.pools[0][:, 0]
    assert len(kept) == 100
    # A uniform sample of 0..9999 has mean ~5000; a recency-biased one sits far
    # higher. The bound is loose enough not to flake and tight enough to fail
    # the old behaviour, which kept almost only the last batches.
    assert 3500 < kept.mean() < 6500
    assert kept.min() < 2000


def test_the_reservoir_fills_before_it_replaces():
    """Below the cap nothing should be evicted at all."""
    marginals = ClassMarginals(1, 1, size=500, seed=0)
    marginals.observe(np.arange(300, dtype=float).reshape(-1, 1),
                      np.zeros(300, dtype=int))
    assert len(marginals.pools[0]) == 300


# ------------------------------------------------------------- the draw

def test_a_drawn_one_hot_block_stays_a_distribution(spec):
    """Two 1.0s or an all-zero block is not a category, and Pi_G cannot repair it."""
    units = feature_units(spec)
    width = spec.width
    rng = np.random.default_rng(0)
    pool = np.zeros((50, width))
    block = tuple(spec.categorical[0].indices)
    for row in range(50):
        pool[row, block[rng.integers(0, len(block))]] = 1.0
    marginals = ClassMarginals(1, width, size=50, seed=1)
    marginals.pools[0] = pool

    drawn = marginals.draw(np.zeros(40, dtype=int), units, width)
    assert np.allclose(drawn[:, block].sum(axis=1), 1.0)


def test_a_draw_is_independent_per_unit(spec):
    """Eq. 4 draws per coordinate; one donor row per view would be a mixup."""
    units = feature_units(spec)
    width = spec.width
    pool = np.tile(np.arange(len(units), dtype=float), (200, 1))
    # Give each donor row a distinct signature per unit.
    full = np.zeros((200, width))
    for u, unit in enumerate(units):
        full[:, unit[0]] = np.arange(200) * 10 + u
    marginals = ClassMarginals(1, width, size=200, seed=2)
    marginals.pools[0] = full

    drawn = marginals.draw(np.zeros(300, dtype=int), units, width)
    firsts = np.array([drawn[:, unit[0]] // 10 for unit in units])
    # If every unit came from one donor, all rows of `firsts` would be equal.
    assert not np.all(firsts == firsts[0])


# -------------------------------------------------------- the projection

def test_projection_leaves_untouched_rows_alone(spec):
    """Rewriting a row nothing replaced is a second augmentation, not Pi_G."""
    X = np.zeros((3, spec.width))
    a, b, ratio = spec.index("a"), spec.index("b"), spec.index("ratio")
    X[:, a], X[:, b], X[:, ratio] = np.log1p(10.0), np.log1p(2.0), 999.0
    replaced = np.zeros_like(X, dtype=bool)
    replaced[1, a] = True                       # only row 1 moved

    out = project_constraints(X, spec, replaced=replaced)
    assert out[0, ratio] == 999.0 and out[2, ratio] == 999.0
    assert out[1, ratio] == pytest.approx(np.log1p(5.0))


def test_projection_repairs_a_replaced_target(spec):
    """Replacing the target alone is the inconsistency Pi_G exists to repair.

    Keying applicability on the operands only would skip exactly this row: the
    target now disagrees with operands that never moved.
    """
    X = np.zeros((1, spec.width))
    a, b, ratio = spec.index("a"), spec.index("b"), spec.index("ratio")
    X[0, a], X[0, b], X[0, ratio] = np.log1p(10.0), np.log1p(2.0), 999.0
    replaced = np.zeros_like(X, dtype=bool)
    replaced[0, ratio] = True                   # only the target moved

    out = project_constraints(X, spec, replaced=replaced)
    assert out[0, ratio] == pytest.approx(np.log1p(5.0))


def test_projection_skips_a_zero_denominator(spec):
    """A ratio over zero is undefined, not zero -- the rule `residuals` uses."""
    X = np.zeros((1, spec.width))
    a, b, ratio = spec.index("a"), spec.index("b"), spec.index("ratio")
    X[0, a], X[0, b], X[0, ratio] = np.log1p(10.0), 0.0, 7.0
    replaced = np.ones_like(X, dtype=bool)
    assert project_constraints(X, spec, replaced=replaced)[0, ratio] == 7.0


def test_projection_skips_a_row_whose_operand_was_absent(spec):
    """An imputed sentinel is not a measurement to recompute a relation from."""
    X = np.zeros((1, spec.width))
    a, b, ratio = spec.index("a"), spec.index("b"), spec.index("ratio")
    X[0, a], X[0, b], X[0, ratio] = np.log1p(10.0), np.log1p(2.0), 7.0
    X[0, spec.indicators[a]] = 1.0              # `a` was absent
    replaced = np.ones_like(X, dtype=bool)
    assert project_constraints(X, spec, replaced=replaced)[0, ratio] == 7.0


def test_projection_without_a_mask_still_works(spec):
    """`replaced` is optional; row_extras and older callers pass nothing."""
    X = np.zeros((1, spec.width))
    a, b, ratio = spec.index("a"), spec.index("b"), spec.index("ratio")
    X[0, a], X[0, b], X[0, ratio] = np.log1p(10.0), np.log1p(2.0), 0.0
    assert project_constraints(X, spec)[0, ratio] == pytest.approx(np.log1p(5.0))


# --------------------------------------------------------------- Eq. 2

def test_a_per_constraint_phi_reaches_the_spec(spec):
    """Eq. 2 writes phi_m per constraint; one global scalar ties them together."""
    assert spec.constraint_weights == {"ratio_definition": 0.25}


def test_the_l1_reduction_is_the_mean_absolute_residual(spec):
    """Eq. 2 is an L1 norm; log-square is a different quantity and voids phi."""
    from talos.data.feature.vectorise import Residual

    residual = Residual(name="r", value=np.array([3.0, -5.0, 0.0]),
                        applicable=np.array([True, True, True]))
    assert residual.mean_abs() == pytest.approx(8.0 / 3.0)
    assert residual.mean_log_square() != residual.mean_abs()
