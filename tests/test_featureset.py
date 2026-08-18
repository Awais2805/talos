"""Gates 1.2/1.3 — declared standardisation and the history decomposition.

Standardisation is applied after the transform and undone before it, so the
order is what these tests pin: getting it backwards makes every constraint
residual wrong without making any of them fail.
"""

from __future__ import annotations

import textwrap

import duckdb
import numpy as np
import pytest

from talos.common.plugins import ConfigPlugins
from talos.data.feature.featureset import (
    FeatureError, FeatureSetLoader, Numeric,
)
from talos.data.feature.vectorise import build as build_vectoriser

SET = textwrap.dedent("""\
    name: probe
    numeric:
      a:
        transform: log1p
        centre: 2.0
        scale: 4.0
      b:
        transform: log1p
    categorical:
      proto:
        values: [tcp, udp]
""")


def loader(tmp_path, text=SET, name="probe"):
    (tmp_path / f"{name}.yaml").write_text(text)
    return FeatureSetLoader(ConfigPlugins(
        "feature set", built_in=tmp_path, error=FeatureError, addressable=False))


# --------------------------------------------------------- standardisation

def test_standardisation_is_applied_after_the_transform(tmp_path):
    """Constants live in log space; applying them first would scale raw units."""
    features = loader(tmp_path).load("probe")
    a = next(f for f in features.numeric if f.name == "a")
    sql = a.sql()
    assert sql.startswith("((ln(1 + greatest(")
    assert sql.endswith("- 2.0) / 4.0")


def test_a_feature_declaring_neither_passes_through(tmp_path):
    """conn standardises byte counts and leaves 0/1 flags alone."""
    features = loader(tmp_path).load("probe")
    b = next(f for f in features.numeric if f.name == "b")
    assert not b.standardised
    assert b.sql() == "ln(1 + greatest(coalesce(CAST(b AS DOUBLE), 0.0), 0))"


def test_declaring_only_one_of_the_pair_is_refused(tmp_path):
    """Half a standardisation is not a transform anybody chose."""
    text = SET.replace("    scale: 4.0\n", "")
    assert "centre: 2.0" in text and "scale:" not in text
    with pytest.raises(FeatureError, match="declares only centre"):
        loader(tmp_path, text, "half").load("half")


def test_a_zero_scale_is_refused(tmp_path):
    """It divides every row by zero; a constant column has no scale."""
    text = SET.replace("scale: 4.0", "scale: 0")
    with pytest.raises(FeatureError, match="scale is 0"):
        loader(tmp_path, text, "zero").load("zero")


def test_two_names_differing_only_by_case_are_refused(tmp_path):
    """DuckDB folds identifiers, so hist_S and hist_s would become one column."""
    text = textwrap.dedent("""\
        name: collide
        numeric:
          hist_s: {}
        derived:
          HIST_S: {}
        categorical:
          proto:
            values: [tcp]
    """)
    with pytest.raises(FeatureError, match="differ only by case|must match"):
        loader(tmp_path, text, "collide").load("collide")


# ------------------------------------------------------------- the reverse

def test_undo_reverses_standardise_then_transform(tmp_path):
    """A residual computed on the wrong scale is wrong, and never fails loudly."""
    features = loader(tmp_path).load("probe")
    spec = build_vectoriser("passthrough", features).columns()
    index = spec.index("a")
    original = np.array([0.0, 5.0, 100.0])
    encoded = (np.log1p(original) - 2.0) / 4.0
    assert np.allclose(spec.undo(index, encoded), original)


def test_undo_is_identity_for_an_unstandardised_feature(tmp_path):
    """Pass-through must stay exactly the old behaviour."""
    features = loader(tmp_path).load("probe")
    spec = build_vectoriser("passthrough", features).columns()
    index = spec.index("b")
    original = np.array([0.0, 5.0, 100.0])
    assert np.allclose(spec.undo(index, np.log1p(original)), original)


def test_the_scaling_map_only_holds_standardised_columns(tmp_path):
    """An entry for every column would make `undo` unpick a transform twice."""
    features = loader(tmp_path).load("probe")
    spec = build_vectoriser("passthrough", features).columns()
    assert set(spec.scaling) == {spec.index("a")}


def test_constraint_projection_round_trips_through_standardisation(tmp_path):
    """Eq. 6 recomputes a target from its operands, in ORIGINAL units.

    With standardisation in play that means undo scale-then-transform, compute,
    then reapply transform-then-scale. A projection that skipped the reapply
    would write a raw magnitude into a standardised column and the encoder
    would never see the scale it was trained on.
    """
    from talos.data.labelling.behavioural.tabcl import project_constraints

    text = textwrap.dedent("""\
        name: proj
        numeric:
          orig_bytes:
            transform: log1p
            centre: 1.0
            scale: 2.0
          duration:
            transform: log1p
        derived:
          orig_rate:
            expr: "orig_bytes / nullif(duration, 0)"
            transform: log1p
            centre: 0.5
            scale: 3.0
        categorical:
          proto:
            values: [tcp]
        constraints:
          - name: rate
            form: ratio
            target: orig_rate
            operands: [orig_bytes, duration]
    """)
    features = loader(tmp_path, text, "proj").load("proj")
    spec = build_vectoriser("passthrough", features).columns()

    bytes_i, dur_i = spec.index("orig_bytes"), spec.index("duration")
    rate_i = spec.index("orig_rate")

    X = np.zeros((1, spec.width))
    X[0, bytes_i] = (np.log1p(1000.0) - 1.0) / 2.0
    X[0, dur_i] = np.log1p(4.0)
    X[0, rate_i] = 0.0                       # deliberately wrong; Eq. 6 repairs it

    projected = project_constraints(X, spec)
    assert np.isclose(spec.undo(rate_i, projected[0, rate_i]), 250.0)


def test_a_valid_flow_has_zero_constraint_residual(tmp_path):
    """Eq. 2 must be evaluated in ORIGINAL units, not as the model sees them.

    In log space `a = b / c` becomes a difference, so `â − b̂/ĉ` scores a
    relation that does not hold and penalises correct reconstructions:
    1000 bytes / 10 s = 100 B/s reads as residual 0 undone and ~1.74 in log
    space. This test fails if the undo is ever removed again.
    """
    from talos.data.feature.vectorise import residuals

    text = textwrap.dedent("""\
        name: exact
        numeric:
          bytes:    {transform: log1p}
          duration: {transform: log1p}
        derived:
          rate:
            expr: "bytes / nullif(duration, 0)"
            transform: log1p
        categorical:
          proto:
            values: [tcp]
        constraints:
          - name: rate_definition
            form: ratio
            target: rate
            operands: [bytes, duration]
    """)
    features = loader(tmp_path, text, "exact").load("exact")
    spec = build_vectoriser("passthrough", features).columns()

    X = np.zeros((1, spec.width))
    X[0, spec.index("bytes")] = np.log1p(1000.0)
    X[0, spec.index("duration")] = np.log1p(10.0)
    X[0, spec.index("rate")] = np.log1p(100.0)      # exactly 1000 / 10

    found = residuals(X, spec)["rate_definition"]
    assert found.n_applicable == 1
    assert found.mean_abs() == pytest.approx(0.0, abs=1e-9)


# -------------------------------------------------------------- the guard

def test_the_log1p_guard_is_countable(tmp_path):
    """D2c: the clamp stays, but a clamp that fires is a screen defect, reported."""
    features = loader(tmp_path).load("probe")
    a = next(f for f in features.numeric if f.name == "a")
    con = duckdb.connect()
    fired = con.execute(
        f"SELECT sum({a.clamped_sql()}) FROM (VALUES (-1.0), (2.0), (3.0)) t(a)"
    ).fetchone()[0]
    assert fired == 1.0


def test_a_feature_with_no_log1p_can_never_clamp(tmp_path):
    """Only the log1p guard clamps, so nothing else may report one."""
    assert Numeric(name="x", expr="x").clamped_sql() == "0.0"


# ---------------------------------------------------------------- conn

def test_conn_loads_and_declares_the_whole_alphabet():
    """A letter with no column is a letter with nowhere to go in another domain."""
    features = FeatureSetLoader().load("conn")
    names = {f.name for f in features.numeric}
    for word in ("syn", "synack", "ack", "data", "fin", "rst", "checksum",
                 "gap", "retransmit", "zero_window", "inconsistent", "multiflag"):
        assert f"hist_orig_{word}" in names, word
        assert f"hist_resp_{word}" in names, word
    assert "hist_flip" in names and "history_len" in names


def test_conn_history_features_count_rather_than_flag():
    """`ShADadFfRR` really occurs in 2017: two resets are not one reset."""
    features = FeatureSetLoader().load("conn")
    rst = next(f for f in features.numeric if f.name == "hist_orig_rst")
    con = duckdb.connect()
    got = con.execute(
        f"SELECT {rst.sql()} FROM (VALUES ('ShADadFfRR'), ('ShADadFf')) t(history)"
    ).fetchall()
    assert [row[0] for row in got] == [2.0, 0.0]


def test_conn_separates_the_two_directions():
    """The case convention is what makes Sr a portscan and not a clean close."""
    features = FeatureSetLoader().load("conn")
    orig = next(f for f in features.numeric if f.name == "hist_orig_rst")
    resp = next(f for f in features.numeric if f.name == "hist_resp_rst")
    con = duckdb.connect()
    row = con.execute(
        f"SELECT {orig.sql()}, {resp.sql()} FROM (VALUES ('Sr')) t(history)"
    ).fetchone()
    assert row == (0.0, 1.0)


def test_conn_ships_no_invented_standardisation_constants():
    """D1 is unresolved; a plausible constant here is exactly the old failure."""
    features = FeatureSetLoader().load("conn")
    assert not [f for f in features.numeric if f.standardised]


def test_conn_is_the_only_declared_feature_set():
    """Consolidation's invariant: one `conn`, no vN siblings to drift apart."""
    assert FeatureSetLoader().load("conn").name == "conn"


# --------------------------------------------- marks: annotate, never remove

MARKED = textwrap.dedent("""\
    name: marked
    numeric:
      plain: {}
      dropped:
        screen: {verdict: drop, check: zero-variance, reason: constant}
      held:
        screen: {verdict: keep}
        relevance: {verdict: hold, reason: high domain AUC and high relevance}
      overruled:
        screen: {verdict: drop, reason: looked dead pre-label}
        relevance: {verdict: keep, reason: relevance says otherwise}
    categorical:
      proto:
        values: [tcp, udp]
        relevance: {verdict: drop, reason: domain leak}
    constraints:
      - name: bogus
        form: sum
        target: dropped
        operands: [plain, held]
""")


def test_an_unmarked_feature_defaults_to_keep(tmp_path):
    """`conn` itself carries no marks -- never screened means never excluded."""
    features = loader(tmp_path, MARKED, "marked").load("marked")
    plain = next(f for f in features.numeric if f.name == "plain")
    assert plain.verdict == "keep"


def test_a_stage_s_verdict_is_read_back(tmp_path):
    features = loader(tmp_path, MARKED, "marked").load("marked")
    dropped = next(f for f in features.numeric if f.name == "dropped")
    assert dropped.verdict == "drop"
    proto = next(c for c in features.categorical if c.name == "proto")
    assert proto.verdict == "drop"


def test_relevance_s_verdict_overrules_screen_s(tmp_path):
    """Relevance runs after screen and knows more; its mark is the one that counts."""
    features = loader(tmp_path, MARKED, "marked").load("marked")
    overruled = next(f for f in features.numeric if f.name == "overruled")
    assert overruled.verdict == "keep"


def test_nothing_is_ever_removed_from_the_loaded_declaration(tmp_path):
    """Marking is annotation: every declared feature loads, whatever its verdict."""
    features = loader(tmp_path, MARKED, "marked").load("marked")
    names = {f.name for f in features.numeric}
    assert names == {"plain", "dropped", "held", "overruled"}
    assert len(features.constraints) == 1


def test_active_excludes_only_the_drop_verdict(tmp_path):
    """`hold` and `keep` both stay in the working matrix; only `drop` leaves.

    `overruled` ends up `keep` (relevance overrules screen's drop), so it
    stays too -- only `dropped`, unmarked keep and unoverruled, is excluded.
    """
    features = loader(tmp_path, MARKED, "marked").load("marked").active()
    names = {f.name for f in features.numeric}
    assert names == {"plain", "held", "overruled"}


def test_active_prunes_a_constraint_orphaned_by_its_own_exclusion(tmp_path):
    """`bogus` targets `dropped`, which `.active()` excludes -- the constraint
    must go with it in this view, though it stays in the unfiltered declaration.
    """
    features = loader(tmp_path, MARKED, "marked").load("marked")
    assert len(features.constraints) == 1
    assert len(features.active().constraints) == 0


def test_an_unknown_verdict_is_refused(tmp_path):
    """A typo'd verdict must not silently mean 'never excluded'."""
    text = MARKED.replace("verdict: drop, check: zero-variance", "verdict: maybe")
    with pytest.raises(FeatureError, match="not one of"):
        loader(tmp_path, text, "bad").load("bad")
