"""Gate 11 — the feature declaration and the matrix it produces.

Real parquet, real DuckDB, generated SQL run for effect. The thing that can be
wrong here is the SQL (D4.1), and a mocked relation would test the mock.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from talos.common.plugins import ConfigPlugins
from talos.data.feature.featureset import (
    FeatureError, FeatureSetLoader, LOG1P, MISSING, OTHER,
)
from talos.data.feature.vectorise import (
    PassthroughVectoriser, VectoriserError, build, residuals, untransform,
)

ROWS = [
    # uid   dur   ob      rb     op   rp   oib     rib    mb    proto  state  lo     lr     history
    ("u1", 10.0, 1000.0, 2000.0, 5.0, 6.0, 1200.0, 2300.0, 0.0, "tcp", "SF", True, False, "ShADadFf"),
    ("u2", None, None, 0.0, 1.0, 0.0, 60.0, 0.0, 0.0, "tcp", "S0", True, False, "S"),
    ("u3", 0.5, 40.0, 0.0, 2.0, 0.0, 100.0, 0.0, 12.0, "wireguard", "ZZZ", None, True, "Sr"),
]
COLUMNS = ("uid, duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts, "
           "orig_ip_bytes, resp_ip_bytes, missed_bytes, proto, conn_state, "
           "local_orig, local_resp, history")


@pytest.fixture
def conn(tmp_path):
    """A real parquet file, read back through a real DuckDB relation."""
    duck = duckdb.connect()
    values = ", ".join(
        "(" + ", ".join("NULL" if v is None else repr(v) for v in row) + ")"
        for row in ROWS)
    duck.execute(f"CREATE TABLE conn AS SELECT * FROM (VALUES {values}) t({COLUMNS})")
    path = tmp_path / "conn.parquet"
    duck.execute(f"COPY conn TO '{path}' (FORMAT PARQUET)")
    return duck.sql(f"SELECT * FROM read_parquet('{path}')")


@pytest.fixture
def features():
    return FeatureSetLoader().load("conn")


def declaration(tmp_path, name: str, doc: str):
    """A feature set on disk, resolved through a throwaway plug-in point."""
    (tmp_path / f"{name}.yaml").write_text(doc)
    point = ConfigPlugins("feature set", built_in=tmp_path, error=FeatureError,
                          addressable=False)
    return FeatureSetLoader(point).load(name)


# ------------------------------------------------------------ the declaration

def test_shipped_feature_set_loads_and_hashes(features):
    assert features.name == "conn"
    assert len(features.sha) == 12
    assert features.numeric and features.categorical and features.constraints


def test_feature_names_exist_on_the_conn_backbone(conn, features):
    """The declaration and the extractor must not drift into naming different columns.

    A typo'd column name would otherwise surface as a DuckDB binder error deep
    inside a training run, hours after the point where it could be read.
    """
    available = set(conn.columns)
    missing = [c for c in features.source_columns if c not in available]
    assert not missing, f"conn reads columns conn does not have: {missing}"


def test_categorical_without_a_declared_vocabulary_is_refused(tmp_path):
    """The cross-domain refusal: a fitted vocabulary means two different rulers."""
    with pytest.raises(FeatureError, match="vocabulary must be written down"):
        declaration(tmp_path, "bad", "name: bad\n"
                                     "numeric: {duration: {}}\n"
                                     "categorical:\n  proto: {}\n")


def test_duplicate_feature_name_is_refused(tmp_path):
    with pytest.raises(FeatureError, match="declared more than once"):
        declaration(tmp_path, "dup",
                    "name: dup\nnumeric: {duration: {}}\n"
                    "categorical:\n  duration: {values: [a, b]}\n")


def test_constraint_over_an_undeclared_feature_is_refused(tmp_path):
    with pytest.raises(FeatureError, match="undeclared feature"):
        declaration(tmp_path, "c",
                    "name: c\nnumeric: {a: {}, b: {}}\n"
                    "constraints:\n  - {name: x, form: sum, target: a, "
                    "operands: [b, nonexistent]}\n")


def test_empty_declaration_is_refused(tmp_path):
    with pytest.raises(FeatureError, match="declares no features"):
        declaration(tmp_path, "empty", "name: empty\n")


def test_unknown_transform_is_refused(tmp_path):
    with pytest.raises(FeatureError, match="transform"):
        declaration(tmp_path, "t", "name: t\nnumeric: {a: {transform: sqrt}}\n")


# ---------------------------------------------------------------- the matrix

def test_column_layout_is_numeric_then_one_hot_blocks(features):
    spec = build("passthrough", features).columns()
    assert spec.width == len(spec.names) == len(set(spec.names))
    assert spec.numeric == tuple(range(len(spec.numeric)))
    for block in spec.categorical:
        assert spec.names[block.start:block.stop][-1].endswith(OTHER)
    # Blocks are contiguous and non-overlapping, in declaration order.
    edges = [(b.start, b.stop) for b in spec.categorical]
    assert edges == sorted(edges)
    for (_, stop), (start, _) in zip(edges, edges[1:]):
        assert stop == start


def test_matrix_shape_types_and_keys(conn, features):
    vectoriser = build("passthrough", features)
    keys, X = vectoriser.matrix(conn)
    assert list(keys) == ["u1", "u2", "u3"]
    assert X.shape == (3, vectoriser.columns().width)
    assert X.dtype == np.float64


def test_one_hot_blocks_have_exactly_one_hot_per_row(conn, features):
    """Including for an unknown value and for NULL, which is the point of `__other`."""
    vectoriser = build("passthrough", features)
    _, X = vectoriser.matrix(conn)
    spec = vectoriser.columns()
    for block in spec.categorical:
        assert np.allclose(X[:, block.start:block.stop].sum(axis=1), 1.0)

    proto = next(b for b in spec.categorical if b.name == "proto")
    # `wireguard` is not in the vocabulary, so it lands in the catch-all.
    assert X[2, proto.stop - 1] == 1.0
    # As does a NULL, rather than becoming an all-zero row.
    local_orig = next(b for b in spec.categorical if b.name == "local_orig")
    assert X[2, local_orig.stop - 1] == 1.0


def test_missing_numeric_is_imputed_and_flagged(conn, features):
    vectoriser = build("passthrough", features)
    _, X = vectoriser.matrix(conn)
    spec = vectoriser.columns()
    value = spec.index("duration")
    flag = spec.index("duration" + MISSING)
    assert X[1, value] == 0.0 and X[1, flag] == 1.0
    assert X[0, flag] == 0.0
    # "we could not tell" and "nothing was sent" must stay distinguishable.
    assert X[1, spec.index("resp_bytes" + MISSING)] == 0.0


def test_transform_is_applied_and_is_invertible(conn, features):
    vectoriser = build("passthrough", features)
    _, X = vectoriser.matrix(conn)
    spec = vectoriser.columns()
    column = spec.index("orig_bytes")
    assert spec.transforms[column] == LOG1P
    assert np.isclose(X[0, column], np.log1p(1000.0))
    assert np.isclose(untransform(X[:, column], LOG1P)[0], 1000.0)


def test_batches_agree_with_the_whole_relation(conn, features):
    """Streaming exists because D_l is tens of millions of rows on a small box."""
    vectoriser = build("passthrough", features)
    keys, X = vectoriser.matrix(conn)
    streamed = list(vectoriser.batches(conn, size=2))
    assert [len(k) for k, _ in streamed] == [2, 1]
    assert np.array_equal(np.vstack([x for _, x in streamed]), X)
    assert list(np.concatenate([k for k, _ in streamed])) == list(keys)


# ------------------------------------------------------------- the constraints

def test_declared_constraints_hold_on_real_rows(conn, features):
    """Every relation must be satisfied by data that has not been reconstructed.

    If a residual is non-zero on the input itself, the constraint is describing
    something the feature set does not actually compute, and penalising a decoder
    on it would train it away from the truth.
    """
    vectoriser = build("passthrough", features)
    _, X = vectoriser.matrix(conn)
    for name, residual in residuals(X, vectoriser.columns()).items():
        assert residual.mean_square() < 1e-12, f"{name} is violated by real input"


def test_a_constraint_does_not_apply_where_an_operand_was_imputed(conn, features):
    """Otherwise the penalty is on OUR missing-value policy, not on the record.

    Row u2 has a NULL `orig_bytes`. `orig_ip_bytes = orig_bytes + overhead` then
    fails by the whole of `orig_ip_bytes` (60) purely because both parts were
    imputed to zero.
    """
    vectoriser = build("passthrough", features)
    _, X = vectoriser.matrix(conn)
    residual = residuals(X, vectoriser.columns())["orig_ip_bytes_decomposition"]
    assert residual.applicable.tolist() == [True, False, True]
    assert residual.value[1] == 0.0
    assert residual.n_applicable == 2


def test_a_ratio_does_not_apply_where_the_denominator_is_zero(conn, features):
    vectoriser = build("passthrough", features)
    _, X = vectoriser.matrix(conn)
    residual = residuals(X, vectoriser.columns())["resp_payload_shape"]
    # u3 has resp_pkts = 0, so "bytes per packet" is undefined, not violated.
    assert residual.applicable[2] == np.False_


def test_residual_is_detected_when_a_reconstruction_breaks_a_relation(conn, features):
    """The penalty has to be able to fire, or it is decoration."""
    vectoriser = build("passthrough", features)
    _, X = vectoriser.matrix(conn)
    spec = vectoriser.columns()
    broken = X.copy()
    broken[0, spec.index("orig_rate")] += 5.0        # a rate that is not bytes/duration
    residual = residuals(broken, spec)["orig_rate_definition"]
    assert residual.mean_square() > 0
    assert residual.applicable[0] == np.True_


def test_mean_square_is_over_applicable_rows_only():
    """Two captures with the same behaviour and different quality must score alike."""
    from talos.data.feature.vectorise import Residual
    residual = Residual("x", np.array([2.0, 0.0, 0.0]),
                        np.array([True, False, False]))
    assert residual.mean_square() == 4.0             # not 4/3
    empty = Residual("y", np.zeros(3), np.zeros(3, dtype=bool))
    assert empty.mean_square() == 0.0


# -------------------------------------------------------------- the vectoriser

def test_passthrough_fits_nothing_from_the_data(conn, features):
    """The same row must vectorise identically whatever it is seen alongside.

    A fitted scale would make a flow's vector depend on the batch it arrived in,
    and a model's input depend on which domain happened to be loaded first.
    """
    vectoriser = build("passthrough", features)
    _, whole = vectoriser.matrix(conn)
    _, subset = vectoriser.matrix(conn.limit(1))
    assert np.allclose(whole[0], subset[0])


def test_vectoriser_is_a_registered_plug_in():
    from talos.data.feature.vectorise import VECTORISERS
    assert "passthrough" in VECTORISERS.names()
    assert VECTORISERS.cls_for("passthrough") is PassthroughVectoriser
    assert VECTORISERS.error is VectoriserError


def test_unknown_column_is_refused_by_name(features):
    with pytest.raises(VectoriserError, match="not a column of this matrix"):
        build("passthrough", features).columns().index("no_such_feature")
