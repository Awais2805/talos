"""EDA v3: `columns_in()`'s identifier extraction and `Spec._adopt_feature_set()`.

Both landed as real, wired code with zero dedicated tests before this file --
the whole point of `_adopt_feature_set` is that a feature added to `conn` is
profiled without anyone touching spec.yaml, so its extraction logic deserves
the same scrutiny a hand-written declaration would get.
"""

from __future__ import annotations

import textwrap

from talos.data.profiling.eda.spec import Spec, columns_in


# ------------------------------------------------------------- columns_in()

def test_a_bare_column_is_its_own_requirement():
    assert columns_in("orig_bytes") == ["orig_bytes"]


def test_a_string_literal_is_not_mistaken_for_a_column():
    """`replace(history, 'S', '')` reads one column, not three."""
    assert columns_in("length(history) - length(replace(history, 'S', ''))") == ["history"]


def test_a_quoted_identifier_is_read_whole():
    """Zeek's `id.resp_p` is not a struct reference; the dot must survive."""
    assert columns_in('coalesce("id.resp_p", -1)') == ["id.resp_p"]


def test_sql_callables_and_keywords_are_excluded():
    assert columns_in("CASE WHEN duration IS NULL THEN 0 ELSE duration END") == ["duration"]
    assert columns_in("coalesce(CAST(x AS DOUBLE), 0.0)") == ["x"]


def test_a_bare_numeral_is_not_a_column():
    assert columns_in("orig_bytes + 40") == ["orig_bytes"]


def test_multiple_distinct_columns_come_back_sorted_and_deduplicated():
    assert columns_in("orig_bytes / nullif(orig_bytes + resp_bytes, 0)") == \
        ["orig_bytes", "resp_bytes"]


# ------------------------------------------------- Spec._adopt_feature_set()

REQUIRED = textwrap.dedent("""\
    version: 99
    group_by: label_class
    benign_class: benign
    quantiles: [0.5]
    top_k: 10
    thresholds: {}
""")


def write_spec(tmp_path, body, name="probe.yaml"):
    path = tmp_path / name
    path.write_text(REQUIRED + body)
    return Spec(path)


def test_adopting_conn_covers_every_declared_feature(tmp_path):
    """Nothing in `conn` should be silently left unprofiled."""
    from talos.data.feature.featureset import FeatureSetLoader

    spec = write_spec(tmp_path, "numeric: {}\ncategorical: {}\nfeatures: conn\n")
    base = FeatureSetLoader().load("conn")
    adopted = {f.name for f in spec.numeric} | {f.name for f in spec.categorical}
    declared = {f.name for f in base.numeric} | {c.name for c in base.categorical}
    assert declared <= adopted


def test_a_local_declaration_wins_over_the_adopted_one(tmp_path):
    """EDA may want a different display policy than the model's bare column."""
    spec = write_spec(tmp_path, textwrap.dedent("""\
        numeric:
          orig_bytes:
            expr: "coalesce(orig_bytes, -1)"
            requires: [orig_bytes]
        categorical: {}
        features: conn
    """))
    orig_bytes = next(f for f in spec.numeric if f.name == "orig_bytes")
    assert orig_bytes.expr == "coalesce(orig_bytes, -1)"


def test_an_adopted_feature_s_requires_is_derived_not_empty(tmp_path):
    """A derived history count needs `columns_in`, or it silently profiles nothing."""
    spec = write_spec(tmp_path, "numeric: {}\ncategorical: {}\nfeatures: conn\n")
    rst = next(f for f in spec.numeric if f.name == "hist_orig_rst")
    assert rst.requires == ["history"]


def test_an_adopted_log1p_feature_carries_the_transform(tmp_path):
    spec = write_spec(tmp_path, "numeric: {}\ncategorical: {}\nfeatures: conn\n")
    orig_bytes = next(f for f in spec.numeric if f.name == "orig_bytes")
    assert orig_bytes.transform == "log1p"


def test_an_adopted_untransformed_feature_stays_none(tmp_path):
    spec = write_spec(tmp_path, "numeric: {}\ncategorical: {}\nfeatures: conn\n")
    history_len = next(f for f in spec.numeric if f.name == "history_len")
    assert history_len.transform == "none"


def test_adopted_edges_fall_back_to_the_default_grid(tmp_path):
    spec = write_spec(tmp_path, textwrap.dedent("""\
        numeric: {}
        categorical: {}
        features: conn
        edges:
          __default__: [0, 1, 2]
    """))
    rst = next(f for f in spec.numeric if f.name == "hist_orig_rst")
    assert rst.edges == [0.0, 1.0, 2.0]


def test_adopted_edges_prefer_a_named_override_over_the_default(tmp_path):
    spec = write_spec(tmp_path, textwrap.dedent("""\
        numeric: {}
        categorical: {}
        features: conn
        edges:
          __default__: [0, 1, 2]
          orig_rate: [0, 100, 200]
    """))
    orig_rate = next(f for f in spec.numeric if f.name == "orig_rate")
    assert orig_rate.edges == [0.0, 100.0, 200.0]


def test_an_adopted_feature_with_no_edges_declared_gets_none(tmp_path):
    """No `__default__` and no override means an empty grid, not a crash."""
    spec = write_spec(tmp_path, "numeric: {}\ncategorical: {}\nfeatures: conn\n")
    rst = next(f for f in spec.numeric if f.name == "hist_orig_rst")
    assert rst.edges == []


def test_adopting_records_the_feature_set_s_hash(tmp_path):
    """A profile without the feature set's sha could not prove which conn it read."""
    spec = write_spec(tmp_path, "numeric: {}\ncategorical: {}\nfeatures: conn\n")
    assert spec.features_sha
