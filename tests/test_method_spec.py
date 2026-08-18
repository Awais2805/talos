"""Method declarations and label spaces.

The distinction this file exists to pin: an IMPLEMENTATION is a class, a
DECLARATION is a `method.yaml`, and one implementation backs many declarations.
`ae` and any `--method-file` variant of it are the same code with different
settings and must be able to produce two labelled tables side by side without
either pretending to be the other -- so the DECLARATION's name scopes the
output path.

And the label space: which classes a study may predict, as a hashed declaration
rather than a constant, with every exclusion carrying its argument.
"""

import textwrap

import pytest

from talos.common.plugins import ConfigPlugins, SEARCHED
from talos.data.labelling.base import LABELLERS, LabellingMethod
from talos.data.labelling.method import METHODS, MethodError, MethodLoader
from talos.data.labelling.space import (
    OUT_OF_SPACE, SPACES, LabelSpaceError, LabelSpaceLoader,
)
from talos.data.labelling.taxonomy import TaxonomyMapper

SPACE = textwrap.dedent("""
    name: toy-space
    classes: [benign, ddos, dos]
    excluded:
      heartbleed: {flows: 1, domains: 1, reason: unestimable}
""")

METHOD = textwrap.dedent("""
    name: toy-v1
    implementation: schedule
    label_space: toy-space
    taxonomy: default
""")


@pytest.fixture
def spaces(tmp_path):
    directory = tmp_path / "spaces"
    directory.mkdir()
    (directory / "toy-space.yaml").write_text(SPACE)
    point = ConfigPlugins("label space", built_in=directory,
                          error=LabelSpaceError, addressable=False)
    return LabelSpaceLoader(point), directory


@pytest.fixture
def methods(tmp_path, spaces):
    loader, _ = spaces
    directory = tmp_path / "methods"
    (directory / "toy-v1").mkdir(parents=True)
    (directory / "toy-v1" / "method.yaml").write_text(METHOD)
    point = ConfigPlugins("method", built_in=directory, filename="method.yaml",
                          error=MethodError, addressable=False)
    return MethodLoader(point, spaces=loader), directory


# ----------------------------------------------------------------- the space

def test_a_space_declares_what_may_be_predicted(spaces):
    loader, _ = spaces
    space = loader.load("toy-space")
    assert space.classes == ("benign", "ddos", "dos")
    assert space.n_classes == 3
    assert space.attack_classes == ("ddos", "dos")
    assert "ddos" in space and "heartbleed" not in space


def test_the_declared_order_is_the_noise_matrix_index(spaces):
    """Confident learning indexes rows and columns by class; the order has to be
    the declaration's, not whatever a set iterated to."""
    loader, _ = spaces
    space = loader.load("toy-space")
    assert [space.index(c) for c in space.classes] == [0, 1, 2]
    with pytest.raises(LabelSpaceError, match="not in label space"):
        space.index("heartbleed")


def test_an_excluded_class_carries_its_reason(spaces):
    loader, _ = spaces
    space = loader.load("toy-space")
    assert space.reason_for("heartbleed") == "unestimable"
    assert space.reason_for("ddos") is None
    assert "1 flows" in space.excluded["heartbleed"].describe()


def test_a_space_without_benign_is_refused(spaces, tmp_path):
    """Closed-world labelling falls to benign, so a space without it would emit a
    class outside itself."""
    loader, directory = spaces
    (directory / "bad.yaml").write_text("name: bad\nclasses: [ddos, dos]\n")
    with pytest.raises(LabelSpaceError, match="no 'benign' class"):
        loader.load("bad")


def test_a_space_with_no_classes_is_refused(spaces, tmp_path):
    loader, directory = spaces
    (directory / "bad.yaml").write_text("name: bad\nclasses: []\n")
    with pytest.raises(LabelSpaceError, match="no classes"):
        loader.load("bad")


def test_a_class_that_is_both_included_and_excluded_is_refused(spaces):
    """One of the two is a leftover and nothing can tell which."""
    loader, directory = spaces
    (directory / "bad.yaml").write_text(
        "name: bad\nclasses: [benign, ddos]\nexcluded:\n  ddos: {reason: oops}\n")
    with pytest.raises(LabelSpaceError, match="classes AND excluded"):
        loader.load("bad")


def test_an_exclusion_without_a_reason_is_refused(spaces):
    """Holding a class out of a study is an argument and has to be on record."""
    loader, directory = spaces
    (directory / "bad.yaml").write_text(
        "name: bad\nclasses: [benign]\nexcluded:\n  ddos: {flows: 5}\n")
    with pytest.raises(LabelSpaceError, match="no reason"):
        loader.load("bad")


@pytest.mark.parametrize("reason", [
    "we could not estimate it",      # prose
    "it's-unestimable",              # an apostrophe would break generated SQL
    "Unestimable",                   # grouped on downstream, so case is not free
])
def test_a_reason_must_be_an_identifier(spaces, reason):
    """It is written into every out-of-space row, grouped on downstream, AND
    interpolated into SQL. All three want the same narrow shape."""
    loader, directory = spaces
    (directory / "bad.yaml").write_text(
        "name: bad\nclasses: [benign]\n"
        f"excluded:\n  ddos: {{reason: {reason!r}}}\n")
    with pytest.raises(LabelSpaceError, match="is not an identifier"):
        loader.load("bad")


def test_a_class_the_taxonomy_never_declared_is_refused(spaces):
    """A typo would silently exclude nothing and predict nothing."""
    loader, directory = spaces
    (directory / "bad.yaml").write_text("name: bad\nclasses: [benign, ddoss]\n")
    with pytest.raises(LabelSpaceError, match="taxonomy does not declare"):
        loader.load("bad", TaxonomyMapper())


def test_a_duplicated_class_is_refused(spaces):
    loader, directory = spaces
    (directory / "bad.yaml").write_text("name: bad\nclasses: [benign, ddos, ddos]\n")
    with pytest.raises(LabelSpaceError, match="more than once"):
        loader.load("bad")


def test_the_exclusion_sql_is_generated_from_the_declaration(spaces):
    """So a class added to `excluded:` is flagged without editing the writer."""
    loader, _ = spaces
    sql = loader.load("toy-space").out_of_space_sql("label_class")
    assert "label_class = 'heartbleed'" in sql and "'unestimable'" in sql


def test_a_space_with_nothing_excluded_yields_a_typed_null(spaces):
    loader, directory = spaces
    (directory / "all.yaml").write_text("name: all\nclasses: [benign, ddos]\n")
    assert loader.load("all").out_of_space_sql() == "CAST(NULL AS VARCHAR)"


# ------------------------------------------------------------ the declaration

def test_a_declaration_names_an_implementation_and_a_space(methods):
    loader, _ = methods
    spec = loader.load("toy-v1")
    assert spec.name == "toy-v1" and spec.implementation == "schedule"
    assert spec.label_space.name == "toy-space"


def test_the_declaration_name_is_what_scopes_the_output_not_the_implementation(methods):
    """Two declarations share a class. If the implementation name won, the
    second run would overwrite the first and `label_method` would not say which."""
    loader, directory = methods
    (directory / "toy-v2").mkdir()
    (directory / "toy-v2" / "method.yaml").write_text(
        METHOD.replace("toy-v1", "toy-v2"))

    one, two = loader.load("toy-v1"), loader.load("toy-v2")
    assert one.implementation == two.implementation == "schedule"
    assert one.name != two.name


def test_an_unknown_implementation_is_refused_and_says_how_to_add_one(methods):
    loader, directory = methods
    (directory / "bad").mkdir()
    (directory / "bad" / "method.yaml").write_text(
        "name: bad\nimplementation: telepathy\nlabel_space: toy-space\n")
    with pytest.raises(MethodError, match="not a registered labelling method"):
        loader.load("bad")


def test_a_name_that_could_not_be_a_directory_is_refused(methods):
    loader, directory = methods
    (directory / "bad").mkdir()
    (directory / "bad" / "method.yaml").write_text(
        "name: a/b\nimplementation: schedule\nlabel_space: toy-space\n")
    with pytest.raises(MethodError, match="path separator"):
        loader.load("bad")


# -------------------------------------------------------------- the composite

def test_the_method_sha_moves_when_the_declaration_moves(methods):
    loader, directory = methods
    before = loader.load("toy-v1").method_sha()
    (directory / "toy-v1" / "method.yaml").write_text(METHOD + "settings: {a: 1}\n")
    assert loader.load("toy-v1").method_sha() != before


def test_the_method_sha_moves_when_the_label_space_moves(methods, spaces):
    """Excluding one more class changes the table, so it has to change the hash."""
    loader, _ = methods
    _, space_dir = spaces
    before = loader.load("toy-v1").method_sha()
    (space_dir / "toy-space.yaml").write_text(
        SPACE + "  botnet: {reason: too-few}\n")
    assert loader.load("toy-v1").method_sha() != before


def test_extra_inputs_join_the_hash(methods, tmp_path):
    """Schedule labelling adds its per-dataset manifest, which only it knows about."""
    loader, _ = methods
    spec = loader.load("toy-v1")
    extra = tmp_path / "manifest.yaml"
    extra.write_text("dataset: toy\n")
    assert spec.method_sha(extra) != spec.method_sha()


# -------------------------------------------------- the shipped declarations

def test_the_shipped_schedule_declaration_loads():
    spec = MethodLoader().load("schedule")
    assert spec.implementation == "schedule" and spec.built_in
    assert spec.label_space.name == "core-5"


def test_core_5_holds_the_five_cross_domain_classes():
    space = LabelSpaceLoader().load("core-5", TaxonomyMapper())
    assert space.classes == ("benign", "ddos", "dos", "brute_force", "botnet")
    assert set(space.excluded) == {
        "portscan", "web_attack", "infiltration", "heartbleed"}


def test_every_core_5_exclusion_states_evidence_and_a_reason():
    """The counts are the argument. An exclusion without one is an assertion."""
    space = LabelSpaceLoader().load("core-5", TaxonomyMapper())
    for label, exclusion in space.excluded.items():
        assert exclusion.reason and " " not in exclusion.reason, label
        assert exclusion.flows is not None and exclusion.domains is not None, label
        assert exclusion.note, label


def test_portscan_is_excluded_for_the_protocol_not_for_the_data():
    """238k flows and genuinely detectable -- it fails leave-one-domain-out only
    because it exists in one domain. The reason has to say so."""
    space = LabelSpaceLoader().load("core-5", TaxonomyMapper())
    portscan = space.excluded["portscan"]
    assert portscan.reason == "single-domain" and portscan.domains == 1
    assert portscan.flows > 200_000


def test_a_user_can_declare_their_own_label_space(tmp_path, monkeypatch):
    monkeypatch.setattr(SPACES, "_added", [])
    (tmp_path / "binary.yaml").write_text(
        "name: binary\nclasses: [benign, ddos]\n")
    SPACES.add(tmp_path)
    space = LabelSpaceLoader().load("binary")
    assert space.n_classes == 2
    assert SPACES.origin_of("binary").where == SEARCHED


# ------------------------------------------------------- what the writer emits

def test_out_of_space_is_a_quality_flag_not_a_filter(spaces):
    """The distinction the July regression targets depend on: an out-of-space
    flow keeps its class and its label_binary, so the COUNTS do not move. The
    exclusion happens when pools are built, not when labels are written."""
    from talos.data.labelling.schedule.closed_world import ClosedWorldResolver

    loader, _ = spaces
    space = loader.load("toy-space")
    sql = ClosedWorldResolver().quality_sql(
        has_regions=False, out_of_space=space.out_of_space_sql())
    assert OUT_OF_SPACE in sql
    assert "label_binary" not in sql and "label_class" not in sql.split("AS")[0]


def test_a_flow_that_is_both_out_of_space_and_quarantined_keeps_the_region_reason(spaces):
    """W9.9, run through DuckDB rather than asserted by string position.

    `label_quality` gives out-of-space precedence -- it excludes the flow from
    every pool, while `uncertain` only marks it. `quarantine_reason` resolves the
    tie the OTHER way, because the region's reason cannot be recovered from the
    row while the out-of-space reason can: look up `label_class` in the space and
    it comes back.
    """
    import duckdb

    from talos.data.labelling.schedule.closed_world import ClosedWorldResolver
    from talos.data.labelling.schedule.join import QUARANTINED

    space = spaces[0].load("toy-space")
    columns = ClosedWorldResolver().quality_sql(
        has_regions=True, out_of_space=space.out_of_space_sql())

    rows = duckdb.connect().sql(f"""
        WITH flows(label_class, qreason) AS (VALUES
            ('heartbleed', 'unadjudicable-tail'),   -- both
            ('heartbleed', NULL),                   -- out-of-space only
            ('ddos',       'unadjudicable-tail'),   -- region only
            ('ddos',       NULL))                   -- neither
        SELECT label_class, {columns}
        FROM flows {QUARANTINED}
    """).fetchall()

    assert rows[0] == ("heartbleed", OUT_OF_SPACE, "unadjudicable-tail")
    assert rows[1] == ("heartbleed", OUT_OF_SPACE, "unestimable")
    assert rows[2] == ("ddos", "uncertain", "unadjudicable-tail")
    assert rows[3] == ("ddos", "certain", None)
