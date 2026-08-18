"""The labelling-method contract: schedule labelling as a method among methods.

Two things are pinned. The output CONTRACT -- the core ten columns every method
owes, whatever it is -- because fusion, confident learning, the benchmark and
training all read it without asking who wrote the table. And the fact that
schedule labelling now goes through that contract like anything else, so the
three-way benchmark is an ordinary operation rather than a special case.

The engine's own behaviour is tested in test_labelling_engine.py and is
deliberately untouched here: gate 8 moved it and wrapped it, it did not change it.
"""

import textwrap

import duckdb
import pytest

from talos.common.config import Config
from talos.common.plugins import CodePlugins
from talos.common.provenance import composite_sha
from talos.data.extraction.base import CONN_BACKBONE, FLOW_ID
from talos.data.labelling.base import (
    CLOSED_WORLD, CORE_COLUMNS, CORE_TYPES, LABEL_SOURCES, LABELLERS, MATCHED,
    LabellingMethod, SchemaError, verify_schema,
)
from talos.data.labelling.schedule import ScheduleLabeller
from talos.data.labelling.schedule.engine import LabellingEngine
from talos.data.labelling.schedule.writer import EXTRAS

MANIFEST = textwrap.dedent("""
    dataset: toy
    utc_offset: "-03:00"
    attacks:
      - {name: FTP-Patator, date: 2017-07-04, start: "09:20", end: "10:20",
         attacker_ips: [10.0.0.1], victim_ips: [10.0.0.2]}
""")

TAXONOMY = textwrap.dedent("""
    canonical_classes:
      benign:
      brute_force:
    raw_to_canonical:
      FTP-Patator: brute_force
    requires_payload: [FTP-Patator]
""")


@pytest.fixture
def cfg(tmp_path):
    return Config({"lake": {"root": str(tmp_path / "lake")}})


@pytest.fixture
def method(tmp_path, cfg):
    """A ScheduleLabeller with its own manifest and taxonomy, both editable."""
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "toy.yaml").write_text(MANIFEST)
    taxonomy = tmp_path / "toy-taxonomy.yaml"
    taxonomy.write_text(TAXONOMY)
    return ScheduleLabeller(cfg, manifests=manifests, taxonomy=taxonomy)


# ----------------------------------------------------- schedule is a method now

def test_schedule_is_registered_as_a_labelling_method():
    assert "schedule" in LABELLERS
    assert LABELLERS.cls_for("schedule") is ScheduleLabeller


def test_the_adapter_and_the_engine_agree_on_the_method_name():
    """They are written in two files and both reach the output path and the rows.
    If they drifted, `--method schedule` would write somewhere `schedule` is not."""
    assert ScheduleLabeller.name == LabellingEngine.method


def test_schedule_declares_what_it_needs_from_an_extractor():
    """Structural, both of them: without one record per connection and a stable
    per-flow id there is nothing to attach a schedule label to."""
    assert set(ScheduleLabeller.requires) == {CONN_BACKBONE, FLOW_ID}


def test_byte_counts_are_not_required_because_their_absence_degrades_one_column():
    from talos.data.extraction.base import BYTE_COUNTS
    assert BYTE_COUNTS not in ScheduleLabeller.requires


def test_a_method_refuses_an_extractor_that_cannot_support_it(cfg, method):
    from talos.data.extraction.base import BaseExtractor, ExtractionError, ExtractionReport

    class Packets(BaseExtractor):
        name = "packets-only"
        version = "1"
        def capabilities(self):
            return frozenset()
        def extract(self, pcap_paths, output_dir):
            return ExtractionReport()

    with pytest.raises(ExtractionError, match="conn_backbone"):
        method.check_extractor(Packets())


# ------------------------------------------------------------ the output path

def test_each_method_writes_to_its_own_place(cfg, method):
    """Three methods labelling one dataset must not overwrite each other."""
    uri = method.output_uri("toy", "zeek_v8.2.1", source="datasets")
    assert uri.endswith("labelled/zeek_v8.2.1/schedule/toy/conn.parquet")


def test_two_methods_resolve_to_different_paths_for_the_same_dataset(cfg, method):
    class Pretend(LabellingMethod):
        name = "ae"
        def method_sha(self, dataset): return "0" * 12
        def label(self, dataset, feature_space, **kwargs): raise NotImplementedError

    other = Pretend(cfg, lake=method.lake)
    assert method.output_uri("toy", "fs") != other.output_uri("toy", "fs")


# ------------------------------------------------------------- the method sha

def test_the_method_sha_covers_the_manifest(method, tmp_path):
    before = method.method_sha("toy")
    (tmp_path / "manifests" / "toy.yaml").write_text(
        MANIFEST.replace('end: "10:20"', 'end: "10:30"'))
    assert method.method_sha("toy") != before


def test_the_method_sha_covers_the_taxonomy_too(method, tmp_path):
    """The one that matters. Remapping a raw name changes every label the
    schedule produced WITHOUT touching the schedule -- a hash over the manifest
    alone would call the two runs identical."""
    before = method.method_sha("toy")
    (tmp_path / "toy-taxonomy.yaml").write_text(
        TAXONOMY.replace("brute_force", "dos"))
    assert method.method_sha("toy") != before


def test_the_composite_sha_does_not_depend_on_the_order_of_its_inputs(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("one")
    b.write_text("two")
    assert composite_sha(a, b) == composite_sha(b, a)


def test_the_composite_sha_is_a_sha12(method):
    sha = method.method_sha("toy")
    assert len(sha) == 12 and all(c in "0123456789abcdef" for c in sha)


# ---------------------------------------------------------------- the contract

def relation(**overrides):
    """A relation with every core column and schedule's extras, correctly typed."""
    literal = {
        "BOOLEAN": "true", "VARCHAR": "'x'", "DOUBLE": "CAST(NULL AS DOUBLE)",
    }
    cols = {name: literal[CORE_TYPES[name]] for name in CORE_COLUMNS}
    cols |= {"label_raw": "'x'", "label_executed": "true", "rule_id": "1",
             "rules_matched": "1", "quarantine_reason": "'x'"}
    cols |= overrides
    for dropped in [k for k, v in cols.items() if v is None]:
        del cols[dropped]
    return duckdb.connect().sql(
        "SELECT " + ", ".join(f"{expr} AS {name}" for name, expr in cols.items()))


def test_a_correct_relation_passes():
    verify_schema(relation(), EXTRAS)


@pytest.mark.parametrize("column", CORE_COLUMNS)
def test_a_missing_core_column_is_refused(column):
    """Every one of the ten, not just the ones someone remembered to test."""
    with pytest.raises(SchemaError, match=column):
        verify_schema(relation(**{column: None}), EXTRAS)


def test_a_declared_extra_that_was_not_emitted_is_refused():
    """A method that advertises a column and does not write it is worse than one
    that never advertised it: a reader has been told it can rely on it."""
    with pytest.raises(SchemaError, match="rule_id"):
        verify_schema(relation(rule_id=None), EXTRAS)


def test_a_wrongly_typed_core_column_is_refused():
    """Presence is not enough. The same column arriving as two types depending on
    which method wrote it makes a WHERE clause behave differently per dataset."""
    with pytest.raises(SchemaError, match="label_binary is INTEGER"):
        verify_schema(relation(label_binary="1"), EXTRAS)


def test_confidence_may_be_null_but_may_not_be_absent():
    """The distinction the whole core/extras split rests on: a method with no
    confidence emits a typed NULL, it does not omit the column."""
    verify_schema(relation(label_confidence="CAST(NULL AS DOUBLE)"), EXTRAS)
    with pytest.raises(SchemaError, match="label_confidence"):
        verify_schema(relation(label_confidence=None), EXTRAS)


def test_label_source_is_a_shared_vocabulary_not_a_per_method_one():
    """The benchmark groups on it, so `matched` has to mean the same thing
    whether a rule or an oracle produced it."""
    assert MATCHED in LABEL_SOURCES and CLOSED_WORLD in LABEL_SOURCES
    assert not any(source.startswith("schedule") for source in LABEL_SOURCES)


# ------------------------------------------------------- bring your own method

def test_a_user_can_register_their_own_labelling_method(tmp_path, monkeypatch):
    """`talos label --method theirs`, with no edit to this package."""
    import talos.common.plugins as plugins

    point = CodePlugins("labelling method", addressable=False)
    monkeypatch.setattr(plugins, "UNDER_TEST", point, raising=False)
    (tmp_path / "mine.py").write_text(textwrap.dedent("""
        import talos.common.plugins as plugins
        from talos.data.labelling.base import LabellingMethod

        @plugins.UNDER_TEST.register
        class Mine(LabellingMethod):
            name = "entropy-v1"
            extras = ("entropy",)
            def method_sha(self, dataset): return "a" * 12
            def label(self, dataset, feature_space, **kwargs): return None
    """))

    assert point.add(tmp_path / "mine.py") == ("entropy-v1",)
    assert point.cls_for("entropy-v1").extras == ("entropy",)
    assert point.origin_of("entropy-v1").where == "external"
