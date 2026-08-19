"""The resolver that decides which rows a stage reads.

`zones.fill` truncates at the first unfilled placeholder. That is right for
addressing a whole zone and wrong for reading data: a caller who forgets
`method=` gets the zone's parent back, and a glob under it reads every
method's and every dataset's flows as one table with no error anywhere. EDA
did exactly this for its whole life. These tests pin the refusal that replaces
it, and the log-type targeting that keeps `dns`/`http`/`ssl` out of a table
that claims to be flows.
"""

from __future__ import annotations

import pytest

from talos.common.config import Config
from talos.data.flows import CONN, FlowTable, SourceError, globs, tables


@pytest.fixture
def lake(tmp_path):
    cfg = Config({"lake": {"root": str(tmp_path / "lake")}})
    return cfg, cfg.lake()


def touch(tmp_path, *parts):
    p = tmp_path.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    return p


# ------------------------------------------------------------ the refusal

def test_the_labelled_zone_refuses_without_a_method(lake):
    cfg, client = lake
    t = FlowTable(dataset="ds", zone="labelled", feature_space="zeek_v8.2.1")
    with pytest.raises(SourceError) as exc:
        t.base(cfg, client)
    assert "method" in str(exc.value)


def test_the_refusal_explains_what_truncation_would_have_read(lake):
    cfg, client = lake
    with pytest.raises(SourceError) as exc:
        FlowTable(dataset="ds", zone="labelled",
                  feature_space="zeek_v8.2.1").base(cfg, client)
    assert "every method" in str(exc.value)


def test_the_parquet_zone_refuses_without_a_feature_space(lake):
    cfg, client = lake
    with pytest.raises(SourceError) as exc:
        FlowTable(dataset="ds", zone="parquet").base(cfg, client)
    assert "feature_space" in str(exc.value)


def test_an_unknown_zone_is_refused(lake):
    cfg, client = lake
    with pytest.raises(SourceError):
        FlowTable(dataset="ds", zone="nonsense").base(cfg, client)


def test_a_fully_scoped_table_resolves(lake):
    cfg, client = lake
    t = FlowTable(dataset="ds", zone="labelled",
                  feature_space="zeek_v8.2.1", method="schedule")
    assert t.base(cfg, client).endswith("labelled/zeek_v8.2.1/schedule/ds")


# --------------------------------------------------------------- identity

def test_each_method_is_a_distinct_view_and_slug():
    kw = dict(dataset="ds", zone="labelled", feature_space="fs")
    sched = FlowTable(method="schedule", **kw)
    ae = FlowTable(method="ae", **kw)
    assert sched.view == "postlabel-schedule" and ae.view == "postlabel-ae"
    assert sched.slug != ae.slug


def test_the_parquet_zone_is_the_prelabel_view():
    t = FlowTable(dataset="ds", zone="parquet", feature_space="fs")
    assert t.view == "prelabel"
    assert t.slug == "prelabel_ds"


def test_a_method_is_not_carried_by_a_prelabel_table(lake):
    cfg, _ = lake
    (t,) = tables(cfg, ["ds"], "parquet", feature_space="fs", method="schedule")
    assert t.method is None, "a method has no meaning before labelling"
    assert t.view == "prelabel"


# ------------------------------------------------------------- glob shape

def test_the_parquet_glob_targets_conn_not_every_log(lake, tmp_path):
    cfg, client = lake
    touch(tmp_path, "lake/sources/datasets/parquet/fs/ds/Monday/conn.parquet")
    (glob,) = FlowTable(dataset="ds", zone="parquet",
                        feature_space="fs").globs(cfg, client)
    assert glob.endswith("/**/conn.parquet")
    assert "*.parquet" not in glob.replace("/**/conn.parquet", "")


def test_named_captures_become_one_glob_each(lake, tmp_path):
    cfg, client = lake
    touch(tmp_path, "lake/sources/datasets/parquet/fs/ds/Monday/conn.parquet")
    got = FlowTable(dataset="ds", zone="parquet", feature_space="fs",
                    captures=("Monday", "Tuesday")).globs(cfg, client)
    assert [g.rsplit("/", 2)[-2] for g in got] == ["Monday", "Tuesday"]


def test_the_labelled_zone_is_one_consolidated_file(lake, tmp_path):
    cfg, client = lake
    touch(tmp_path, "lake/sources/datasets/labelled/fs/schedule/ds/conn.parquet")
    (glob,) = FlowTable(dataset="ds", zone="labelled", feature_space="fs",
                        method="schedule").globs(cfg, client)
    assert glob.endswith("/schedule/ds/conn.parquet")


def test_another_log_type_can_be_named(lake, tmp_path):
    cfg, client = lake
    touch(tmp_path, "lake/sources/datasets/parquet/fs/ds/Monday/dns.parquet")
    (glob,) = FlowTable(dataset="ds", zone="parquet", feature_space="fs",
                        log="dns").globs(cfg, client)
    assert glob.endswith("/**/dns.parquet")


def test_an_empty_zone_is_refused(lake):
    cfg, client = lake
    with pytest.raises(SourceError) as exc:
        FlowTable(dataset="ds", zone="parquet", feature_space="fs").globs(cfg, client)
    assert "nothing to read" in str(exc.value)


def test_an_explicit_source_bypasses_resolution(lake):
    cfg, client = lake
    assert globs(cfg, client, ["ds"], "labelled", source="/tmp/x.parquet") == \
        ["/tmp/x.parquet"]


def test_several_datasets_resolve_to_several_globs(lake, tmp_path):
    cfg, client = lake
    for ds in ("a", "b"):
        touch(tmp_path, f"lake/sources/datasets/parquet/fs/{ds}/Mon/conn.parquet")
    got = globs(cfg, client, ["a", "b"], "parquet", feature_space="fs")
    assert len(got) == 2 and all(g.endswith(f"/**/{CONN}.parquet") for g in got)
