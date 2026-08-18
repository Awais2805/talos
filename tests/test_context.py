"""ContextFetcher: the flows around a candidate, without one query per row.

Built as two directional equi-joins UNIONed together -- the exact shape
`Corroborator.join_sql` had to be rewritten into after an OR-conditioned join
turned into an unfinishable nested loop on the real 2017 join. Candidates here
are a couple hundred rows, cheap either way, but there's no reason to repeat
a mistake already paid for once.
"""

from __future__ import annotations

import duckdb
import pytest

from talos.common.duck import DuckEngine
from talos.data.labelling.audit.context import ContextFetcher

# Two hosts talk six times close together (a burst), once far outside the
# window, and once in the reverse direction. A third pair never comes up.
# A self-pair host (orig == resp) talks to itself twice, close together --
# the case the double-count guard exists for.
FLOWS = [
    ("burst1", 1000.0, "10.0.0.1", "10.0.0.2"),
    ("burst2", 1010.0, "10.0.0.1", "10.0.0.2"),
    ("burst3", 1020.0, "10.0.0.1", "10.0.0.2"),
    ("reverse", 1030.0, "10.0.0.2", "10.0.0.1"),   # same pair, other direction
    ("far_away", 5000.0, "10.0.0.1", "10.0.0.2"),  # outside the window
    ("other_pair", 1005.0, "10.0.0.9", "10.0.0.8"),
    ("lonely", 2000.0, "10.5.5.5", "10.5.5.6"),     # no neighbours at all
    ("selfpair", 3000.0, "10.9.9.9", "10.9.9.9"),   # orig == resp
    ("selfpair2", 3005.0, "10.9.9.9", "10.9.9.9"),  # its would-be neighbour
]


def _write_parquet(path):
    rows = ", ".join(
        f"('{u}', {ts!r}, 'cap', '{o}', '{r}', 1, 2, 'tcp', 'x', 1.0, 10, 10, 'SF')"
        for u, ts, o, r in FLOWS)
    duckdb.connect().sql(
        f"SELECT * FROM (VALUES {rows}) AS t(uid, ts, capture, \"id.orig_h\", "
        f"\"id.resp_h\", \"id.orig_p\", \"id.resp_p\", proto, service, duration, "
        f"orig_bytes, resp_bytes, conn_state)"
    ).write_parquet(str(path))


@pytest.fixture
def sources(tmp_path):
    path = tmp_path / "conn.parquet"
    _write_parquet(path)
    return [str(path)]


@pytest.fixture
def duck():
    return DuckEngine()


def candidates_relation(duck, *uids):
    chosen = [row for row in FLOWS if row[0] in uids]
    rows = ", ".join(
        f"('{u}', {ts!r}, 'cap', '{o}', '{r}')" for u, ts, o, r in chosen)
    return duck.relation(
        f"SELECT * FROM (VALUES {rows}) AS t(uid, ts, capture, \"id.orig_h\", "
        f"\"id.resp_h\")")


def test_the_burst_finds_every_other_flow_between_the_same_two_hosts(duck, sources):
    context = ContextFetcher(window=60.0).fetch(
        duck, sources, candidates_relation(duck, "burst1"))
    uids = {n["uid"] for n in context["burst1"]}
    assert uids == {"burst2", "burst3", "reverse"}


def test_a_flow_outside_the_window_is_not_a_neighbour(duck, sources):
    context = ContextFetcher(window=60.0).fetch(
        duck, sources, candidates_relation(duck, "burst1"))
    assert "far_away" not in {n["uid"] for n in context["burst1"]}


def test_a_different_host_pair_is_not_a_neighbour(duck, sources):
    context = ContextFetcher(window=60.0).fetch(
        duck, sources, candidates_relation(duck, "burst1"))
    assert "other_pair" not in {n["uid"] for n in context["burst1"]}


def test_the_reverse_direction_still_counts(duck, sources):
    """Same two hosts, source and destination swapped -- still the same
    conversation, matched by the second directional branch."""
    context = ContextFetcher(window=60.0).fetch(
        duck, sources, candidates_relation(duck, "reverse"))
    assert {"burst1", "burst2", "burst3"} <= {n["uid"] for n in context["reverse"]}


def test_neighbours_are_ordered_nearest_first(duck, sources):
    context = ContextFetcher(window=60.0).fetch(
        duck, sources, candidates_relation(duck, "burst1"))
    tss = [float(n["ts"]) for n in context["burst1"]]
    assert tss == sorted(tss, key=lambda t: abs(t - 1000.0))


def test_a_candidate_with_no_neighbours_gets_an_empty_list_not_a_missing_key(duck, sources):
    context = ContextFetcher(window=60.0).fetch(
        duck, sources, candidates_relation(duck, "lonely"))
    assert context["lonely"] == []


def test_a_self_pair_candidate_does_not_double_count_a_match(duck, sources):
    """orig == resp would satisfy both directional branches for the same
    neighbour flow without the guard -- exercised for real here, not just
    reasoned about, by checking the match appears exactly once."""
    context = ContextFetcher(window=60.0).fetch(
        duck, sources, candidates_relation(duck, "selfpair"))
    matches = [n for n in context["selfpair"] if n["uid"] == "selfpair2"]
    assert len(matches) == 1


def test_the_limit_caps_neighbours_shown_nearest_first(duck, sources):
    # burst1 is ts=1000; nearest two by |delta| are burst2 (+10) and burst3
    # (+20), ahead of reverse (+30).
    context = ContextFetcher(window=60.0, limit=2).fetch(
        duck, sources, candidates_relation(duck, "burst1"))
    assert len(context["burst1"]) == 2
    assert {n["uid"] for n in context["burst1"]} == {"burst2", "burst3"}


def test_every_candidate_is_a_key_even_with_multiple_at_once(duck, sources):
    context = ContextFetcher(window=60.0).fetch(
        duck, sources, candidates_relation(duck, "burst1", "lonely", "selfpair"))
    assert set(context) == {"burst1", "lonely", "selfpair"}
