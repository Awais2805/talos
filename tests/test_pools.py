"""D_l and D_s: disjoint by construction, and proved so against real parquet.

The one thing that must never happen is a flow in both pools -- fine-tuning on
rows the encoder already pretrained on makes every number afterwards
meaningless. Membership is therefore computed from a hash rather than sampled
and recorded, and the ranges come from one cumulative walk, so overlap is not
"checked for and absent" but unrepresentable.

Tests run the generated SQL through DuckDB over real parquet. The thing that can
be wrong is the SQL (D4.1), and an arithmetic proof of disjointness says nothing
about whether the predicate was built correctly.
"""

import textwrap

import duckdb
import pytest

from talos.common.duck import DuckEngine
from talos.common.plugins import ConfigPlugins
from talos.data.labelling.behavioural.pool import (
    BY_CAPTURE, BY_UID, PartitionLoader, PoolError,
)

PARTITION = textwrap.dedent("""
    name: toy
    source:
      datasets: [toy]
      labels: schedule
    exclude: [out-of-space]
    partition_by: uid
    seed: "toy-seed"
    buckets: 1000
    pools:
      d_s:  {share: 0.02, role: audited}
      d_l:  {share: 0.78, role: unlabelled}
      held: {share: 0.20, role: held}
""")

CLASSES = ("benign", "ddos", "dos")


@pytest.fixture
def partitions(tmp_path):
    directory = tmp_path / "pools"
    directory.mkdir()
    (directory / "toy.yaml").write_text(PARTITION)
    point = ConfigPlugins("partition", built_in=directory, error=PoolError,
                          addressable=False)
    return PartitionLoader(point), directory


@pytest.fixture
def table(tmp_path):
    """6,000 flows across 3 classes and 4 captures, 300 of them out-of-space."""
    path = tmp_path / "conn.parquet"
    duckdb.connect().sql(f"""
        SELECT
            'flow-' || i                                  AS uid,
            {list(CLASSES)}[(i % 3) + 1]                  AS label_class,
            'capture-' || (i % 4)                         AS capture,
            CASE WHEN i % 20 = 0 THEN 'out-of-space'
                 WHEN i % 21 = 0 THEN 'uncertain'
                 ELSE 'certain' END                       AS label_quality
        FROM range(6000) t(i)
    """).write_parquet(str(path))
    return str(path)


@pytest.fixture
def duck():
    return DuckEngine()


def members(duck, partition, pool, table):
    return {row[0] for row in
            partition.relation(duck, pool, [table]).project("uid").fetchall()}


# ------------------------------------------------------------ the arithmetic

def test_shares_must_total_one(partitions):
    """A total below 1 silently discards the remainder -- so it is not treated as
    a rounding question."""
    loader, directory = partitions
    (directory / "bad.yaml").write_text(
        PARTITION.replace("share: 0.20", "share: 0.10"))
    with pytest.raises(PoolError, match="shares total"):
        loader.load("bad")


def test_ranges_are_contiguous_with_no_gap_or_overlap(partitions):
    loader, _ = partitions
    pools = list(loader.load("toy"))
    assert pools[0].lo == 0
    assert pools[-1].hi == 1000
    for earlier, later in zip(pools, pools[1:]):
        assert earlier.hi == later.lo


def test_rounding_is_cumulative_not_per_pool(partitions):
    """Rounding each share independently accumulates error and leaves a few
    buckets in a gap -- or in two pools. Ten tenths is the case that exposes it."""
    loader, directory = partitions
    body = "\n".join(f"  p{i}: {{share: 0.1}}" for i in range(10))
    (directory / "tenths.yaml").write_text(
        "name: tenths\nseed: s\nbuckets: 1000\npools:\n" + body)

    pools = list(loader.load("tenths"))
    assert [p.n_buckets for p in pools] == [100] * 10
    assert pools[-1].hi == 1000


# ---------------------------------------------------- disjoint, over real data

def test_every_flow_lands_in_exactly_one_pool(duck, partitions, table):
    """The property the whole module exists for, measured rather than argued."""
    loader, _ = partitions
    partition = loader.load("toy")

    sets = {p.name: members(duck, partition, p.name, table) for p in partition}
    d_s, d_l, held = sets["d_s"], sets["d_l"], sets["held"]

    assert d_s & d_l == set()
    assert d_s & held == set()
    assert d_l & held == set()

    total = duck.one(f"SELECT count(*) FROM read_parquet('{table}') "
                     f"WHERE label_quality IS DISTINCT FROM 'out-of-space'")[0]
    assert len(d_s) + len(d_l) + len(held) == total


def test_membership_is_stable_across_runs(duck, partitions, table):
    """Recomputed, never recorded -- so it has to give the same answer twice."""
    loader, _ = partitions
    partition = loader.load("toy")
    assert members(duck, partition, "d_s", table) == \
        members(duck, partition, "d_s", table)


def test_a_different_seed_reshuffles_but_still_partitions(duck, partitions, table):
    loader, directory = partitions
    (directory / "other.yaml").write_text(
        PARTITION.replace('seed: "toy-seed"', 'seed: "other-seed"'))

    first = members(duck, loader.load("toy"), "d_s", table)
    second = members(duck, loader.load("other"), "d_s", table)
    assert first != second
    assert abs(len(first) - len(second)) < len(first)      # same order of size


def test_the_seed_is_prepended_not_added(partitions):
    """Adding a seed to the hash shifts every flow by one constant and leaves the
    ordering intact, so two seeds give rotations of each other rather than
    independent draws."""
    loader, _ = partitions
    sql = loader.load("toy")["d_s"].bucket_sql()
    assert "'toy-seed' || CAST" in sql


# ------------------------------------------------------------- stratification

def test_a_share_is_proportional_within_every_class(duck, partitions, table):
    """Stratification is free: a hash is uniform within any subset, so nobody has
    to ask for it. What is NOT free is a per-class floor -- see the watch-out."""
    loader, _ = partitions
    partition = loader.load("toy")
    rows = partition.relation(duck, "d_l", [table]).aggregate(
        "label_class, count(*)").fetchall()
    per_class = dict(rows)

    assert set(per_class) == set(CLASSES)
    smallest, largest = min(per_class.values()), max(per_class.values())
    assert largest - smallest < 0.2 * largest          # proportional, not exact


def test_a_tiny_share_of_a_rare_class_can_be_empty(duck, partitions, tmp_path):
    """The watch-out, pinned as a test so it is not discovered later: uniform
    sampling gives a rare class nothing, and D_s needs a per-class floor that
    this module deliberately does not provide."""
    loader, _ = partitions
    partition = loader.load("toy")
    rare = tmp_path / "rare.parquet"
    duckdb.connect().sql(
        "SELECT 'r-' || i AS uid, 'heartbleed' AS label_class, "
        "'c' AS capture, 'certain' AS label_quality FROM range(3) t(i)"
    ).write_parquet(str(rare))

    assert members(duck, partition, "d_s", str(rare)) == set()


# --------------------------------------------------------------- leakage safety

def test_keying_on_capture_never_splits_a_capture(duck, partitions, table):
    """W5.9: capture is the leakage-safe unit. Keyed that way, a capture is
    wholly in one pool or wholly out of it."""
    loader, directory = partitions
    (directory / "bycap.yaml").write_text(
        PARTITION.replace("partition_by: uid", "partition_by: capture"))
    partition = loader.load("bycap")

    seen: dict[str, str] = {}
    for pool in partition:
        rows = partition.relation(duck, pool.name, [table]).aggregate(
            "capture").fetchall()
        for (capture,) in rows:
            assert capture not in seen, f"{capture} split across pools"
            seen[capture] = pool.name
    assert len(seen) == 4


def test_the_key_is_declared_and_a_typo_is_refused(partitions):
    loader, directory = partitions
    (directory / "bad.yaml").write_text(
        PARTITION.replace("partition_by: uid", "partition_by: flow"))
    with pytest.raises(PoolError, match="partition_by"):
        loader.load("bad")
    assert BY_UID in ("uid",) and BY_CAPTURE in ("capture",)


# ------------------------------------------------------------- the exclusions

def test_out_of_space_flows_never_enter_any_pool(duck, partitions, table):
    """Where the label space finally bites. Gate 9 flags them and leaves them in
    the labelled table; this is the line at which they stop being training data."""
    loader, _ = partitions
    partition = loader.load("toy")
    for pool in partition:
        rows = partition.relation(duck, pool.name, [table]).filter(
            "label_quality = 'out-of-space'").count("*").fetchone()[0]
        assert rows == 0


def test_uncertain_flows_are_kept_unless_asked_for(duck, partitions, table):
    """Confident learning exists to weigh uncertain rows, not to discard them."""
    loader, _ = partitions
    partition = loader.load("toy")
    kept = sum(partition.relation(duck, p.name, [table])
               .filter("label_quality = 'uncertain'").count("*").fetchone()[0]
               for p in partition)
    assert kept > 0


def test_a_null_quality_is_not_silently_dropped(duck, partitions, tmp_path):
    """A NOT IN against NULL is NULL, not true -- which would quietly drop every
    row from a method that leaves the column empty."""
    loader, _ = partitions
    partition = loader.load("toy")
    nulls = tmp_path / "nulls.parquet"
    duckdb.connect().sql(
        "SELECT 'n-' || i AS uid, 'benign' AS label_class, 'c' AS capture, "
        "CAST(NULL AS VARCHAR) AS label_quality FROM range(500) t(i)"
    ).write_parquet(str(nulls))

    kept = sum(len(members(duck, partition, p.name, str(nulls)))
               for p in partition)
    assert kept == 500


def test_an_unknown_exclusion_value_is_refused(partitions):
    """A typo would silently exclude nothing."""
    loader, directory = partitions
    (directory / "bad.yaml").write_text(
        PARTITION.replace("exclude: [out-of-space]", "exclude: [out_of_space]"))
    with pytest.raises(PoolError, match="not a label_quality value"):
        loader.load("bad")


# -------------------------------------------------------------- the refusals

def test_an_unseeded_partition_is_refused(partitions):
    """Without a seed the split is a property of the DuckDB build, not of the
    declaration, so it is not reproducible from the declaration alone."""
    loader, directory = partitions
    (directory / "bad.yaml").write_text(
        PARTITION.replace('seed: "toy-seed"', "seed: \"\""))
    with pytest.raises(PoolError, match="no seed"):
        loader.load("bad")


def test_a_partition_with_no_pools_is_refused(partitions):
    loader, directory = partitions
    (directory / "bad.yaml").write_text("name: bad\nseed: s\npools: {}\n")
    with pytest.raises(PoolError, match="no pools"):
        loader.load("bad")


@pytest.mark.parametrize("share,message", [
    ("share: 2", "not between 0 and 1"),
    ("share: '0.5'", "must be a number"),
])
def test_a_share_that_is_not_a_fraction_is_refused(partitions, share, message):
    loader, directory = partitions
    (directory / "bad.yaml").write_text(
        f"name: bad\nseed: s\npools:\n  a: {{{share}}}\n")
    with pytest.raises(PoolError, match=message):
        loader.load("bad")


def test_a_pool_name_that_could_not_be_an_identifier_is_refused(partitions):
    loader, directory = partitions
    (directory / "bad.yaml").write_text(
        "name: bad\nseed: s\npools:\n  'D S': {share: 1.0}\n")
    with pytest.raises(PoolError, match="pool name"):
        loader.load("bad")


# ------------------------------------------------------ the shipped partition

def test_the_shipped_partition_loads_and_totals_one():
    partition = PartitionLoader().load("xdg-v3")
    assert set(partition.pools) == {"d_s", "d_l", "held"}
    assert list(partition)[-1].hi == partition.buckets


def test_the_holdout_domain_is_absent_from_every_pool():
    """cic-ddos-2019 is `role: holdout` in experiments/xdg-v3. A pool that
    quietly included it would defeat the guard that role exists for."""
    partition = PartitionLoader().load("xdg-v3")
    assert "cic-ddos-2019" not in partition.datasets
    assert set(partition.datasets) == {"cic-ids-2017", "cic-ids-2018"}


def test_the_shipped_partition_excludes_out_of_space_but_keeps_uncertain():
    partition = PartitionLoader().load("xdg-v3")
    assert partition.exclude == ("out-of-space",)


def test_the_pool_sha_moves_when_the_declaration_moves(partitions):
    loader, directory = partitions
    before = loader.load("toy").sha
    (directory / "toy.yaml").write_text(PARTITION + "\n# a comment\n")
    assert loader.load("toy").sha != before
