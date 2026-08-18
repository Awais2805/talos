"""Phase 3 — task relevance and the domain probe, on hand-computable fixtures.

Each feature is planted so its verdict is arithmetic: one that determines the
label and hides its domain, one that determines the domain and says nothing
about the label, one that does both, and one that does neither.
"""

from __future__ import annotations

import math

import duckdb
import pytest

from talos.data.preprocess.relevance import (
    DROP, HOLD, KEEP, UNMEASURED, RelevanceAnalyst, _mi_bits,
)

import numpy as np


class Duck:
    def __init__(self, con):
        self.con = con

    def sql(self, query):
        return self.con.execute(query).fetchall()

    def one(self, query):
        return self.con.execute(query).fetchone()


@pytest.fixture
def duck():
    con = duckdb.connect()
    # 2000 rows over two domains and two classes, balanced.
    con.execute("""
        CREATE TABLE flows AS SELECT
            CASE WHEN i % 2 = 0 THEN 'benign' ELSE 'ddos' END      AS label_class,
            CASE WHEN i < 1000 THEN 'd2017' ELSE 'd2018' END       AS dataset,
            -- determines the label, identical in both domains
            CASE WHEN i % 2 = 0 THEN 1.0 ELSE 100.0 END            AS tells_label,
            -- determines the domain, independent of the label
            CASE WHEN i < 1000 THEN 1.0 ELSE 100.0 END             AS tells_domain,
            -- determines both
            CASE WHEN i < 1000 THEN (CASE WHEN i % 2 = 0 THEN 1.0 ELSE 10.0 END)
                 ELSE (CASE WHEN i % 2 = 0 THEN 100.0 ELSE 1000.0 END) END
                                                                    AS tells_both,
            -- determines neither
            CAST(i % 7 AS DOUBLE)                                   AS noise,
            CASE WHEN i % 2 = 0 THEN 'tcp' ELSE 'udp' END           AS proto_label
        FROM range(0, 2000) t(i)""")
    return Duck(con)


def analyse(duck, **kwargs):
    return RelevanceAnalyst(duck, **kwargs).analyse(
        "toy", "flows",
        ["tells_label", "tells_domain", "tells_both", "noise"],
        ["proto_label"])


# ------------------------------------------------------------------ the MI

def test_mi_of_a_perfect_predictor_is_the_label_entropy():
    """Two balanced classes carry 1 bit; a feature that fixes them carries it all."""
    table = np.array([[500.0, 0.0], [0.0, 500.0]])
    mi, normalised = _mi_bits(table)
    assert mi == pytest.approx(1.0)
    assert normalised == pytest.approx(1.0)


def test_mi_of_an_independent_feature_is_zero():
    """Independence must read as zero, not as a small positive artefact."""
    table = np.array([[250.0, 250.0], [250.0, 250.0]])
    mi, normalised = _mi_bits(table)
    assert mi == pytest.approx(0.0, abs=1e-12)
    assert normalised == pytest.approx(0.0, abs=1e-12)


def test_mi_is_never_negative():
    """Near-cancelling logs can land a few ulps below zero; -0.0000 reads as a bug."""
    table = np.array([[3333.0, 3333.0, 3334.0], [3333.0, 3334.0, 3333.0]])
    mi, normalised = _mi_bits(table)
    assert mi >= 0.0 and normalised >= 0.0
    assert not math.copysign(1.0, mi) < 0


def test_mi_is_normalised_by_the_label_entropy(duck):
    """Raw bits grow with bin count; the ratio is what compares across features."""
    analyst = RelevanceAnalyst(duck)
    mi, normalised = analyst.mutual_information(
        "flows", "tells_label", "label_class", numeric=True)
    assert normalised == pytest.approx(1.0, abs=0.01)
    assert mi == pytest.approx(1.0, abs=0.01)


def test_a_useless_feature_scores_near_zero_relevance(duck):
    """`noise` is i%7 against a label of i%2 — coprime, so close to independent."""
    analyst = RelevanceAnalyst(duck)
    _mi, normalised = analyst.mutual_information(
        "flows", "noise", "label_class", numeric=True)
    assert normalised < 0.05


def test_a_categorical_feature_is_scored_without_binning(duck):
    """Its values are already the bins; imposing quantiles on them would be wrong."""
    analyst = RelevanceAnalyst(duck)
    _mi, normalised = analyst.mutual_information(
        "flows", "proto_label", "label_class", numeric=False)
    assert normalised == pytest.approx(1.0, abs=0.01)


def test_nulls_get_their_own_bin_rather_than_being_dropped(duck):
    """Absence is a fact about the flow; dropping those rows scores a different pool."""
    duck.con.execute("""CREATE TABLE holey AS SELECT
        CASE WHEN i % 2 = 0 THEN 'benign' ELSE 'ddos' END AS label_class,
        CASE WHEN i % 2 = 0 THEN NULL ELSE 1.0 END        AS x
        FROM range(0, 1000) t(i)""")
    analyst = RelevanceAnalyst(duck)
    _mi, normalised = analyst.mutual_information(
        "holey", "x", "label_class", numeric=True)
    # NULL occurs exactly for benign, so absence alone determines the label.
    assert normalised == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------- the domain AUC

def test_a_domain_determining_feature_scores_auc_one(duck):
    """`tells_domain` is 1.0 in one domain and 100.0 in the other."""
    analyst = RelevanceAnalyst(duck)
    assert analyst.domain_auc(
        "flows", "tells_domain", "dataset", "d2017", numeric=True) == \
        pytest.approx(1.0)


def test_a_domain_blind_feature_scores_auc_half(duck):
    """`tells_label` has the same distribution in both domains."""
    analyst = RelevanceAnalyst(duck)
    auc = analyst.domain_auc(
        "flows", "tells_label", "dataset", "d2017", numeric=True)
    assert auc == pytest.approx(0.5, abs=0.01)


def test_auc_direction_is_ignored(duck):
    """A feature that is perfectly LOW in one domain identifies it just as well."""
    duck.con.execute("""CREATE TABLE flipped AS SELECT
        CASE WHEN i < 500 THEN 'a' ELSE 'b' END AS dataset,
        CASE WHEN i < 500 THEN 100.0 ELSE 1.0 END AS x
        FROM range(0, 1000) t(i)""")
    analyst = RelevanceAnalyst(duck)
    assert analyst.domain_auc(
        "flipped", "x", "dataset", "a", numeric=True) == pytest.approx(1.0)


def test_ties_use_midranks(duck):
    """A constant feature must score 0.5, not 1.0 from an arbitrary rank order."""
    duck.con.execute("""CREATE TABLE flat AS SELECT
        CASE WHEN i < 500 THEN 'a' ELSE 'b' END AS dataset,
        1.0 AS x FROM range(0, 1000) t(i)""")
    analyst = RelevanceAnalyst(duck)
    assert analyst.domain_auc(
        "flat", "x", "dataset", "a", numeric=True) == pytest.approx(0.5)


def test_a_categorical_is_scored_by_its_domain_rate(duck):
    """No order exists, so target encoding stands in for the one-hot model."""
    duck.con.execute("""CREATE TABLE cats AS SELECT
        CASE WHEN i < 500 THEN 'a' ELSE 'b' END AS dataset,
        CASE WHEN i < 500 THEN 'x' ELSE 'y' END AS c FROM range(0, 1000) t(i)""")
    analyst = RelevanceAnalyst(duck)
    assert analyst.domain_auc(
        "cats", "c", "dataset", "a", numeric=False) == pytest.approx(1.0)


# ----------------------------------------------------------- the verdicts

def test_the_four_planted_features_get_their_hand_computed_verdicts(duck):
    """The whole cross-reference, on one fixture."""
    verdicts = {f.name: f.verdict for f in analyse(duck).features}
    assert verdicts["tells_label"] == KEEP     # low AUC, high relevance
    assert verdicts["tells_domain"] == DROP    # high AUC, low relevance
    assert verdicts["tells_both"] == HOLD      # both high
    assert verdicts["noise"] == KEEP           # low AUC, low relevance


def test_high_and_high_is_held_not_decided(duck):
    """Only measured performance under LODO can say whether it is signal or leak."""
    both = next(f for f in analyse(duck).features if f.name == "tells_both")
    assert both.verdict == HOLD
    assert "Held, not decided" in both.reason


def test_every_verdict_carries_its_numbers(duck):
    """A verdict with no evidence cannot be argued with, only believed."""
    for feature in analyse(duck).features:
        assert feature.reason
        if feature.verdict != UNMEASURED:
            assert feature.domain_auc is not None


def test_the_relevance_cut_is_the_median_and_is_recorded(duck):
    """Self-calibrating, so no absolute number needs defending -- but it is on record."""
    report = analyse(duck)
    scored = sorted(f.mi_normalised for f in report.features
                    if f.domain_auc is not None)
    assert report.relevance_cut == pytest.approx(float(np.median(scored)))
    assert report.to_dict()["thresholds"]["relevance_cut"] == report.relevance_cut


def test_one_domain_leaves_every_verdict_unmeasured(duck):
    """A probe with nothing to discriminate must refuse, not report 0.5."""
    duck.con.execute("""CREATE TABLE solo AS SELECT
        'benign' AS label_class, 'only' AS dataset, CAST(i AS DOUBLE) AS x
        FROM range(0, 100) t(i)""")
    report = RelevanceAnalyst(duck).analyse("solo", "solo", ["x"], [])
    assert report.of(UNMEASURED) == ("x",)
    assert "at least two" in report.joint_note


def test_the_worst_domain_decides_the_leak(duck):
    """A feature giving ONE domain away is unusable however well it hides the rest."""
    report = analyse(duck)
    domain = next(f for f in report.features if f.name == "tells_domain")
    assert domain.domain_auc == max(domain.per_domain.values())


def test_held_features_stay_in_the_schema(duck):
    """Only `drop` leaves; the post-training stage is what decides HELD."""
    report = analyse(duck)
    assert set(report.of(HOLD)) & {"tells_both"}
    assert "tells_both" not in report.of(DROP)
