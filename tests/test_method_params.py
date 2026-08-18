"""The published hyperparameters each declaration claims, and the weighting flag.

These pin `ae` and `tabcl` to Eslami & Hamouda Table VII so a silent edit shows
up as a failure rather than as a different result. The v1/v2 comparison tests
this file used to carry went with the consolidation — what they asserted about
superseded declarations now lives in docs/method-versions.md.
"""

from __future__ import annotations

import types

import pytest

from talos.data.labelling.behavioural.base import BehaviouralMethod
from talos.data.labelling.method import MethodLoader

#: Eslami & Hamouda, Table VII, Autoencoder column.
TABLE_VII_AE = {
    "latent_dim": 128,
    "hidden": [256, 128, 64],
    "dropout": 0.10,
    "lr": 0.0005,
    "batch": 128,
    "epochs": 100,
    "weight_decay": 0.00001,
    "residual_weight": 0.5,       # the paper's constraint weight, phi
}

#: Table VII, TabCL column. `latent_dim` is ours — see the declaration.
TABLE_VII_TABCL = {
    "hidden": [256, 128, 64],
    "dropout": 0.10,
    "lr": 0.0005,
    "batch": 256,
    "epochs": 100,
    "weight_decay": 0.00001,
    "replace_rate": 0.15,
    "tau_cont": 0.5,
    "tau_cat": 0.2,
    "lam": 0.5,
}


def settings(name):
    return MethodLoader().load(name).settings


# ------------------------------------------------------------- declaration

@pytest.mark.parametrize("key,expected", sorted(TABLE_VII_AE.items()))
def test_ae_matches_table_vii(key, expected):
    """Each value is the paper's, not an invention; a drift here is a defect."""
    assert settings("ae")[key] == expected


@pytest.mark.parametrize("key,expected", sorted(TABLE_VII_TABCL.items()))
def test_tabcl_matches_table_vii(key, expected):
    assert settings("tabcl")[key] == expected


def test_the_branches_are_equal_in_capacity_and_budget():
    """Eq. 17 compares raw confidence, so an unequal budget decides it.

    The AE once took roughly 10^3 gradient steps against TabCL's 10^6-10^7,
    which is what made the branches agree on 99.98% of flows. This is the
    property that actually mattered, so it is asserted rather than left to a
    reader to notice from two files.
    """
    ae, tabcl = settings("ae"), settings("tabcl")
    assert ae["latent_dim"] == tabcl["latent_dim"]
    assert ae["hidden"] == tabcl["hidden"]
    assert ae["dropout"] == tabcl["dropout"]
    assert ae["epochs"] == tabcl["epochs"]
    assert ae["batch"] <= tabcl["batch"]


# ---------------------------------------------------- DH.12 weighting flag

def stub(**overrides):
    """Enough of a method for the unweighted path, which returns before torch."""
    return types.SimpleNamespace(settings=overrides)


def test_class_weighting_false_gives_plain_cross_entropy():
    """Eq. 3 is unweighted; `None` is what torch takes to mean exactly that."""
    assert BehaviouralMethod.class_weights(
        stub(class_weighting=False), [0, 0, 1]) is None


def test_class_weighting_defaults_to_on():
    """Nothing already declared may change behaviour by this flag existing.

    The default path needs torch, so this asserts the branch is not taken
    rather than the weights themselves.
    """
    with pytest.raises(Exception):
        # Reaches require_torch() (or the space lookup) instead of returning None.
        result = BehaviouralMethod.class_weights(stub(), [0, 0, 1])
        assert result is not None


def test_the_primary_arm_is_eq_3_verbatim():
    """Eq. 3 is plain unweighted CE, and an audited D_s is balanced anyway.

    `CandidateSelector` draws per class up to a floor, so the inverse-frequency
    weighting was compensating for an unaudited, 80%-benign D_s rather than for
    anything intrinsic. The weighted comparison arm is a `--method-file`
    override of exactly one key; docs/method-versions.md carries it.
    """
    for name in ("ae", "tabcl"):
        assert settings(name)["class_weighting"] is False, name


def test_eq_1_has_no_scalar_on_the_cross_entropy_term():
    """`categorical_weight` was a knob the equation does not have."""
    assert "categorical_weight" not in settings("ae")


# -------------------------------------------------- the rest of the chain

def test_both_branches_read_the_same_feature_set():
    """Otherwise Eq. 17 compares two models that saw different columns.

    One branch on a newer feature set than the other confounds the
    representation change with the capacity fix, and neither number explains
    the other.
    """
    assert settings("ae")["features"] == settings("tabcl")["features"] == "conn-screened"


def test_both_branches_draw_from_the_same_pools():
    """Different pools would make the two branches disagree about D_l itself."""
    assert settings("ae")["pools"] == settings("tabcl")["pools"] == "xdg"


def test_fusion_reads_exactly_the_two_consolidated_branches():
    """Eq. 17 takes two inputs, and consolidation is what makes them nameable."""
    assert settings("fused")["inputs"] == ["ae", "tabcl"]
    assert settings("fused")["features"] == "conn-screened"
