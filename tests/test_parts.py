"""Gate 11 — the model parts, and the rule that keeps them parts."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from talos import parts
from talos.parts.base import BACKENDS, PartError, Part
from talos.parts.mlp import torch_available
from talos.parts.inert import fit_width

PARTS_DIR = Path(parts.__file__).parent

torch_only = pytest.mark.skipif(not torch_available(),
                                reason="the [torch] extra is not installed")


# ------------------------------------------------------------------ the rule

#: Vocabulary that would mean a part had learned what it is being used for.
FORBIDDEN = ("flow", "attack", "benign", "conn", "dataset", "capture", "pcap",
             "victim", "malicious", "intrusion", "zeek", "suricata", "label")


def code_of(path: Path) -> list[str]:
    """Every identifier and non-docstring literal in a module.

    Docstrings and comments are EXEMPT, and deliberately so: the rule polices
    what the code knows, not what its authors are allowed to explain. The
    modules stating the boundary have to be able to name the thing they are
    refusing to depend on.
    """
    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.append(node.id)
        elif isinstance(node, ast.Attribute):
            found.append(node.attr)
        elif isinstance(node, ast.arg):
            found.append(node.arg)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(getattr(node, "module", "") or "")
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                found.append(node.value)
    return [f.lower() for f in found]


def test_nn_package_knows_nothing_about_flows():
    """The one architectural rule: `talos/parts/` holds PARTS, not a labeller.

    An encoder that mentions attacks is a labelling component wearing a
    modelling name, and once it exists the detection engine either imports out
    of a data-engineering stage or reimplements it and drifts.
    """
    offences = []
    for path in sorted(PARTS_DIR.rglob("*.py")):
        for fragment in code_of(path):
            for word in FORBIDDEN:
                if word in fragment:
                    offences.append(f"{path.name}: {word!r} in {fragment!r}")
    assert not offences, "; ".join(offences)


def test_the_policing_test_can_actually_fail():
    """A guard that cannot fire is not a guard.

    The exemption for docstrings is broad enough that it is worth proving the
    check still catches the thing it exists to catch.
    """
    offender = Path(__file__).parent / "_nn_offender_probe.py"
    offender.write_text('"""A docstring may say flow."""\n'
                        'def encode(attack_row):\n    return attack_row\n')
    try:
        fragments = code_of(offender)
        assert any("attack" in f for f in fragments)
        assert not any("flow" in f for f in fragments), "docstring must be exempt"
    finally:
        offender.unlink()


def test_nn_imports_without_torch():
    """`import talos.parts` must succeed on a machine that will never train.

    Run in a subprocess with torch made unimportable, because the in-process
    test would pass simply by virtue of torch not being installed here -- and
    would then stop testing anything the moment somebody installs it.
    """
    script = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name == 'torch' or name.startswith('torch.') else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('torch is blocked for this test')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import talos.parts as parts\n"
        "assert 'inert' in parts.ENCODERS.names()\n"
        "assert 'mlp' in parts.ENCODERS.names(), 'mlp must still be addressable'\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_mlp_is_registered_but_refuses_without_torch():
    """A missing dependency must not read as a typo.

    Registering nothing gives `unknown encoder 'mlp'. Have: inert`, which sends
    the operator looking for a misspelling.
    """
    assert "mlp" in parts.ENCODERS.names()
    if torch_available():
        pytest.skip("torch is installed, so construction succeeds")
    with pytest.raises(PartError, match="PyTorch"):
        parts.build("encoder", "mlp", input_dim=4, latent_dim=2)


# -------------------------------------------------------------- the contract

def test_every_registered_part_declares_a_known_backend_and_kind():
    for point in parts.POINTS_BY_KIND.values():
        for name in point.names():
            cls = point.cls_for(name)
            assert issubclass(cls, Part)
            assert cls.backend in BACKENDS, f"{name}: backend {cls.backend!r}"
            assert cls.kind in parts.POINTS_BY_KIND, f"{name}: kind {cls.kind!r}"


def test_build_refuses_an_unknown_kind_by_name():
    with pytest.raises(PartError, match="not a kind of part"):
        parts.build("transmogrifier", "mlp", input_dim=1)


def test_width_mismatch_names_both_numbers():
    encoder = parts.build("encoder", "inert", input_dim=5, latent_dim=2)
    with pytest.raises(PartError, match="takes 5 columns and was given 9"):
        encoder.encode(np.zeros((3, 9)))


# ------------------------------------------------------------- the inert parts

def test_inert_parts_are_shaped_correctly_and_declare_they_cannot_learn():
    encoder = parts.build("encoder", "inert", input_dim=6, latent_dim=3)
    decoder = parts.build("decoder", "inert", latent_dim=3, output_dim=6)
    head = parts.build("head", "inert", input_dim=3, output_dim=4)

    X = np.arange(24, dtype=float).reshape(4, 6)
    Z = encoder.encode(X)
    assert Z.shape == (4, 3)
    assert decoder.decode(Z).shape == (4, 6)
    assert head.project(Z).shape == (4, 4)
    assert not any(p.trainable for p in (encoder, decoder, head))


def test_inert_classifier_is_uniform_not_majority_class():
    """Uniform is the honest answer, and it is not the obvious one.

    A majority-class predictor would score respectably and look like a signal.
    A uniform row arrives downstream as the lowest expressible confidence.
    """
    classifier = parts.build("classifier", "inert", input_dim=3, n_classes=5)
    P = classifier.predict_proba(np.random.default_rng(0).normal(size=(7, 3)))
    assert P.shape == (7, 5)
    assert np.allclose(P, 0.2)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_classifier_refuses_a_single_class():
    with pytest.raises(PartError, match="cannot express a decision"):
        parts.build("classifier", "inert", input_dim=3, n_classes=1)


def test_fit_width_truncates_and_pads():
    A = np.arange(6, dtype=float).reshape(2, 3)
    assert fit_width(A, 2).tolist() == [[0.0, 1.0], [3.0, 4.0]]
    assert fit_width(A, 5).tolist() == [[0, 1, 2, 0, 0], [3, 4, 5, 0, 0]]
    assert fit_width(A, 3) is not None and fit_width(A, 3).shape == (2, 3)


def test_config_reports_what_a_part_was_built_with():
    encoder = parts.build("encoder", "inert", input_dim=6, latent_dim=3)
    config = encoder.config()
    assert config["name"] == "inert" and config["kind"] == "encoder"
    assert config["input_dim"] == 6 and config["output_dim"] == 3
    assert config["trainable"] is False


# ------------------------------------------------------------ the torch parts

@torch_only
def test_mlp_encoder_forward_pass_and_round_trip(tmp_path):
    encoder = parts.build("encoder", "mlp", input_dim=8, latent_dim=4,
                       hidden=[16], seed=7)
    X = np.random.default_rng(1).normal(size=(5, 8))
    Z = encoder.encode(X)
    assert Z.shape == (5, 4) and Z.dtype == np.float64
    assert encoder.trainable

    checkpoint = tmp_path / "enc.pt"
    encoder.save(checkpoint)
    reloaded = parts.build("encoder", "mlp", input_dim=8, latent_dim=4, hidden=[16],
                        seed=99)
    reloaded.load(checkpoint)
    assert np.allclose(encoder.encode(X), reloaded.encode(X))


@torch_only
def test_mlp_is_seeded_so_two_builds_agree():
    X = np.random.default_rng(2).normal(size=(4, 6))
    first = parts.build("encoder", "mlp", input_dim=6, latent_dim=3, seed=11)
    second = parts.build("encoder", "mlp", input_dim=6, latent_dim=3, seed=11)
    assert np.allclose(first.encode(X), second.encode(X))


@torch_only
def test_mlp_classifier_emits_a_distribution():
    classifier = parts.build("classifier", "mlp", input_dim=5, n_classes=4, seed=3)
    P = classifier.predict_proba(np.random.default_rng(4).normal(size=(9, 5)))
    assert P.shape == (9, 4)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P >= 0).all()
