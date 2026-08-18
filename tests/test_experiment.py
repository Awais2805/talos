"""The experiment declaration — roles, and the validation that keeps them honest.

The point of this layer is that "may this dataset train" stops being a property
of the lake. What is mostly tested here is the refusals: a role that is missing,
misspelled, or asserted without an argument behind it.
"""

import textwrap

import pytest

from talos.experiment import ExperimentError, ExperimentLoader, ExperimentSpec

REFERENCE = "xdg"                       # the real one, in experiments/


def write(tmp_path, body: str, name: str = "toy") -> ExperimentLoader:
    (tmp_path / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / name / "experiment.yaml").write_text(textwrap.dedent(body))
    return ExperimentLoader(tmp_path)


MINIMAL = """\
    name: toy
    extractor: zeek
    labels: schedule
    datasets:
      a: {role: train}
      b: {role: train}
      c: {role: holdout, reason: "would leak the domain"}
      d: {role: audit}
"""


# --------------------------------------------------------------------- roles

def test_roles_come_from_the_experiment_not_the_lake(tmp_path):
    spec = write(tmp_path, MINIMAL).load("toy")
    assert spec.datasets_with_role("train") == ["a", "b"]
    assert spec.role("c") == "holdout"
    assert spec.role("d") == "audit"


def test_an_unmentioned_dataset_is_unassigned_never_train(tmp_path):
    """Defaulting the other way would let a dataset dropped into the lake join a
    training pool because nobody wrote a line about it."""
    spec = write(tmp_path, MINIMAL).load("toy")
    assert spec.role("something-nobody-declared") == "unassigned"
    assert "something-nobody-declared" not in spec.datasets_with_role("train")


def test_a_missing_role_is_refused_rather_than_defaulted(tmp_path):
    loader = write(tmp_path, """\
        name: toy
        datasets:
          a: {}
    """)
    with pytest.raises(ExperimentError, match="no role"):
        loader.load("toy")


def test_a_misspelled_role_is_refused(tmp_path):
    """The sharp one. `hold-out` would read as 'nobody said' rather than 'this
    must never train', and the guard meant to keep it out never fires."""
    loader = write(tmp_path, """\
        name: toy
        datasets:
          a: {role: hold-out}
    """)
    with pytest.raises(ExperimentError, match="hold-out"):
        loader.load("toy")


def test_a_holdout_must_carry_its_argument(tmp_path):
    """Excluding a dataset from training is a claim, and it has to be on record."""
    loader = write(tmp_path, """\
        name: toy
        datasets:
          a: {role: train}
          b: {role: holdout}
    """)
    with pytest.raises(ExperimentError, match="no reason"):
        loader.load("toy")


def test_an_experiment_that_selects_nothing_is_refused(tmp_path):
    loader = write(tmp_path, "name: toy\ndatasets: {}\n")
    with pytest.raises(ExperimentError, match="no datasets"):
        loader.load("toy")


def test_a_labels_reference_that_could_not_be_a_directory_is_refused(tmp_path):
    loader = write(tmp_path, """\
        name: toy
        labels: a/b
        datasets:
          a: {role: train}
    """)
    with pytest.raises(ExperimentError, match="path separator"):
        loader.load("toy")


# ---------------------------------------------------------------- provenance

def test_the_spec_is_content_hashed(tmp_path):
    """A result names the exact declaration that produced it, arguments included."""
    first = write(tmp_path, MINIMAL).load("toy")
    assert len(first.sha) == 12

    edited = MINIMAL.replace('reason: "would leak the domain"', 'reason: "changed my mind"')
    assert write(tmp_path, edited).load("toy").sha != first.sha


def test_the_reason_travels_with_the_role(tmp_path):
    spec = write(tmp_path, MINIMAL).load("toy")
    assert spec.reason("c") == "would leak the domain"
    assert spec.reason("a") is None


def test_an_unknown_experiment_says_what_is_available(tmp_path):
    loader = write(tmp_path, MINIMAL)
    with pytest.raises(ExperimentError, match="toy"):
        loader.load("nope")


# ----------------------------------------------------- the real declaration

def test_the_reference_experiment_loads_and_holds_the_roles():
    """xdg is where 2017/2018/2019's roles went when they left config.yml."""
    spec = ExperimentLoader().load(REFERENCE)
    assert spec.datasets_with_role("train") == ["cic-ids-2017", "cic-ids-2018"]
    assert spec.role("cic-ddos-2019") == "holdout"
    assert "domain leakage" in spec.reason("cic-ddos-2019")
    assert spec.labels == "schedule"
    assert isinstance(spec, ExperimentSpec)


def test_talos_never_imports_from_experiments():
    """The platform/experiment split, enforced rather than hoped.

    `talos/` holds the mechanisms that make two results comparable; the
    declarations of what is being compared live outside it. Lint catches the
    leakage that discipline will not.
    """
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    hits = subprocess.run(
        ["grep", "-rnE", r"^\s*(from|import)\s+experiments", str(repo / "src" / "talos")],
        capture_output=True, text=True).stdout.strip()
    assert hits == "", f"talos/ imports from experiments/:\n{hits}"
