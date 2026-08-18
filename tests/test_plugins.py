"""Bring your own: the one mechanism behind every replaceable part of Talos.

Two kinds, by what the user hands us -- a `.py` with a class (`CodePlugins`) or a
`.yaml` (`ConfigPlugins`). The mechanical properties are tested here, and so is
the one that matters: a user can substitute their own extractor, manifest or
method WITHOUT editing the installed package, and a run that used theirs does not
look identical to a run that used ours.

Tests construct their points with `addressable=False`. Only the module-level
point a stage owns may claim a kind, because the kind is how config.yml
addresses it.
"""

import textwrap

import pytest

from talos.common.plugins import (
    BUILT_IN, EXTERNAL, SEARCHED, CodePlugins, ConfigPlugins, Declaration,
    POINTS, PluginError, install,
)


def code(kind="widget", error=PluginError):
    return CodePlugins(kind, error=error, addressable=False)


def config(kind="widget", **kwargs):
    return ConfigPlugins(kind, addressable=False, **kwargs)


class Widget:
    name = "widget"


class Gadget:
    name = "gadget"


class Fussy:
    """Needs a constructor argument. The old registry called cls() at import time
    to read the name, so a plugin shaped like this broke the whole package."""

    name = "fussy"

    def __init__(self, required):
        self.required = required


class OwnError(Exception):
    """Stands in for ExtractionError / ManifestError / ExperimentError."""


# ------------------------------------------------------------- code plug-ins

def test_the_name_is_read_off_the_class_without_constructing_anything():
    point = code()
    point.register(Fussy)
    assert "fussy" in point
    assert point.cls_for("fussy") is Fussy          # no instance was ever made
    assert point.get("fussy", required="2.0").required == "2.0"


def test_a_class_without_a_name_is_refused():
    with pytest.raises(PluginError, match="class-level"):
        code().register(type("Nameless", (), {}))


def test_two_classes_claiming_one_name_are_refused():
    point = code()
    point.register(Widget)
    with pytest.raises(PluginError, match="both call themselves"):
        point.register(type("Impostor", (), {"name": "widget"}))


def test_registering_the_same_class_twice_is_not_a_conflict():
    """Module re-import is normal; only a genuine collision is an error."""
    point = code()
    point.register(Widget)
    point.register(Widget)
    assert point.names() == ("widget",)


def test_an_unknown_name_lists_what_exists():
    point = code()
    point.register(Widget)
    point.register(Gadget)
    with pytest.raises(PluginError, match="Have: gadget, widget"):
        point.get("sprocket")


def test_an_empty_point_says_none_rather_than_listing_nothing():
    with pytest.raises(PluginError, match="Have: none"):
        code().get("anything")


def test_a_point_raises_the_stage_s_own_exception():
    """Porting a stage onto this must not change what its callers catch."""
    with pytest.raises(OwnError):
        code("extractor", error=OwnError).get("nope")


def test_a_registered_class_reports_itself_as_built_in():
    point = code()
    point.register(Widget)
    assert point.origin_of("widget").built_in


# ----------------------------------------------- bringing your own code, by file

PLUGIN = """\
import talos.common.plugins as plugins

@plugins.UNDER_TEST.register
class Mine:
    name = "mine"
"""


@pytest.fixture
def hosted(monkeypatch):
    """A point an external file can reach by import, as a real one would.

    In production a plugin does `from talos.data.extraction import
    register_extractor`; the module attribute stands in for that, so `add` is
    exercised without permanently mutating a real point.
    """
    import talos.common.plugins as plugins

    point = code("widget", error=OwnError)
    monkeypatch.setattr(plugins, "UNDER_TEST", point, raising=False)
    return point


def test_an_external_file_adds_a_plugin_without_touching_the_package(hosted, tmp_path):
    """The capability this gate exists for."""
    path = tmp_path / "mine.py"
    path.write_text(PLUGIN)

    assert hosted.add(path) == ("mine",)
    assert hosted.get("mine").name == "mine"


def test_an_externally_added_plugin_is_marked_external_with_its_file(hosted, tmp_path):
    """A run that used the user's plugin must not look like one that used ours."""
    path = tmp_path / "mine.py"
    path.write_text(PLUGIN)
    hosted.add(path)

    origin = hosted.origin_of("mine")
    assert origin.where == EXTERNAL and origin.source == path.resolve()
    assert "external" in origin.describe()


def test_adding_a_directory_imports_every_plugin_in_it(hosted, tmp_path):
    (tmp_path / "mine.py").write_text(PLUGIN)
    (tmp_path / "other.py").write_text(PLUGIN.replace("Mine", "Other")
                                       .replace('"mine"', '"other"'))
    (tmp_path / "_ignored.py").write_text("raise AssertionError('underscore skipped')")
    assert hosted.add(tmp_path) == ("mine", "other")


def test_adding_the_same_file_twice_is_a_no_op(hosted, tmp_path):
    """Re-import would make a second class object of the same name and trip the
    collision refusal on the user's own plugin."""
    path = tmp_path / "mine.py"
    path.write_text(PLUGIN)
    assert hosted.add(path) == ("mine",)
    assert hosted.add(path) == ()


def test_a_file_that_registers_nothing_is_refused(hosted, tmp_path):
    """Importing something inert and carrying on is how a run silently uses the
    shipped plugin while its operator believes it used theirs."""
    path = tmp_path / "inert.py"
    path.write_text("VALUE = 1\n")
    with pytest.raises(OwnError, match="registered no widget"):
        hosted.add(path)


def test_a_file_that_raises_reports_why_and_leaves_no_broken_module(hosted, tmp_path):
    import sys

    before = set(sys.modules)
    path = tmp_path / "broken.py"
    path.write_text("raise ValueError('boom')\n")
    with pytest.raises(OwnError, match="ValueError: boom"):
        hosted.add(path)
    assert set(sys.modules) == before


def test_a_missing_plugin_file_is_named(hosted, tmp_path):
    with pytest.raises(OwnError, match="nowhere.py"):
        hosted.add(tmp_path / "nowhere.py")


# ----------------------------------------------------------- config plug-ins

FLAT = "dataset: toy\nvalue: 1\n"


def flat_dir(tmp_path, *names, where="flat"):
    directory = tmp_path / where
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / f"{name}.yaml").write_text(FLAT)
    return directory


def nested_dir(tmp_path, *names, filename="experiment.yaml"):
    directory = tmp_path / "nested"
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / name).mkdir()
        (directory / name / filename).write_text(FLAT)
    return directory


def test_the_flat_layout_resolves_name_dot_yaml(tmp_path):
    """Manifests are `<dir>/<name>.yaml`."""
    point = config("manifest", built_in=flat_dir(tmp_path, "toy"))
    found = point.get("toy")
    assert found.name == "toy" and found.doc["value"] == 1
    assert found.built_in and found.origin.where == BUILT_IN


def test_the_nested_layout_resolves_name_slash_filename(tmp_path):
    """Experiments are `<dir>/<name>/experiment.yaml`."""
    point = config("experiment", built_in=nested_dir(tmp_path, "xdg"),
                   filename="experiment.yaml")
    assert point.get("xdg").name == "xdg"


def test_an_unknown_name_lists_what_is_available(tmp_path):
    point = config("manifest", built_in=flat_dir(tmp_path, "a", "b"))
    with pytest.raises(PluginError, match="Have: a, b"):
        point.get("c")


def test_an_added_directory_wins_and_says_so(tmp_path):
    """An override is meant to win. `origin` is what stops it being invisible."""
    point = config("manifest", built_in=flat_dir(tmp_path, "toy"))
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "toy.yaml").write_text("dataset: toy\nvalue: 99\n")
    point.add(mine)

    found = point.get("toy")
    assert found.doc["value"] == 99
    assert found.origin.where == SEARCHED and not found.built_in
    assert str(mine) in found.describe()             # a report cannot omit it


def test_a_direct_path_resolves_and_is_marked_external(tmp_path):
    """`--manifest ~/mine.yaml`, the try-it-once case."""
    path = tmp_path / "elsewhere.yaml"
    path.write_text(FLAT)
    point = config("method", built_in=flat_dir(tmp_path, "toy"))
    found = point.get(path)
    assert found.origin.where == EXTERNAL and found.path == path


def test_a_search_path_must_be_a_directory_not_a_file(tmp_path):
    point = config("manifest", built_in=flat_dir(tmp_path, "toy"))
    with pytest.raises(PluginError, match="not a directory"):
        point.add(tmp_path / "flat" / "toy.yaml")


def test_names_deduplicate_across_directories(tmp_path):
    point = config("manifest", built_in=flat_dir(tmp_path, "a", "b"))
    mine = flat_dir(tmp_path, "a", "c", where="mine")
    point.add(mine)
    assert point.names() == ("a", "c", "b")          # nearest directory first


def test_the_sha_is_the_file_s_content_and_moves_with_it(tmp_path):
    directory = flat_dir(tmp_path, "toy")
    point = config("manifest", built_in=directory)
    before = point.get("toy").sha
    (directory / "toy.yaml").write_text(FLAT + "extra: 2\n")
    assert point.get("toy").sha != before
    assert len(before) == 12


def test_a_document_that_is_not_a_mapping_is_refused(tmp_path):
    """A YAML list where a block of named fields belongs. Reading `.get` off it
    would raise somewhere far from the file that caused it."""
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / "toy.yaml").write_text("- one\n- two\n")
    with pytest.raises(PluginError, match="not a mapping"):
        config("manifest", built_in=directory).get("toy")


def test_invalid_yaml_names_the_file(tmp_path):
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / "toy.yaml").write_text("a: [1,\n")
    with pytest.raises(PluginError, match="toy.yaml"):
        config("manifest", built_in=directory).get("toy")


# ------------------------------------------------------ addressing from config

def test_a_point_claims_its_kind_so_config_yml_can_address_it():
    from talos.data.extraction.base import EXTRACTORS
    from talos.data.labelling.schedule.manifest import MANIFESTS
    from talos.experiment.loader import EXPERIMENTS

    assert POINTS["extractor"] is EXTRACTORS
    assert POINTS["manifest"] is MANIFESTS
    assert POINTS["experiment"] is EXPERIMENTS


def test_two_points_cannot_claim_one_kind():
    """The kind is how config.yml addresses a point, so it has to be unique."""
    with pytest.raises(PluginError, match="both call themselves"):
        CodePlugins("extractor")


def test_config_installs_a_users_manifest_directory(tmp_path, monkeypatch):
    from talos.data.labelling.schedule.manifest import MANIFESTS

    monkeypatch.setattr(MANIFESTS, "_added", [])
    flat_dir(tmp_path, "acme-corp-2026")
    lines = install({"manifest": [str(tmp_path / "flat")]})

    assert "acme-corp-2026" in MANIFESTS.names()
    assert "acme-corp-2026" in lines[0]


def test_config_refuses_a_kind_that_is_not_a_plug_in_point():
    """A typo would otherwise be config that looks applied and does nothing."""
    with pytest.raises(PluginError, match="not a plug-in point"):
        install({"extracter": ["~/typo"]})


def test_a_bad_plugins_block_fails_config_load_with_the_config_named(tmp_path):
    from talos.common.config import Config, ConfigError

    path = tmp_path / "config.yml"
    path.write_text("lake:\n  root: ./lake\nplugins:\n  extracter: [~/typo]\n")
    with pytest.raises(ConfigError, match="not a plug-in point"):
        Config.load(path)


# ------------------------------------------------------ the stages are ported

def test_the_three_existing_stages_all_use_this_mechanism():
    """Asserted directly, so a later change cannot quietly unport one of them."""
    from talos.data.extraction.base import EXTRACTORS, ExtractionError
    from talos.data.labelling.schedule.manifest import ManifestError, ManifestLoader
    from talos.experiment.loader import ExperimentError, ExperimentLoader

    assert isinstance(EXTRACTORS, CodePlugins) and EXTRACTORS.error is ExtractionError

    manifests = ManifestLoader().specs
    assert isinstance(manifests, ConfigPlugins) and manifests.error is ManifestError

    experiments = ExperimentLoader().specs
    assert isinstance(experiments, ConfigPlugins) and experiments.error is ExperimentError


def test_the_shipped_manifests_still_resolve_and_report_as_built_in():
    from talos.data.labelling.schedule.manifest import ManifestLoader

    found = ManifestLoader().specs.get("cic-ids-2017")
    assert isinstance(found, Declaration)
    assert found.built_in and found.doc["dataset"] == "cic-ids-2017"


def test_a_user_can_bring_their_own_schedule_for_a_shipped_dataset(tmp_path, monkeypatch):
    """Override a CIC schedule without editing the package, and have the run be
    able to say that is what happened."""
    from talos.data.labelling.schedule.manifest import MANIFESTS, ManifestLoader

    monkeypatch.setattr(MANIFESTS, "_added", [])
    (tmp_path / "cic-ids-2017.yaml").write_text(textwrap.dedent("""
        dataset: cic-ids-2017
        utc_offset: "-03:00"
        attacks:
          - {name: PortScan, date: 2017-07-07, start: "13:55", end: "14:36",
             attacker_ips: [10.0.0.1], victim_ips: [10.0.0.2]}
    """))
    MANIFESTS.add(tmp_path)

    manifest = ManifestLoader().load("cic-ids-2017")
    assert len(manifest.rules) == 1                  # theirs, not the shipped 18
    assert MANIFESTS.origin_of("cic-ids-2017").where == SEARCHED


def test_a_loader_scoped_to_a_directory_ignores_the_users_search_paths(tmp_path, monkeypatch):
    """`ManifestLoader(dir)` reads that directory deliberately, and must not
    silently pick up an override the user installed globally."""
    from talos.data.labelling.schedule.manifest import MANIFESTS

    monkeypatch.setattr(MANIFESTS, "_added", [])
    MANIFESTS.add(flat_dir(tmp_path, "cic-ids-2017", where="theirs"))

    scoped = ConfigPlugins("manifest", built_in=tmp_path / "empty", addressable=False)
    assert scoped.names() == ()
