#!/usr/bin/env python3
"""Where things are, resolved by looking rather than by counting.

Every module in this package used to locate the repo root with a hardcoded
`Path(__file__).resolve().parents[N]`. That works exactly until a file moves,
and then it fails silently: the path still resolves, just to the wrong
directory. It is not a hypothetical failure -- it is what left
`src/preprocess/label/label_flows.py` unrunnable after it moved one level
deeper, pointing MANIFESTS at a directory that has never existed.

Restructuring this package into stage subdirectories would have reintroduced
exactly that fragility at four different depths. So the depth is no longer
written down anywhere: the root is found by walking up until the marker file
appears, and package-relative paths are resolved from this file's own location.
Moving a module now costs nothing.
"""

from pathlib import Path

# The file that identifies the repo root. Chosen because every stage of the
# pipeline already reads it, so a tree without it is not a tree this code can
# run against anyway.
ROOT_MARKER = "config.yml"

PACKAGE = Path(__file__).resolve().parent.parent      # src/label_validation


def repo_root(start=None):
    """Nearest ancestor holding the root marker.

    Raising rather than guessing: a wrong root produces a confusing
    file-not-found several stages later, and the whole point here is to fail
    where the mistake is.
    """
    here = Path(start).resolve() if start else PACKAGE
    for parent in (here, *here.parents):
        if (parent / ROOT_MARKER).is_file():
            return parent
    raise RuntimeError(
        f"no {ROOT_MARKER} in any parent of {here} — run from inside the repo")


def config(name):
    """A file in this package's config/ directory."""
    return PACKAGE / "config" / name


ROOT = repo_root()
REPORTS = ROOT / "reports" / "validation"
MANIFESTS = ROOT / "src" / "preprocess" / "manifests"
POLICY = config("policy.yaml")
EXPECTATIONS = config("expectations.yaml")


def portable(path):
    """A path as it should appear in a report: relative to the repo root.

    Reports get read on other machines, pasted into dissertations and attached
    to issues. An absolute path from the machine that produced them is useless
    to every one of those readers, and it carries the author's home directory --
    a username -- into an artefact meant to be shareable. Anything outside the
    repo is reduced to its filename for the same reason.
    """
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return p.name
