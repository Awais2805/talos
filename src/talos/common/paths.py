"""Output locations, resolved against the working directory rather than __file__.

Once Talos is an installed package, __file__ points into site-packages, so anything
derived from it -- `Path(__file__).parents[2] / "reports"` -- silently resolves
outside the project. Reports and config are therefore resolved from the current
working directory, overridable by environment variable.

Step 2 of the refactor replaces these environment lookups with config.yml; the
function boundary is here so that change lands in one place.
"""

import os
from pathlib import Path


def reports_dir() -> Path:
    """Root for generated reports. `TALOS_REPORTS` overrides."""
    return Path(os.environ.get("TALOS_REPORTS", "reports"))


def eda_dir() -> Path:
    """Where profile_*.json, compare_*.json and the rendered HTML live."""
    return reports_dir() / "eda"


def models_dir() -> Path:
    """Root for model checkpoints. `TALOS_MODELS` overrides."""
    return Path(os.environ.get("TALOS_MODELS", "models"))


def default_config() -> Path:
    """Config search: `TALOS_CONFIG`, else ./config.yml."""
    return Path(os.environ.get("TALOS_CONFIG", "config.yml"))
