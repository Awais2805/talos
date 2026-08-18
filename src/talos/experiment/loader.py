"""Read, validate and hash an experiment declaration.

Validation is fail-fast and deliberately narrow
"""

from __future__ import annotations

from pathlib import Path

from talos.common.plugins import ConfigPlugins
from talos.experiment.spec import ROLES, ExperimentSpec

DEFAULT_DIR = Path("experiments")
SPEC_FILE = "experiment.yaml"
DEFAULT_LABELS = "schedule"


class ExperimentError(Exception):
    pass


#: The experiment plug-in point: a user declares their own study without it
#: having to live in this repository's `experiments/`.
EXPERIMENTS = ConfigPlugins("experiment", built_in=DEFAULT_DIR,
                            filename=SPEC_FILE, error=ExperimentError)


class ExperimentLoader:
    """Finds and loads experiment declarations."""

    def __init__(self, directory: str | Path | None = None):
        """Reads the shared `EXPERIMENTS` point unless scoped to one directory."""
        self.specs = EXPERIMENTS if directory is None else ConfigPlugins(
            "experiment", built_in=directory, filename=SPEC_FILE,
            error=ExperimentError, addressable=False)

    @property
    def directory(self) -> Path | None:
        return self.specs.built_in

    def path_for(self, name: str) -> Path:
        """`experiments/<name>/experiment.yaml`, or the file itself."""
        return self.specs.path_for(name)

    def load(self, name: str) -> ExperimentSpec:
        declaration = self.specs.get(name)
        doc, path = declaration.doc, declaration.path
        datasets = doc.get("datasets") or {}
        self.validate(doc, datasets, path)

        return ExperimentSpec(
            name=doc.get("name", declaration.name),
            lake=doc.get("lake"),
            extractor=doc.get("extractor", "zeek"),
            labels=doc.get("labels", DEFAULT_LABELS),
            datasets=datasets,
            sha=declaration.sha,
            path=path,
            doc=doc,
        )

    @staticmethod
    def validate(doc: dict, datasets: dict, path: Path) -> None:
        errs = []
        if not datasets:
            errs.append("declares no datasets, so it selects nothing from the lake")

        for name, entry in datasets.items():
            role = (entry or {}).get("role")
            if role is None:
                errs.append(f"{name}: no role. Say train, holdout or audit "
                            f"explicitly — an omitted role is not a safe default")
            elif role not in ROLES:
                errs.append(f"{name}: role {role!r} is not one of "
                            f"{', '.join(ROLES)}. A misspelling reads as "
                            f"'nobody said' and silently disables the guard")
            if role == "holdout" and not str((entry or {}).get("reason", "")).strip():
                errs.append(f"{name}: holdout with no reason. Excluding a dataset "
                            f"from training is an argument, and it has to be on record")

        # `labels` names a METHOD DECLARATION, not a family. It is not checked
        # against the registry here: an experiment that imported the labelling
        # package to validate one string would invert the dependency, and an
        # unknown name is refused loudly by whichever stage resolves it. Only the
        # shape is checked, because a name becomes a directory in the lake.
        labels = doc.get("labels", DEFAULT_LABELS)
        if not isinstance(labels, str) or not labels.strip():
            errs.append("labels must name one labelling method declaration, "
                        "e.g. `labels: fused`")
        elif "/" in labels:
            errs.append(f"labels {labels!r} becomes a directory in the labelled "
                        f"zone, so it may not contain a path separator")

        if errs:
            raise ExperimentError(f"{path}:\n  " + "\n  ".join(errs))
