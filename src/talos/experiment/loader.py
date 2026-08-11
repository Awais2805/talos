"""Read, validate and hash an experiment declaration.

Validation is fail-fast and deliberately narrow
"""

from __future__ import annotations

from pathlib import Path

import yaml

from talos.common.provenance import content_sha
from talos.experiment.spec import ROLES, ExperimentSpec

DEFAULT_DIR = Path("experiments")
SPEC_FILE = "experiment.yaml"
STRATEGIES = ("schedule", "behavioural")


class ExperimentError(Exception):
    pass


class ExperimentLoader:
    """Finds and loads experiment declarations."""

    def __init__(self, directory: str | Path = DEFAULT_DIR):
        self.directory = Path(directory)

    def path_for(self, name: str) -> Path:
        """`experiments/<name>/experiment.yaml`, or the file itself."""
        candidate = Path(name)
        if candidate.suffix in (".yaml", ".yml") and candidate.exists():
            return candidate
        path = self.directory / name / SPEC_FILE
        if not path.exists():
            available = ", ".join(sorted(
                p.parent.name for p in self.directory.glob(f"*/{SPEC_FILE}"))) or "none"
            raise ExperimentError(
                f"no experiment {name!r} under {self.directory}. Have: {available}")
        return path

    def load(self, name: str) -> ExperimentSpec:
        path = self.path_for(name)
        doc = yaml.safe_load(path.read_text()) or {}
        datasets = doc.get("datasets") or {}
        self.validate(doc, datasets, path)

        return ExperimentSpec(
            name=doc.get("name", path.parent.name),
            lake=doc.get("lake"),
            extractor=doc.get("extractor", "zeek"),
            labelling_strategy=doc.get("labelling_strategy", "schedule"),
            datasets=datasets,
            sha=content_sha(path),
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

        strategy = doc.get("labelling_strategy", "schedule")
        if strategy not in STRATEGIES:
            errs.append(f"labelling_strategy {strategy!r} is not one of "
                        f"{', '.join(STRATEGIES)}")

        if errs:
            raise ExperimentError(f"{path}:\n  " + "\n  ".join(errs))
