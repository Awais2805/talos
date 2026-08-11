"""The experiment declaration — what is being compared, as opposed to what compares it.

`Role` is the clearest case. "May this dataset contribute to a training pool" is
not a property of the data — CIC-DDoS-2019 is a holdout *for this study*, and
another study could legitimately train on it. So roles live here, pinned inside
`experiment_sha` next to the result they produced, rather than in `config.yml`
where they would silently apply to every experiment that ever ran against the
lake.

This module is platform: the loader, the shape, the hashing. The declarations it
reads live in `experiments/` and are never imported by `talos/`.
"""

from talos.experiment.loader import ExperimentError, ExperimentLoader
from talos.experiment.spec import ROLES, ExperimentSpec, Role

__all__ = ["ExperimentError", "ExperimentLoader", "ExperimentSpec", "Role", "ROLES"]
