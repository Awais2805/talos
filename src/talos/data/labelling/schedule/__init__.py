"""Path A — deterministic schedule labelling.

Labels never come from the traffic: a capture records what happened, not whether
it was malicious. This method joins an external record of ground truth -- one
transcribed attack schedule per dataset -- onto the flows by time and endpoint,
and calls everything it did not match benign.

That closed-world reading is what makes the approach tractable, and also what
bounds it: it needs a schedule, so it can never label a honeypot capture. The
behavioural methods exist for the traffic this one cannot reach.
"""

from talos.data.labelling.schedule.labeller import ScheduleLabeller
from talos.data.labelling.schedule.manifest import (
    MANIFESTS, AttackRule, Manifest, ManifestError, ManifestLoader,
)
from talos.data.labelling.schedule.quarantine import (
    QuarantineError, QuarantineLoader, QuarantineRegion,
)
from talos.data.labelling.schedule.windows import Window, WindowError

__all__ = [
    "MANIFESTS", "AttackRule", "Manifest", "ManifestError", "ManifestLoader",
    "QuarantineError", "QuarantineLoader", "QuarantineRegion",
    "ScheduleLabeller", "Window", "WindowError",
]
