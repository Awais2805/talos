"""Reconciling several methods, and weighing what comes out.

Fusion reads written tables, so a branch can be re-run without re-running the
others. Confident learning then attenuates rather than discards: an uncertain
row is down-weighted, never deleted, so the table stays the whole dataset.
"""

from talos.data.labelling.fusion.confident import (
    ConfidentJoint, ConfidentLearning, ConfidentLearningError, NoiseReport,
)
from talos.data.labelling.fusion.fuse import FusedLabeller, FusionError, FusionReport

__all__ = [
    "ConfidentJoint", "ConfidentLearning", "ConfidentLearningError",
    "FusedLabeller", "FusionError", "FusionReport", "NoiseReport",
]
