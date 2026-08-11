"""Independent evidence about the labels the schedule produced.

Schedule labelling is self-consistent by construction: the manifest says an
attack ran, so the flows in that window are attacks. Nothing inside that loop
can tell you the window is in the wrong place. The oracle exists to break the
loop with a source that never saw the manifest.
"""

from talos.data.labelling.oracle.corroborate import (
    Corroborator, CorroborationReport,
)
from talos.data.labelling.oracle.offset import OffsetProbe, OffsetVerdict
from talos.data.labelling.oracle.suricata import SuricataError, SuricataOracle

__all__ = [
    "Corroborator", "CorroborationReport",
    "OffsetProbe", "OffsetVerdict",
    "SuricataError", "SuricataOracle",
]
