"""Asking a person, and scoring the machine against their answer.

The audited slice is the only ground truth in the project — everything else is
one method's opinion. Rebuild rule 3 is what this exists for: generated labels
must not be audited by generated expectations.
"""

from talos.data.labelling.audit.adjudication import (
    AdjudicationTable, Adjudication, CONFIRMED, CORRECTED, UNADJUDICATED, UNKNOWN,
)
from talos.data.labelling.audit.render import AuditPage
from talos.data.labelling.audit.benchmark import (
    BenchmarkReport, ClassScore, LabelBenchmark, MethodScore,
)
from talos.data.labelling.audit.candidates import (
    AuditError, CandidateSelector, Selection, Shortfall,
)

__all__ = [
    "AdjudicationTable", "Adjudication", "AuditError", "AuditPage", "BenchmarkReport",
    "CandidateSelector", "ClassScore", "CONFIRMED", "CORRECTED", "LabelBenchmark",
    "MethodScore", "Selection", "Shortfall", "UNADJUDICATED", "UNKNOWN",
]
