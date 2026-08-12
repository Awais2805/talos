"""Deciding what a flow is.

There is more than one way, and they are not interchangeable:

    schedule/      Path A -- an external attack schedule joined by time and
                   endpoint. Exact where it applies, and it cannot apply at all
                   without a published schedule.
    behavioural/   Path B -- learned from the traffic itself, so it reaches the
                   captures Path A cannot: a honeypot has no schedule.
    fusion/        several methods reconciled, then confident learning over the
                   result.
    oracle/        not a method. Independent evidence ABOUT labels, from a source
                   that never saw the manifest.

`base` is what they have in common: the plug-in contract, and the ten core
columns every method's table carries so that fusion, the benchmark, EDA and
training never have to ask which one wrote it.
"""

from talos.data.labelling.base import (
    CORE_COLUMNS, CORE_TYPES, LABEL_SOURCES, LABELLERS, LabellingError,
    LabellingMethod, SchemaError, verify_schema,
)
from talos.data.labelling.taxonomy import TAXONOMIES, TaxonomyError, TaxonomyMapper

__all__ = [
    "CORE_COLUMNS", "CORE_TYPES", "LABEL_SOURCES", "LABELLERS",
    "LabellingError", "LabellingMethod", "SchemaError", "verify_schema",
    "TAXONOMIES", "TaxonomyError", "TaxonomyMapper",
]
