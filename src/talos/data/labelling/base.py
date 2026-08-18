"""What a labelling method is, and what every one of them must produce.

Schedule labelling is not *the* labeller, it is *a* labeller. An autoencoder
branch, a contrastive branch and a fused result are three more, and a honeypot
capture can only ever be reached by one of the latter. They disagree by design:
the three-way benchmark exists to measure exactly that.

Which is only possible if their outputs are comparable. So the contract here is
mostly about the TABLE, not about the algorithm:

    CORE     ten columns, always emitted, types pinned. Fusion, confident
             learning, EDA, the benchmark and training read these and never ask
             which method wrote them.
    EXTRAS   whatever else a method naturally produces, declared by it.

Core columns are emitted even when the method has no value for them. Schedule
labelling writes `label_confidence` as a typed NULL -- not 1.0, which would be a
number invented to fill a column and then consumed as evidence by the very stage
that estimates label noise. NULL means "not measurable" and is never collapsed
to a value; that rule already cost one bug, when `label_executed` went out as an
untyped NULL and took a different type per extractor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from talos.common.plugins import CodePlugins
from talos.common.provenance import Sha12
from talos.data.extraction.base import CONN_BACKBONE, FLOW_ID


class LabellingError(Exception):
    pass


#: The labelling-method plug-in point. A user brings their own labeller as a
#: `.py` and it is addressable as `talos label --method theirs`.
LABELLERS = CodePlugins("labelling method", error=LabellingError)


# ------------------------------------------------------------------- the table

#: Emitted by every method, whatever it is. Order is the written column order.
CORE_COLUMNS = (
    "label_binary",       # attack or not
    "label_class",        # canonical class, benign included
    "label_source",       # HOW this row got its label -- see LABEL_SOURCES
    "label_method",       # WHICH method -- 'schedule', 'ae', 'fused'
    "method_sha",         # composite hash of everything that determined it
    "label_confidence",   # NULL when the method cannot measure one
    "label_weight",       # NULL until confident learning assigns one
    "label_quality",      # certain | uncertain | out-of-space
    "dataset",
    "capture",            # leakage-safe split key
    "labelled_at",
)

#: Presence is not enough. The same column arriving as two different types
#: depending on which method or extractor produced it makes `WHERE
#: label_executed` behave differently per dataset -- which it once did.
CORE_TYPES = {
    "label_binary": "BOOLEAN",
    "label_class": "VARCHAR",
    "label_source": "VARCHAR",
    "label_method": "VARCHAR",
    "method_sha": "VARCHAR",
    "label_confidence": "DOUBLE",
    "label_weight": "DOUBLE",
    "label_quality": "VARCHAR",
    "dataset": "VARCHAR",
    "capture": "VARCHAR",
    "labelled_at": "VARCHAR",
}

# How a row got its label. A closed vocabulary because the benchmark groups on
# it: `matched` and `predicted` are claims of very different strength, and
# `closed-world` is not a claim about the flow at all -- it is the absence of one.
MATCHED = "matched"              # a rule or an oracle positively identified it
CLOSED_WORLD = "closed-world"    # benign because nothing matched it
PREDICTED = "predicted"          # a model assigned it
FUSED = "fused"                  # several methods were reconciled into it

LABEL_SOURCES = (MATCHED, CLOSED_WORLD, PREDICTED, FUSED)

#: Typed NULLs for the core columns a method cannot fill. Spelled out rather
#: than written as bare NULL so DuckDB is never left to infer the type.
NO_CONFIDENCE = "CAST(NULL AS DOUBLE)"
NO_WEIGHT = "CAST(NULL AS DOUBLE)"


class SchemaError(LabellingError):
    pass


class StaleOutputError(LabellingError):
    """Refusing to overwrite a table built from different inputs.

    A LabellingError, not a ValueError: `talos label` catches one type and
    prints one line. As a ValueError it escaped the handler and reached the
    user as a traceback, which is how a deliberate, correct refusal ends up
    looking like a crash.
    """


def verify_schema(rel, extras: tuple[str, ...] = ()) -> None:
    """Every core column present and correctly typed, plus a method's own extras.

    Run before every write. A column silently vanishing from a select list is
    the kind of change that only surfaces three stages later.
    """
    columns = list(rel.columns)
    missing = [c for c in CORE_COLUMNS if c not in columns]
    if missing:
        raise SchemaError(
            f"labelled relation is missing core column(s): {', '.join(missing)}. "
            f"Every method emits all of {len(CORE_COLUMNS)}, NULL where it has "
            f"no value -- a reader must not have to know which method wrote it.")

    absent_extras = [c for c in extras if c not in columns]
    if absent_extras:
        raise SchemaError(
            f"method declares extra column(s) it did not emit: "
            f"{', '.join(absent_extras)}")

    actual = dict(zip(columns, (str(t) for t in rel.types)))
    wrong = [f"{c} is {actual[c]}, expected {want}"
             for c, want in CORE_TYPES.items() if actual.get(c) != want]
    if wrong:
        raise SchemaError("labelled relation has the wrong column types: "
                          + "; ".join(wrong))


# ------------------------------------------------------------------ the method

class LabellingMethod(ABC):
    """One way of deciding what a flow is.

    `name` is a CLASS attribute, read by the registry at import time without
    constructing anything -- the same contract `BaseExtractor` uses, for the same
    reason.
    """

    #: How `talos label --method X` and the labelled zone path address this.
    name: ClassVar[str] = ""

    #: Columns this method emits beyond the core ten.
    extras: ClassVar[tuple[str, ...]] = ()

    #: What the extractor must supply for this method to run at all. Both
    #: defaults are structural: without one record per connection and a stable
    #: per-flow id there is nothing to attach a label to.
    requires: ClassVar[tuple[str, ...]] = (CONN_BACKBONE, FLOW_ID)

    def __init__(self, cfg, lake=None, spec=None, allow_untrained: bool = False,
                 skip_audit_gate: bool = False):
        self.cfg = cfg
        self.lake = lake if lake is not None else cfg.lake()
        self.spec = spec
        #: May this method write a table it cannot vouch for? Part of the
        #: contract, so one CLI flag means the same thing whichever method ran.
        self.allow_untrained = allow_untrained
        #: May a behavioural method fine-tune on D_s's raw (unaudited) label?
        #: Same contract shape as `allow_untrained` -- one flag, every method,
        #: methods that do not fine-tune (schedule, fused) accept and ignore it.
        self.skip_audit_gate = skip_audit_gate

    @property
    def run_name(self) -> str:
        """What scopes the output path and lands in `label_method`.

        The DECLARATION's name, not the implementation's. One implementation
        serves many declarations -- `ae` and any `--method-file` variant of it
        are one class with different settings and must produce separate tables;
        if the implementation name won, the second would overwrite the first and
        `label_method` would not say which had run.
        """
        return self.spec.name if self.spec is not None else self.name

    # ------------------------------------------------------------- contract

    @abstractmethod
    def method_sha(self, dataset: str) -> Sha12:
        """Composite hash of everything that determined these labels.

        Per dataset, because a method's inputs can be -- schedule labelling uses
        a different manifest for each. Stamped on every row, so a stale labelled
        table is detectable without reading one.
        """

    @abstractmethod
    def label(self, dataset: str, feature_space: str, **kwargs):
        """Label one dataset. Returns the method's own report."""

    # --------------------------------------------------------------- shared

    def check_extractor(self, extractor) -> None:
        """Refuse early when the extractor cannot support this method at all."""
        if extractor is not None:
            extractor.require(*self.requires)

    def output_uri(self, dataset: str, feature_space: str,
                   source: str | None = None, rel: str = "conn.parquet") -> str:
        """Where this method's labels go. Scoped by `{method}`, so no two collide."""
        return self.lake.uri("labelled", dataset=dataset, source=source,
                             feature_space=feature_space, method=self.run_name,
                             rel=rel)

    def describe(self) -> str:
        origin = LABELLERS.origin_of(self.name)
        return origin.describe()
