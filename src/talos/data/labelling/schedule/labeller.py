"""Schedule labelling, as a method among methods.

A thin adapter, deliberately. The engine below it is unchanged and owns the
algorithm; this says what schedule labelling *is* to the rest of the platform --
its name, the columns it adds beyond the core ten, what it needs from an
extractor, and the hash of the inputs that determined its output.

Keeping the adapter thin is the point. If it grew logic, `--method schedule`
and a direct `LabellingEngine` call would start producing different tables.
"""

from __future__ import annotations

from typing import ClassVar

from talos.common.provenance import Sha12, composite_sha
from talos.data.extraction.base import CONN_BACKBONE, FLOW_ID
from talos.data.labelling.base import LABELLERS, LabellingMethod
from talos.data.labelling.schedule.engine import LabellingEngine, LabellingReport
from talos.data.labelling.schedule.manifest import ManifestLoader
from talos.data.labelling.schedule.writer import EXTRAS
from talos.data.labelling.taxonomy import DEFAULT_TAXONOMY, TaxonomyMapper


@LABELLERS.register
class ScheduleLabeller(LabellingMethod):
    """Path A: an attack schedule joined onto flows by time and endpoint."""

    name: ClassVar[str] = "schedule"
    extras: ClassVar[tuple[str, ...]] = EXTRAS

    #: Byte counts are deliberately NOT required. Their absence degrades one
    #: column -- `label_executed` goes NULL -- rather than the whole method, so
    #: it is recorded and the run carries on.
    requires: ClassVar[tuple[str, ...]] = (CONN_BACKBONE, FLOW_ID)

    def __init__(self, cfg, lake=None, spec=None, manifests=None, taxonomy=None,
                 allow_untrained: bool = False, **kwargs):
        """`spec` is the declaration; `manifests`/`taxonomy` scope a run without one.

        Both paths exist because a declaration is the normal route and a scoped
        loader is how a test reads one directory deliberately. When both are
        given the explicit argument wins, so a test cannot be silently overridden
        by whatever the shipped declaration happens to say.
        """
        # `allow_untrained` is accepted and unused: schedule labelling learns
        # nothing, so there is nothing it could fail to vouch for.
        super().__init__(cfg, lake, spec=spec, allow_untrained=allow_untrained)
        self.taxonomy_name = taxonomy or (spec.taxonomy if spec else DEFAULT_TAXONOMY)
        self.engine = LabellingEngine(
            cfg, lake=self.lake, manifests=manifests,
            taxonomy=self.taxonomy_name,
            space=spec.label_space if spec else None,
            method=self.run_name, **kwargs)

    # ------------------------------------------------------------ provenance

    def inputs(self, dataset: str) -> tuple:
        """Every file that decides this dataset's labels.

        The manifest and the taxonomy at minimum. Remapping `DoS-Hulk` from `dos`
        to something else changes every label the schedule produced without
        touching the schedule, and a hash that missed it would call the two runs
        identical. With a declaration, the declaration and its label space join
        them -- excluding a class changes the table too.
        """
        manifest = self.engine.manifests.path_for(dataset)
        if self.spec is not None:
            return (manifest, *self.spec.inputs())
        return (manifest, TaxonomyMapper(self.taxonomy_name).path)

    def method_sha(self, dataset: str) -> Sha12:
        return composite_sha(*self.inputs(dataset))

    # ------------------------------------------------------------------- run

    def label(self, dataset: str, feature_space: str, **kwargs) -> LabellingReport:
        # The engine cannot see the declaration or the label space, so the
        # method hands it the full input set rather than letting it hash a
        # subset and call the answer `method_sha`.
        self.engine.sha_inputs = self.inputs(dataset)
        return self.engine.label(dataset, feature_space, **kwargs)
