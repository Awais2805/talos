"""The labelled table: every original Zeek column, plus the core ten and six more.

  core     label_binary · label_class · label_source · label_method ·
           method_sha · label_confidence · label_weight · dataset ·
           capture · labelled_at                        (see labelling.base)
  extras   label_raw · label_executed · rule_id · rules_matched ·
           label_quality · quarantine_reason

`manifest_sha` and `taxonomy_sha` are NOT columns. They are folded into the
composite `method_sha`, which changes when either of them does -- one column
that answers "was this built from the inputs I think it was" instead of one per
input. Both still appear in the run report, where a human reads them.

`label_confidence` and `label_weight` go out as typed NULLs. Schedule labelling
has no calibrated confidence, and writing 1.0 would invent a number that
confident learning then consumes as evidence.

`capture` comes from the parquet path via the conn CTE; the rest of the
provenance is stamped by `ProvenanceService`, because no stage reimplements
lineage locally.
"""

from __future__ import annotations

from talos.common.provenance import ProvMeta, ProvenanceService
from talos.data.labelling.base import (
    NO_CONFIDENCE, NO_WEIGHT, refuse_stale_output, verify_schema,
)
from talos.data.labelling.schedule.closed_world import ClosedWorldResolver
from talos.data.labelling.schedule.execution import ExecutionClassifier
from talos.data.labelling.schedule.join import MATCHED, QUARANTINED

#: What schedule labelling emits beyond the core ten.
EXTRAS = (
    "label_raw", "label_executed", "rule_id", "rules_matched",
    # `label_quality` is NOT here: it moved into core when pools began filtering
    # on it. Out-of-space is a property of the label space, every method has one,
    # so every method can and must answer. `quarantine_reason` stays -- only
    # schedule labelling has regions.
    "quarantine_reason",
)


class LabelWriter:
    """Builds the labelled relation and puts it in the lake."""

    def __init__(self, execution: ExecutionClassifier | None = None,
                 closed_world: ClosedWorldResolver | None = None,
                 provenance: ProvenanceService | None = None,
                 space=None):
        """`space` is the declared label space, or None to flag nothing.

        None is not "an empty space": it means no space was declared, so no class
        is out of it. An empty space would make every flow out-of-space, which is
        the opposite, and is why this is not defaulted to one.
        """
        self.execution = execution or ExecutionClassifier()
        self.closed_world = closed_world or ClosedWorldResolver()
        self.provenance = provenance or ProvenanceService()
        self.space = space

    # ----------------------------------------------------------------- query

    def out_of_space_sql(self) -> str | None:
        """Exclusion reason per class, or None when no space is declared.

        Keyed off the label-class EXPRESSION rather than the output alias, so it
        does not depend on the engine supporting lateral column references.
        """
        if self.space is None:
            return None
        return self.space.out_of_space_sql(self.closed_world.label_class_sql())

    def select_list(self, has_regions: bool, measurable: bool = True) -> str:
        """The label and evidence columns. Provenance is stamped separately."""
        cw = self.closed_world
        return f"""c.*,
             {cw.label_binary_sql()} AS label_binary,
             {cw.label_class_sql()} AS label_class,
             {cw.label_source_sql()} AS label_source,
             {NO_CONFIDENCE} AS label_confidence,
             {NO_WEIGHT} AS label_weight,
             m.rname AS label_raw,
             {self.execution.classify_sql(measurable=measurable)} AS label_executed,
             m.rid AS rule_id,
             {cw.rules_matched_sql()} AS rules_matched,
             {cw.quality_sql(has_regions, self.out_of_space_sql())}"""

    def build_query(self, ctes: str, has_regions: bool,
                    measurable: bool = True) -> str:
        """Every conn flow, labelled.

        A LEFT JOIN, so unmatched flows survive as benign rather than being
        dropped — the output is the whole dataset, not an attack list.
        """
        join = (f"\n      LEFT JOIN quarantined {QUARANTINED} USING (uid)"
                if has_regions else "")
        return (f"{ctes}\n      SELECT {self.select_list(has_regions, measurable)}\n"
                f"      FROM conn c LEFT JOIN matched {MATCHED} USING (uid){join}")

    # ----------------------------------------------------------------- write

    def stamp(self, rel, dataset: str, method: str, method_sha: str,
              labelled_at: str):
        """Attach the content-addressed provenance columns.

        Content-addressed because a hand-maintained version field drifts from
        the file it describes. This is what makes a stale labelled table
        detectable without reading a row.
        """
        return self.provenance.stamp(rel, ProvMeta.of(
            label_method=method, method_sha=method_sha,
            dataset=dataset, labelled_at=labelled_at))

    @staticmethod
    def verify_schema(rel) -> None:
        """The core ten plus schedule's six, present and typed."""
        verify_schema(rel, EXTRAS)

    def write(self, lake, rel, dataset: str, feature_space: str,
              method: str, method_sha: str = "", uri: str | None = None,
              source: str | None = None, force: bool = False) -> str:
        """Write the labelled table, refusing to silently replace a different one.

        Re-running the same inputs overwrites its own output, which is what a
        re-run should do. Running DIFFERENT inputs over it destroys labels
        derived from a schedule that is still on record, with nothing left to say
        they ever existed -- so that case is refused and names both hashes.
        """
        self.verify_schema(rel)
        target = uri or lake.uri("labelled", dataset=dataset, source=source,
                                 feature_space=feature_space, method=method,
                                 rel="conn.parquet")

        refuse_stale_output(lake, target, method_sha, force, method,
                            "the manifest, the taxonomy, the label space or "
                            "the declaration")
        return lake.write_parquet(rel, target)
