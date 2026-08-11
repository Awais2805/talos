"""The labelled table: every original Zeek column plus fourteen.
The fourteen, in four groups:

  label       label_binary · label_class · label_raw · label_executed
  evidence    label_source · rule_id · rules_matched
  quality     label_quality · quarantine_reason
  provenance  capture · dataset · manifest_sha · taxonomy_sha · labelled_at

`capture` comes from the parquet path via the conn CTE; the other four
provenance columns are stamped by `ProvenanceService`, not written here, because
no stage may reimplement lineage locally.
"""

from __future__ import annotations

from talos.common.provenance import ProvMeta, ProvenanceService
from talos.data.labelling.closed_world import ClosedWorldResolver
from talos.data.labelling.execution import ExecutionClassifier
from talos.data.labelling.join import MATCHED, QUARANTINED

# Asserted after every run. A column silently vanishing from the select list is
# the kind of change that only surfaces three stages later.
LABEL_COLUMNS = (
    "label_binary", "label_class", "label_raw", "label_executed",
    "label_source", "rule_id", "rules_matched",
    "label_quality", "quarantine_reason",
    "capture", "dataset", "manifest_sha", "taxonomy_sha", "labelled_at",
)

# The columns whose TYPE is load-bearing, and what it must be. Presence alone is
# not enough: `label_executed` was briefly emitted as an untyped NULL, which
# DuckDB inferred as something other than BOOLEAN, so the same column had two
# types depending on which extractor produced the flows. A reader doing
# `WHERE label_executed` would then behave differently per dataset.
LABEL_TYPES = {
    "label_binary": "BOOLEAN",
    "label_executed": "BOOLEAN",
    "label_class": "VARCHAR",
    "label_source": "VARCHAR",
    "label_quality": "VARCHAR",
}


class LabelWriter:
    """Builds the labelled relation and puts it in the lake."""

    def __init__(self, execution: ExecutionClassifier | None = None,
                 closed_world: ClosedWorldResolver | None = None,
                 provenance: ProvenanceService | None = None):
        self.execution = execution or ExecutionClassifier()
        self.closed_world = closed_world or ClosedWorldResolver()
        self.provenance = provenance or ProvenanceService()

    # ----------------------------------------------------------------- query

    def select_list(self, has_regions: bool, measurable: bool = True) -> str:
        """The label and evidence columns. Provenance is stamped separately."""
        cw = self.closed_world
        return f"""c.*,
             {cw.label_binary_sql()} AS label_binary,
             {cw.label_class_sql()} AS label_class,
             m.rname AS label_raw,
             {self.execution.classify_sql(measurable=measurable)} AS label_executed,
             {cw.label_source_sql()} AS label_source,
             m.rid AS rule_id,
             {cw.rules_matched_sql()} AS rules_matched,
             {cw.quality_sql(has_regions)}"""

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

    def stamp(self, rel, dataset: str, manifest_sha: str, taxonomy_sha: str,
              labelled_at: str):
        """Attach the content-addressed provenance columns.

        Content-addressed because a hand-maintained version field drifts from
        the file it describes. This is what makes a stale labelled table
        detectable without reading a row.
        """
        return self.provenance.stamp(rel, ProvMeta.of(
            dataset=dataset, manifest_sha=manifest_sha,
            taxonomy_sha=taxonomy_sha, labelled_at=labelled_at))

    @staticmethod
    def verify_schema(rel) -> None:
        """Every one of the fourteen is present, and the load-bearing ones typed."""
        missing = [c for c in LABEL_COLUMNS if c not in rel.columns]
        if missing:
            raise ValueError(f"labelled relation is missing {', '.join(missing)}")

        actual = dict(zip(rel.columns, (str(t) for t in rel.types)))
        wrong = [f"{c} is {actual[c]}, expected {want}"
                 for c, want in LABEL_TYPES.items() if actual.get(c) != want]
        if wrong:
            raise ValueError("labelled relation has the wrong column types: "
                             + "; ".join(wrong))

    def existing_manifest_sha(self, lake, uri: str) -> str | None:
        """The manifest hash of whatever is already at `uri`, if anything is.

        One row, one column. Cheap enough to run before every write, and it is
        the point of content-addressing: the zone can say which schedule it was
        built from without anybody having recorded it separately.
        """
        if not lake.exists(uri):
            return None
        try:
            row = lake.read_parquet(uri, columns=["manifest_sha"]).limit(1).fetchone()
        except Exception:                                # noqa: BLE001
            return None                                  # unreadable: treat as absent
        return row[0] if row else None

    def write(self, lake, rel, dataset: str, feature_space: str,
              uri: str | None = None, source: str | None = None,
              manifest_sha: str = "", force: bool = False) -> str:
        """Write the labelled table, refusing to silently replace a different one.

        Re-running the same manifest overwrites its own output, which is what a
        re-run should do. Running a DIFFERENT manifest over it destroys labels
        derived from a schedule that is still on record, with nothing left to
        say they ever existed -- so that case is refused and names both hashes.
        """
        self.verify_schema(rel)
        target = uri or lake.uri("labelled", dataset=dataset, source=source,
                                 feature_space=feature_space, rel="conn.parquet")

        previous = self.existing_manifest_sha(lake, target)
        if previous and manifest_sha and previous != manifest_sha and not force:
            raise ValueError(
                f"{target} was labelled from manifest {previous}, and this run "
                f"used {manifest_sha}. Overwriting would discard labels derived "
                f"from a different schedule. Re-run with force=True if that is "
                f"intended.")
        return lake.write_parquet(rel, target)
