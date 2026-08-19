"""Which rows a stage reads: one resolver, every stage.

`zones.fill` truncates at the first unfilled placeholder, which is right for
addressing a whole zone (listing, `talos config`) and catastrophic for reading
data: a caller that forgets `method=` gets the zone's PARENT directory back,
and a glob under it silently reads every method's and every dataset's flows as
if they were one table. That is not a hypothetical -- EDA profiled
`labelled/**/*.parquet` for its whole life, so `--dataset` named the output
file and nothing else.

So a reader never calls `fill` directly. It calls this, which refuses on a
missing scope key instead of truncating, and names a LOG rather than globbing
`*.parquet` -- `dns`, `http` and `ssl` sit in the same capture folder as
`conn`, and `union_by_name` over the four of them is not a flow table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

#: Zeek's flow backbone. Every other log type joins to it by `uid`, so a stage
#: that says "the flows" means this one until supplementary sets are declared.
CONN = "conn"

#: Filled by the caller from the dataset's own declaration, never guessed here.
_DEFAULTED = ("source", "dataset")


class SourceError(Exception):
    pass


def feature_space_of(cfg) -> str:
    """The configured extractor's feature space, without running the extractor.

    Here rather than in the CLI because a feature space is part of a table's
    ADDRESS -- `parquet/{feature_space}/{dataset}` -- so every reader needs it,
    including the stage modules that are runnable on their own.
    """
    from talos.data.extraction import get_extractor
    return get_extractor(cfg.extractor, **cfg.doc.get(cfg.extractor, {})).feature_space


def placeholders(template: str) -> set[str]:
    """The scope keys a template actually needs, read off the template itself.

    Not a fixed list: a lake Talos did not create can declare its own templates
    (`lake.zones`), and the legacy CIC bucket's `labelled/{dataset}` genuinely
    needs neither a feature space nor a method. Requiring a key that zone has
    no slot for would refuse a correct read.
    """
    return set(re.findall(r"\{([a-z_]+)\}", template))


@dataclass(frozen=True)
class FlowTable:
    """One dataset's flows, in one zone, at one point in the pipeline.

    Frozen and hashable because it is an IDENTITY as much as a location: two
    profiles are comparable only if they describe the same `view`, and a
    filename derived from anything less than this collides. `method` is what
    makes a post-label view specific -- schedule, an autoencoder and a fused
    result are three different opinions about the same flows.
    """

    dataset: str
    zone: str
    feature_space: str | None = None
    method: str | None = None
    schema_version: str | None = None
    captures: tuple[str, ...] = ()
    log: str = CONN

    @property
    def view(self) -> str:
        """The comparability key, and the filename component built from it.

        Profiles of different views are not comparable at all: schedule-labelled
        2017 against ae-labelled 2018 measures the labeller, not the domain. So
        the view travels in the profile and in its name, and `compare` groups on
        it rather than trusting a directory to hold one kind of thing.
        """
        if self.zone == "labelled":
            return f"postlabel-{self.method}" if self.method else "postlabel"
        if self.zone == "parquet":
            return "prelabel"
        return self.zone

    @property
    def slug(self) -> str:
        return f"{self.view}_{self.dataset}"

    def scope(self, cfg) -> dict:
        return {"feature_space": self.feature_space, "method": self.method,
                "schema_version": self.schema_version,
                "source": cfg.source_of(self.dataset)}

    def base(self, cfg, lake) -> str:
        """The zone directory, with every placeholder the template declares filled."""
        try:
            template = lake.templates[self.zone]
        except KeyError:
            raise SourceError(f"unknown zone {self.zone!r}; known: "
                              f"{', '.join(sorted(lake.templates))}") from None
        scope = self.scope(cfg)
        missing = sorted(k for k in placeholders(template)
                         if k not in _DEFAULTED and scope.get(k) is None)
        if missing:
            raise SourceError(
                f"cannot read the {self.zone} zone without {', '.join(missing)}: "
                f"its layout is {template!r}. Resolving it anyway would truncate "
                f"the path at {'{'}{missing[0]}{'}'} and read every "
                f"{missing[0].replace('_', ' ')} under it as one table.")
        return lake.uri(self.zone, dataset=self.dataset, **scope)

    def globs(self, cfg, lake) -> list[str]:
        """Parquet globs for this table, one per capture when captures are named.

        The `parquet` zone mirrors Zeek's per-capture tree (`Monday/conn.parquet`,
        `Friday-WorkingHours/conn.parquet`); every zone downstream of labelling
        holds one consolidated file per dataset, because the labeller joins the
        captures before it writes.
        """
        base = self.base(cfg, lake)
        if not lake.list(base):
            raise SourceError(f"nothing to read: {base} has no files.")
        if self.zone != "parquet":
            return [f"{base}/{self.log}.parquet"]
        if self.captures:
            return [f"{base}/{c}/{self.log}.parquet" for c in self.captures]
        return [f"{base}/**/{self.log}.parquet"]


def tables(cfg, datasets: Sequence[str], zone: str, *, feature_space=None,
           method=None, schema_version=None, captures=None, log=CONN) -> list[FlowTable]:
    """One `FlowTable` per dataset, all sharing a scope."""
    return [FlowTable(dataset=d, zone=zone, feature_space=feature_space,
                      method=method if zone == "labelled" else None,
                      schema_version=schema_version,
                      captures=tuple(captures or ()), log=log)
            for d in datasets]


def globs(cfg, lake, datasets: Sequence[str], zone: str, *, source: str | None = None,
          **scope) -> list[str]:
    """Every glob for several datasets read as one relation.

    `source` is the raw-glob escape hatch every reading CLI already exposes:
    an explicit path is the caller taking responsibility for scope, so nothing
    is resolved or checked.
    """
    if source:
        return [source]
    out: list[str] = []
    for table in tables(cfg, datasets, zone, **scope):
        out.extend(table.globs(cfg, lake))
    return out
