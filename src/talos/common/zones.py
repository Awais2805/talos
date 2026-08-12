"""The lake's shape, independent of where the lake physically lives.

A zone is a stage in the pipeline, and every stage reads zone n and writes zone
n+1 without mutating its input. That contract is what makes provenance possible,
and it holds wherever the root points.

**The lake is source-major.** Talos takes traffic from three kinds of place, and
they are not interchangeable

    lake/
      lake.yaml
      sources/
        datasets/            public IDS datasets
          raw/{dataset}
          extracted/{feature_space}/{dataset}
          parquet/{feature_space}/{dataset}
          labelled/{feature_space}/{dataset}
        netem/               emulator captures — labels known by construction
        honeypot/            live capture — no schedule, behavioural labelling only
      canonical/{schema_version}/{dataset}

The end-user flow the layout is built for: pick a source, drop pcaps into its
raw zone, watch them move zone by zone.

"""

import re
from dataclasses import dataclass
from pathlib import Path

# Order matters: it is the pipeline order, and `init` creates them in it.
ZONE_ORDER = ["raw", "extracted", "parquet", "labelled", "canonical"]

# The three kinds of place traffic comes from. Not a taxonomy of datasets -- a
# taxonomy of PROVENANCE, which is why it decides what labelling can be applied.
SOURCES = ("datasets", "netem", "honeypot")
DEFAULT_SOURCE = "datasets"

# Zones 1-4 are scoped by `source`, and 2-4 additionally by `feature_space`
# because an extractor defines what a flow *is*: the same capture extracted two
# ways is two different tables, and separate paths make accidental pooling
# impossible rather than merely discouraged. The canonical zone keys on
# schema_version, which pins the feature space transitively.
DEFAULT_TEMPLATES = {
    "raw": "sources/{source}/raw/{dataset}",
    "extracted": "sources/{source}/extracted/{feature_space}/{dataset}",
    "parquet": "sources/{source}/parquet/{feature_space}/{dataset}",
    # `{method}` scopes the labelled zone for the same reason `{feature_space}`
    # scopes the three above it: schedule labelling, an autoencoder and a fused
    # result are three different opinions about the same flows, and separate
    # paths make pooling them by accident impossible rather than discouraged.
    # Fusion and the three-way benchmark read several of these deliberately.
    "labelled": "sources/{source}/labelled/{feature_space}/{method}/{dataset}",
    "canonical": "canonical/{schema_version}/{dataset}",
    # An EXPORT, not a pipeline stage: nothing reads it, it exists for people
    # and other tools. Addressable like any zone so no stage has to invent a
    # path, but deliberately absent from ZONE_ORDER so `init` does not create it
    # and `config` does not list it as a step.
    "csv": "sources/{source}/csv/{feature_space}/{dataset}",
}

DESCRIPTIONS = {
    "csv": "human-readable export of the parquet zone",
    "raw": "captures, immutable — drop pcaps here",
    "extracted": "extractor output, per feature space",
    "parquet": "one parquet per source log, same tree",
    "labelled": "flows + ground truth",
    "canonical": "registry-governed, train-ready — all sources merged",
}

SOURCE_DESCRIPTIONS = {
    "csv": "human-readable export of the parquet zone",
    "datasets": "public IDS datasets — labelled from a published schedule",
    "netem": "emulator captures — labels known by construction",
    "honeypot": "live capture — no schedule, behavioural labelling only",
}


@dataclass(frozen=True)
class ZoneDir:
    """One zone location, and where it belongs.

    `uri` is a lake URI, not a Path: a lake may be a directory, a bucket or
    anything else fsspec can reach, and this module has no business knowing
    which. `created` is filled in by whoever materialises the plan.
    """

    zone: str
    source: str | None          # None for a zone shared across sources
    uri: str
    created: bool = False

    @property
    def name(self) -> str:
        """Last segment — what the zone is actually called on disk."""
        return self.uri.rstrip("/").rsplit("/", 1)[-1]


def is_remote(root: str) -> bool:
    """True when the lake root is an object store rather than a local path."""
    return "://" in str(root) and not str(root).startswith("file://")


def normalise_root(root: str) -> str:
    """Strip a trailing slash, and expand `file://` and `~` for local roots."""
    root = str(root).rstrip("/")
    if root.startswith("file://"):
        root = root[len("file://"):]
    if not is_remote(root):
        root = str(Path(root).expanduser())
    return root


def join(root: str, *parts: str) -> str:
    """Join under a root that may be a URI or a local path."""
    tail = "/".join(str(p).strip("/") for p in parts if str(p).strip("/"))
    root = normalise_root(root)
    return f"{root}/{tail}" if tail else root


def fill(template: str, **values) -> str:
    """Substitute a zone template's placeholders, stopping at the first unfilled one.

    Truncating rather than raising is what makes a whole zone addressable:
    everything up to the first unknown placeholder is the static prefix that
    listing or globbing needs. The walk follows the order the placeholders
    appear in the template, not the order the caller passed them, because which
    one comes first is a property of the path -- `extracted/{feature_space}/
    {dataset}` cannot be scoped by dataset alone, and truncating there is the
    honest answer rather than a path with a literal `{feature_space}` in it.
    """
    for token in re.findall(r"\{([a-z_]+)\}", template):
        value = values.get(token)
        if value is None:
            return template.split("{" + token + "}", 1)[0]
        template = template.replace("{" + token + "}", str(value))
    return template


def resolve(root: str, template: str, dataset: str | None = None, **scope) -> str:
    """Full URI for one zone, optionally scoped to a dataset and a feature space.

    Without a dataset the static prefix is returned, which is what listing or
    globbing a whole zone needs.
    """
    return join(root, fill(template, dataset=dataset, **scope))


def is_source_scoped(template: str) -> bool:
    """Whether a zone lives under a source.

    Read off the template, exactly as feature-space scoping is. A lake that
    declares a zone without `{source}` gets a shared one, and nothing else has
    to be told.
    """
    return "{source}" in template


def plan(root: str, templates: dict[str, str] | None = None,
         sources: tuple[str, ...] = SOURCES) -> list[ZoneDir]:
    """Every zone location a lake at `root` should have. Pure — touches nothing.

    A user who installs Talos and initialises a lake should get somewhere to put
    pcaps without reading any documentation: three sources, each with its zones,
    and the shared canonical zone alongside.

    Locations are reported rather than derived from zone names, because the two
    differ — a legacy lake puts `raw` at `{dataset}/pcaps` and its canonical zone
    at `mapped/`.

    This function does no I/O at all, which is what lets one plan serve a
    directory, a bucket, or anything else fsspec reaches. Materialising it is
    `LakeClient.init`, because that is the only class allowed to touch storage.
    """
    templates = templates or DEFAULT_TEMPLATES
    out: list[ZoneDir] = []
    for zone in ZONE_ORDER:
        template = templates.get(zone, DEFAULT_TEMPLATES[zone])
        scopes = sources if is_source_scoped(template) else (None,)
        for source in scopes:
            prefix = fill(template, source=source) if source else fill(template)
            out.append(ZoneDir(zone=zone, source=source, uri=join(root, prefix)))
    return out
