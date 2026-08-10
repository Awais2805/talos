"""The lake's shape, independent of where the lake physically lives.

A zone is a stage in the pipeline, and every stage reads zone n and writes zone
n+1 without mutating its input. That contract is what makes provenance possible,
and it holds wherever the root points.

Zone paths are templates rather than fixed names for two reasons. Zones 2-4 are
scoped by the feature space that produced them, so the placeholder set differs
per zone. And a pre-existing lake may not follow the current convention -- one
such lake puts raw captures at `{dataset}/pcaps` -- so a template lets an old
layout and a freshly initialised one be addressed by the same code.
"""

import re
from pathlib import Path

# Order matters: it is the pipeline order, and `init` creates them in it.
ZONE_ORDER = ["raw", "extracted", "parquet", "labelled", "canonical"]

# Zones 2-4 are scoped by feature_space because an extractor defines what a flow
# *is*: the same dataset extracted two ways is two different tables, and giving
# them separate paths makes accidental pooling impossible rather than merely
# discouraged. The canonical zone keys on schema_version, which pins the feature
# space transitively.
DEFAULT_TEMPLATES = {
    "raw": "raw/{dataset}",
    "extracted": "extracted/{feature_space}/{dataset}",
    "parquet": "parquet/{feature_space}/{dataset}",
    "labelled": "labelled/{feature_space}/{dataset}",
    "canonical": "canonical/{schema_version}/{dataset}",
}

DESCRIPTIONS = {
    "raw": "captures, immutable — drop pcaps here",
    "extracted": "extractor output, per feature space",
    "parquet": "one parquet per source log, same tree",
    "labelled": "flows + ground truth",
    "canonical": "registry-governed, train-ready",
}


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


def resolve(root: str, template: str, dataset: str | None = None) -> str:
    """Full URI for one zone, optionally scoped to a dataset.

    Without a dataset the static prefix is returned -- everything up to the
    first `{dataset}` -- which is what listing or globbing a whole zone needs.
    """
    if dataset is None:
        template = re.split(r"\{[a-z_]+\}", template, maxsplit=1)[0]
    else:
        template = template.replace("{dataset}", dataset)
    return join(root, template)


def init(root: str, templates: dict[str, str] | None = None) -> dict[str, tuple[Path, bool]]:
    """Create the zone directories under a local root.

    Returns {zone: (path, created)} -- the path is reported rather than derived
    from the zone name, because the two differ: the `canonical` zone lives at
    `mapped/`, and a legacy lake puts `raw` at `{dataset}/pcaps`.

    Remote roots need no initialisation: object stores have no directories, and
    a prefix springs into existence when the first object is written.
    """
    if is_remote(root):
        raise ValueError(
            f"{root} is a remote lake — object stores have no directories to "
            "create. Set lake.root to a local path to use `talos init`.")

    templates = templates or DEFAULT_TEMPLATES
    base = Path(normalise_root(root))
    result = {}
    for zone in ZONE_ORDER:
        template = templates.get(zone, DEFAULT_TEMPLATES[zone])
        prefix = re.split(r"\{[a-z_]+\}", template, maxsplit=1)[0]
        path = base / prefix.strip("/")
        created = not path.exists()
        path.mkdir(parents=True, exist_ok=True)
        result[zone] = (path, created)
    return result
