"""What a labelling run declares, as opposed to what performs it.

Two different things wear the word "method" and keeping them apart is the point
of this module:

    the IMPLEMENTATION   a class, registered in `LABELLERS`, e.g. `autoencoder`
    the DECLARATION      a `method.yaml`, e.g. `ae` and `tabcl`


Declarations ship inside the package, beside the code that reads them, exactly as
manifests and the EDA spec do. A user brings their own with `--method-file` or a
`plugins: method:` directory, and `Origin` says which was used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from talos.common.plugins import ConfigPlugins, Declaration, Origin
from talos.common.provenance import Sha12, composite_sha
from talos.data.labelling.base import LABELLERS, LabellingError
from talos.data.labelling.space import DEFAULT_SPACE, LabelSpace, LabelSpaceLoader
from talos.data.labelling.taxonomy import (
    DEFAULT_TAXONOMY, TAXONOMIES, TaxonomyMapper,
)

LABELLING_DIR = Path(__file__).parent
METHOD_FILE = "method.yaml"

#: Directories that ship declarations. Each labelling family keeps its own
#: beside its code, so this grows by one entry per family rather than by a
#: central list of every method.
BUILT_IN_DIRS = (LABELLING_DIR, LABELLING_DIR / "behavioural",
                 LABELLING_DIR / "fusion")


class MethodError(LabellingError):
    pass


#: The method-declaration plug-in point.
METHODS = ConfigPlugins("method", built_in=BUILT_IN_DIRS, filename=METHOD_FILE,
                        error=MethodError)


@dataclass(frozen=True)
class MethodSpec:
    """One declared labelling run: which code, over which vocabulary, how."""

    name: str
    implementation: str
    label_space: LabelSpace
    taxonomy: str
    settings: Mapping[str, Any]
    sha: Sha12
    path: Path
    origin: Origin
    doc: Mapping[str, Any] = field(default_factory=dict)

    @property
    def built_in(self) -> bool:
        return self.origin.built_in

    def inputs(self) -> tuple[Path, ...]:
        """Every file that determines this method's output, except per-dataset ones.

        A per-dataset input -- schedule labelling's manifest -- is added by the
        method itself, because only it knows there is one.
        """
        # `path_for`, not `TaxonomyMapper(...).path`: the mapper parses and
        # validates a whole YAML file, and this is called on every `method_sha`.
        return (self.path, self.label_space.path, TAXONOMIES.path_for(self.taxonomy))

    def method_sha(self, *extra: Path) -> Sha12:
        """Composite over the declaration and everything it names.

        One hash that moves when ANY input does, so a row needs one provenance
        column instead of one per input. Changing the latent width, remapping an
        attack name and excluding a class all move it.
        """
        return composite_sha(*self.inputs(), *extra)

    def build(self, cfg, **kwargs):
        """Construct the implementation this declaration names."""
        return LABELLERS.get(self.implementation, cfg=cfg, spec=self, **kwargs)

    def describe(self) -> str:
        whose = "" if self.built_in else f"   [{self.origin.where} {self.path}]"
        return (f"{self.name}  ({self.implementation})  sha {self.sha}{whose}\n"
                f"  {self.label_space.describe()}")


class MethodLoader:
    """Reads a method declaration and refuses the ways it can be wrong."""

    def __init__(self, point: ConfigPlugins = METHODS,
                 spaces: LabelSpaceLoader | None = None):
        self.point = point
        self.spaces = spaces or LabelSpaceLoader()

    def load(self, name: str | Path) -> MethodSpec:
        # Implementations exist only once their module has been imported, and
        # this validates a declaration against them. Without this the answer to
        # "is `schedule` a real implementation" depends on what the caller
        # happened to import first -- the same trap as W7.9, one layer up.
        # Imported inside the call, not at module scope: `points` imports this.
        import talos.points                       # noqa: F401

        declaration = self.point.get(name)
        doc = declaration.doc
        resolved = doc.get("name", declaration.name)
        implementation = doc.get("implementation", resolved)
        taxonomy = doc.get("taxonomy", DEFAULT_TAXONOMY)

        self._validate(declaration, resolved, implementation)
        space = self.spaces.load(doc.get("label_space", DEFAULT_SPACE),
                                 TaxonomyMapper(taxonomy))

        return MethodSpec(
            name=resolved, implementation=implementation, label_space=space,
            taxonomy=taxonomy, settings=doc.get("settings") or {},
            sha=declaration.sha, path=declaration.path,
            origin=declaration.origin, doc=doc)

    @staticmethod
    def _validate(declaration: Declaration, name: str, implementation: str) -> None:
        errs = []
        if implementation not in LABELLERS:
            errs.append(
                f"names implementation {implementation!r}, which is not a "
                f"registered labelling method. Have: "
                f"{', '.join(LABELLERS.names()) or 'none'}. Bring your own with "
                f"`plugins: labelling method:` in config.yml")
        if "/" in name or name != name.strip():
            errs.append(f"name {name!r} becomes a directory in the labelled zone, "
                        f"so it may not contain a path separator or stray space")
        if errs:
            raise MethodError(f"{declaration.path}:\n  " + "\n  ".join(errs))
