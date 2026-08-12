"""Task-free parts: encoders, decoders, projection heads, classifiers.

Nothing here may know what a flow is — a test enforces it — so any stage can
assemble the same parts without importing another. No part owns a loss or a
training loop; those belong to whoever assembles them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from talos.common.plugins import CodePlugins


class PartError(Exception):
    pass


#: `numpy` parts cannot learn; they exist so the machinery can run without torch.
NUMPY = "numpy"
TORCH = "torch"
BACKENDS = (NUMPY, TORCH)


class Part(ABC):
    """One replaceable piece of a model."""

    #: Read off the CLASS at import time, so registering constructs nothing.
    name: ClassVar[str] = ""
    kind: ClassVar[str] = ""
    backend: ClassVar[str] = NUMPY

    #: False means this part CANNOT LEARN, and an assembly must say so.
    trainable: ClassVar[bool] = False

    def __init__(self, **settings: Any):
        self.settings = dict(settings)

    # ---------------------------------------------------------------- shape

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Columns this part expects."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Columns this part produces."""

    def check_input(self, X) -> None:
        """Refuse a width mismatch here, where both numbers can be named."""
        width = getattr(X, "shape", (None, None))[-1]
        if width is not None and width != self.input_dim:
            raise PartError(
                f"{self.kind} {self.name!r} takes {self.input_dim} columns and "
                f"was given {width}. The declaration and this part's "
                f"configuration disagree about how wide a row is.")

    # ------------------------------------------------------------- lifecycle

    def config(self) -> dict[str, Any]:
        """What this part was built with, for the run report and the checkpoint."""
        return {"name": self.name, "kind": self.kind, "backend": self.backend,
                "trainable": self.trainable, "input_dim": self.input_dim,
                "output_dim": self.output_dim, **self.settings}

    def save(self, path: str | Path) -> None:
        """Persist learned state. A part with none writes nothing."""

    def load(self, path: str | Path) -> None:
        """Restore learned state written by `save`."""

    def describe(self) -> str:
        learns = "" if self.trainable else "   [cannot learn]"
        return (f"{self.kind:<16}{self.name:<12}{self.input_dim:>6} -> "
                f"{self.output_dim:<6} {self.backend}{learns}")


# ------------------------------------------------------------------ the parts

class Encoder(Part):
    """Rows in, latent representation out."""

    kind: ClassVar[str] = "encoder"

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        ...

    @property
    def output_dim(self) -> int:
        return self.latent_dim

    @abstractmethod
    def encode(self, X):
        """(n, input_dim) -> (n, latent_dim)."""


class Decoder(Part):
    """Latent representation in, a reconstruction of the original row out."""

    kind: ClassVar[str] = "decoder"

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        ...

    @property
    def input_dim(self) -> int:
        return self.latent_dim

    @abstractmethod
    def decode(self, Z):
        """(n, latent_dim) -> (n, output_dim)."""


class ProjectionHead(Part):
    """A representation mapped into the space a contrastive loss works in.

    Separate from the encoder because the projection is discarded after
    pretraining and the encoder is kept.
    """

    kind: ClassVar[str] = "head"

    @abstractmethod
    def project(self, Z):
        """(n, input_dim) -> (n, output_dim)."""


class Classifier(Part):
    """A representation in, a distribution over classes out."""

    kind: ClassVar[str] = "classifier"

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """How many classes there are. Only the count reaches here, never a name."""

    @property
    def output_dim(self) -> int:
        return self.n_classes

    @abstractmethod
    def predict_proba(self, Z):
        """(n, input_dim) -> (n, n_classes), rows summing to 1.

        A distribution, never a hard class: confident learning consumes exactly
        this, and an argmax thrown away here cannot be recovered later.
        """


# ------------------------------------------------------------------- registries

ENCODERS = CodePlugins("encoder", error=PartError)
DECODERS = CodePlugins("decoder", error=PartError)
HEADS = CodePlugins("head", error=PartError)
CLASSIFIERS = CodePlugins("classifier", error=PartError)

#: Keyed by the `kind` a part declares, so an assembly resolves `encoder: mlp`.
POINTS_BY_KIND = {
    Encoder.kind: ENCODERS,
    Decoder.kind: DECODERS,
    ProjectionHead.kind: HEADS,
    Classifier.kind: CLASSIFIERS,
}


def build(kind: str, name: str, **settings) -> Part:
    """Construct a part by kind and name, refusing an unknown kind by name."""
    point = POINTS_BY_KIND.get(kind)
    if point is None:
        raise PartError(f"{kind!r} is not a kind of part. Have: "
                      f"{', '.join(sorted(POINTS_BY_KIND))}")
    return point.get(name, **settings)
