"""Model parts, shared by every stage that assembles a model.

An MLP encoder is a part, not a model. It lives here rather than beside the
first stage that needed one, so `engine/L2` does not import backwards through
the architecture. Importing this package never imports torch.
"""

from talos.parts.base import (
    BACKENDS, CLASSIFIERS, DECODERS, ENCODERS, HEADS, NUMPY, POINTS_BY_KIND,
    TORCH, Classifier, Decoder, Encoder, PartError, Part, ProjectionHead, build,
)
from talos.parts import mlp as _mlp        # noqa: F401  registers the torch parts
from talos.parts import inert as _inert    # noqa: F401  registers the inert parts

__all__ = [
    "BACKENDS", "CLASSIFIERS", "DECODERS", "ENCODERS", "HEADS", "NUMPY",
    "POINTS_BY_KIND", "TORCH", "Classifier", "Decoder", "Encoder", "PartError",
    "Part", "ProjectionHead", "build",
]
