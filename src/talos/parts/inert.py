"""Parts that produce correctly-shaped output and learn nothing.

They let the machinery around a model be executed and tested before a model is
written into it. All four declare `trainable = False`, which is what an assembly
reads to refuse to present their output as a model's opinion.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from talos.parts.base import (
    CLASSIFIERS, DECODERS, ENCODERS, HEADS, Classifier, Decoder, Encoder,
    PartError, NUMPY, ProjectionHead,
)


def fit_width(A, width: int):
    """Truncate or zero-pad columns to `width`. Deterministic and stateless."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        A = A.reshape(len(A), -1)
    if A.shape[1] == width:
        return A
    if A.shape[1] > width:
        return A[:, :width]
    return np.hstack([A, np.zeros((A.shape[0], width - A.shape[1]))])


@ENCODERS.register
class InertEncoder(Encoder):
    """Shape-correct, information-destroying."""

    name: ClassVar[str] = "inert"
    backend: ClassVar[str] = NUMPY
    trainable: ClassVar[bool] = False

    def __init__(self, input_dim: int, latent_dim: int = 8, **settings):
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, **settings)
        self._input_dim = int(input_dim)
        self._latent_dim = int(latent_dim)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def encode(self, X):
        self.check_input(X)
        return fit_width(X, self.latent_dim)


@DECODERS.register
class InertDecoder(Decoder):
    """Returns something the right shape, which reconstructs nothing."""

    name: ClassVar[str] = "inert"
    backend: ClassVar[str] = NUMPY
    trainable: ClassVar[bool] = False

    def __init__(self, latent_dim: int, output_dim: int, **settings):
        super().__init__(latent_dim=latent_dim, output_dim=output_dim, **settings)
        self._latent_dim = int(latent_dim)
        self._output_dim = int(output_dim)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def decode(self, Z):
        self.check_input(Z)
        return fit_width(Z, self.output_dim)


@HEADS.register
class InertHead(ProjectionHead):
    name: ClassVar[str] = "inert"
    backend: ClassVar[str] = NUMPY
    trainable: ClassVar[bool] = False

    def __init__(self, input_dim: int, output_dim: int = 8, **settings):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **settings)
        self._input_dim = int(input_dim)
        self._output_dim = int(output_dim)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def project(self, Z):
        self.check_input(Z)
        return fit_width(Z, self.output_dim)


@CLASSIFIERS.register
class InertClassifier(Classifier):
    """A uniform distribution — the lowest expressible confidence, which is honest.

    A majority-class predictor would score respectably and look like a signal.
    """

    name: ClassVar[str] = "inert"
    backend: ClassVar[str] = NUMPY
    trainable: ClassVar[bool] = False

    def __init__(self, input_dim: int, n_classes: int, **settings):
        super().__init__(input_dim=input_dim, n_classes=n_classes, **settings)
        self._input_dim = int(input_dim)
        self._n_classes = int(n_classes)
        if self._n_classes < 2:
            raise PartError(
                f"a classifier over {self._n_classes} class(es) cannot express a "
                f"decision. Whatever declares the classes must declare two or more.")

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def predict_proba(self, Z):
        self.check_input(Z)
        rows = np.asarray(Z).shape[0]
        return np.full((rows, self.n_classes), 1.0 / self.n_classes)
