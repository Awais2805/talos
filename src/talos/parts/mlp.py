"""Multi-layer perceptron parts, on torch imported lazily.

They REGISTER even when torch is absent and fail when constructed: registering
nothing gives "unknown encoder 'mlp'", which reads as a typo, not a missing
dependency. Arrays in and out for inference; `.module` is the training surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np

from talos.parts.base import (
    CLASSIFIERS, DECODERS, ENCODERS, HEADS, Classifier, Decoder, Encoder,
    PartError, ProjectionHead, TORCH,
)

DEFAULT_HIDDEN = (128, 64)
DEFAULT_DROPOUT = 0.0
DEFAULT_SEED = 0


def torch_available() -> bool:
    """Whether the `[torch]` extra is installed. Imports nothing."""
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def require_torch():
    """The torch module, or the error naming what to install."""
    try:
        import torch
    except ImportError as exc:                           
        raise PartError(
            "this part needs PyTorch, which is not installed. Install the extra "
            "with `pip install -e '.[torch]'`, or use the `null` parts to exercise "
            "the pipeline without a model.") from exc
    return torch


def pick_device(requested: str = "auto") -> str:
    """`auto` resolves to the best available; anything else is taken literally.

    Literal because a run that silently fell back to CPU after being told `cuda`
    has produced a timing result that means nothing.
    """
    if requested != "auto":
        return requested
    torch = require_torch()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stack(dims: Sequence[int], dropout: float, final_activation: bool):
    """A Linear/ReLU/Dropout tower over consecutive widths in `dims`."""
    torch = require_torch()
    layers: list = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        # A latent space or a logit layer is linear; only inner layers activate.
        if i != len(dims) - 2 or final_activation:
            layers.append(torch.nn.ReLU())
            if dropout:
                layers.append(torch.nn.Dropout(dropout))
    return torch.nn.Sequential(*layers)


class TorchPart:
    """State handling shared by every torch part. Not a `Part` itself."""

    module = None
    settings: dict

    def _build(self, dims: Sequence[int], *, final_activation: bool = False):
        torch = require_torch()
        # Seeded at construction: weight init is the hardest thing to reproduce later.
        torch.manual_seed(int(self.settings.get("seed", DEFAULT_SEED)))
        self.device = pick_device(str(self.settings.get("device", "auto")))
        self.module = stack(dims, float(self.settings.get("dropout", DEFAULT_DROPOUT)),
                            final_activation).to(self.device)

    def _forward(self, A):
        """Arrays in, arrays out, gradients off. The inference surface."""
        torch = require_torch()
        self.check_input(A)                              # type: ignore[attr-defined]
        tensor = torch.as_tensor(np.asarray(A, dtype=np.float32), device=self.device)
        self.module.eval()
        with torch.no_grad():
            out = self.module(tensor)
        return out.detach().to("cpu").numpy().astype(np.float64)

    def parameters(self):
        """For the assembly's optimiser."""
        return self.module.parameters()

    def save(self, path: str | Path) -> None:
        torch = require_torch()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.module.state_dict(), str(path))

    def load(self, path: str | Path) -> None:
        torch = require_torch()
        self.module.load_state_dict(torch.load(str(path), map_location=self.device))


def hidden_of(settings: dict) -> tuple[int, ...]:
    return tuple(int(h) for h in settings.get("hidden", DEFAULT_HIDDEN))


# ------------------------------------------------------------------- the parts

@ENCODERS.register
class MLPEncoder(TorchPart, Encoder):
    name: ClassVar[str] = "mlp"
    backend: ClassVar[str] = TORCH
    trainable: ClassVar[bool] = True

    def __init__(self, input_dim: int, latent_dim: int = 16, **settings):
        Encoder.__init__(self, input_dim=input_dim, latent_dim=latent_dim,
                         **settings)
        self._input_dim, self._latent_dim = int(input_dim), int(latent_dim)
        self._build((self._input_dim, *hidden_of(self.settings), self._latent_dim))

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def encode(self, X):
        return self._forward(X)


@DECODERS.register
class MLPDecoder(TorchPart, Decoder):
    name: ClassVar[str] = "mlp"
    backend: ClassVar[str] = TORCH
    trainable: ClassVar[bool] = True

    def __init__(self, latent_dim: int, output_dim: int, **settings):
        Decoder.__init__(self, latent_dim=latent_dim, output_dim=output_dim,
                         **settings)
        self._latent_dim, self._output_dim = int(latent_dim), int(output_dim)
        # Mirrored: the encoder's tower read backwards unless declared otherwise.
        self._build((self._latent_dim, *reversed(hidden_of(self.settings)),
                     self._output_dim))

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def decode(self, Z):
        return self._forward(Z)


@HEADS.register
class MLPHead(TorchPart, ProjectionHead):
    """One hidden layer by default — the shallow head contrastive work uses."""

    name: ClassVar[str] = "mlp"
    backend: ClassVar[str] = TORCH
    trainable: ClassVar[bool] = True

    def __init__(self, input_dim: int, output_dim: int = 32, **settings):
        ProjectionHead.__init__(self, input_dim=input_dim, output_dim=output_dim,
                                **settings)
        self._input_dim, self._output_dim = int(input_dim), int(output_dim)
        hidden = tuple(int(h) for h in self.settings.get("hidden", (input_dim,)))
        self._build((self._input_dim, *hidden, self._output_dim))

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def project(self, Z):
        return self._forward(Z)


@CLASSIFIERS.register
class MLPClassifier(TorchPart, Classifier):
    """Emits logits; `predict_proba` is their softmax.

    The module stays logit-valued because every loss worth using here is defined
    on logits.
    """

    name: ClassVar[str] = "mlp"
    backend: ClassVar[str] = TORCH
    trainable: ClassVar[bool] = True

    def __init__(self, input_dim: int, n_classes: int, **settings):
        Classifier.__init__(self, input_dim=input_dim, n_classes=n_classes,
                            **settings)
        self._input_dim, self._n_classes = int(input_dim), int(n_classes)
        if self._n_classes < 2:
            raise PartError(
                f"a classifier over {self._n_classes} class(es) cannot express a "
                f"decision. Whatever declares the classes must declare two or more.")
        self._build((self._input_dim, *hidden_of(self.settings), self._n_classes))

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def predict_proba(self, Z):
        logits = self._forward(Z)
        shifted = logits - logits.max(axis=1, keepdims=True)      # for stability
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
