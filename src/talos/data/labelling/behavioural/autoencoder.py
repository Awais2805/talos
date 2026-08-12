"""Path B, autoencoder branch: constraint-consistent reconstruction, then a head.

Pretrain encoder+decoder on D_l with squared error on continuous columns,
cross-entropy per one-hot block, and a penalty on rows that break a declared
relation. Then swap the decoder for a classifier and fine-tune on D_s.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from talos.data.feature.vectorise import residuals
from talos.data.labelling.base import LABELLERS
from talos.data.labelling.behavioural.base import BehaviouralMethod
from talos.data.labelling.behavioural.training import (
    Epoch, cross_entropy_blocks, minibatches, train,
)
from talos.parts.base import CLASSIFIERS, DECODERS, ENCODERS
from talos.parts.mlp import require_torch

#: log1p of any representable byte count is under 45; beyond this the untransform
#: overflows and the whole batch becomes NaN.
TRANSFORM_CEILING = 50.0

DEFAULTS = {
    "latent_dim": 16, "epochs": 5, "finetune_epochs": 5, "freeze_epochs": 2,
    "lr": 1e-3, "categorical_weight": 1.0, "residual_weight": 0.1, "seed": 0,
    "encoder": "mlp", "decoder": "mlp", "classifier": "mlp",
}


@LABELLERS.register
class AutoEncoderLabeller(BehaviouralMethod):
    """Reconstruct a flow; a representation that can rebuild one can describe one."""

    name: ClassVar[str] = "autoencoder"
    extras: ClassVar[tuple[str, ...]] = ("recon_error", "latent_dim")

    #: Emitted per row; the rest of `extras` are constants.
    row_extra_names: ClassVar[tuple[str, ...]] = ("recon_error",)

    def setting(self, key: str):
        return self.settings.get(key, DEFAULTS[key])

    # ------------------------------------------------------------- the parts

    def build_parts(self, width: int) -> dict:
        latent = int(self.setting("latent_dim"))
        shared = {"seed": int(self.setting("seed")),
                  "device": self.settings.get("device", "auto"),
                  **({"hidden": self.settings["hidden"]} if "hidden" in self.settings
                     else {})}
        return {
            "encoder": ENCODERS.get(self.setting("encoder"), input_dim=width,
                                    latent_dim=latent, **shared),
            "decoder": DECODERS.get(self.setting("decoder"), latent_dim=latent,
                                    output_dim=width, **shared),
            "classifier": CLASSIFIERS.get(self.setting("classifier"),
                                          input_dim=latent,
                                          n_classes=self.space.n_classes, **shared),
        }

    @property
    def device(self) -> str:
        return getattr(self.parts["encoder"], "device", "cpu")

    # -------------------------------------------------------------- the loss

    def reconstruction_loss(self, X):
        """Squared error, per-block cross-entropy, and the constraint residual."""
        torch = require_torch()
        spec = self.vectoriser.columns()
        Z = self.parts["encoder"].module(X)
        X_hat = self.parts["decoder"].module(Z)

        numeric = list(spec.numeric)
        loss = torch.nn.functional.mse_loss(X_hat[:, numeric], X[:, numeric])
        loss = loss + float(self.setting("categorical_weight")) * cross_entropy_blocks(
            X, X_hat, spec.categorical, torch)
        return loss + float(self.setting("residual_weight")) * self.residual_loss(
            X_hat, X, spec, torch)

    def residual_loss(self, X_hat, X, spec, torch):
        """How implausible the reconstruction is, given the declared relations.

        Applicability comes from the INPUT: whether a value was absent is a fact
        about the record, not about what the decoder produced.
        """
        if not spec.constraints:
            return torch.zeros((), device=X.device)
        bounded = X_hat.clamp(min=-1.0, max=TRANSFORM_CEILING)
        found = residuals(bounded, spec, xp=torch, mask_from=X)
        total = torch.zeros((), device=X.device)
        for residual in found.values():
            total = total + residual.mean_log_square(xp=torch)
        return total

    # ------------------------------------------------------------- the stages

    def pretrain(self, batches, report) -> tuple:
        return tuple(train(
            [self.parts["encoder"], self.parts["decoder"]], batches,
            self.reconstruction_loss, epochs=int(self.setting("epochs")),
            lr=float(self.setting("lr")), device=self.device))

    def finetune(self, X, y, report) -> tuple:
        """Head first with the encoder frozen, then both together.

        Fine-tuning everything from the start lets a randomly-initialised head
        push large gradients through an encoder that took the whole pretraining
        stage to learn anything.
        """
        torch = require_torch()

        def loss_fn(features, target):
            logits = self.parts["classifier"].module(self.parts["encoder"].module(features))
            return torch.nn.functional.cross_entropy(logits, target)

        batches = minibatches(self.batch, X, y, seed=int(self.setting("seed")))
        history: list[Epoch] = list(train(
            [self.parts["classifier"]], batches, loss_fn,
            epochs=int(self.setting("freeze_epochs")), lr=float(self.setting("lr")),
            device=self.device, also=[self.parts["encoder"]]))
        history += train(
            [self.parts["encoder"], self.parts["classifier"]], batches, loss_fn,
            epochs=int(self.setting("finetune_epochs")),
            lr=float(self.setting("lr")), device=self.device)
        return tuple(history)

    # -------------------------------------------------------------- the output

    def predict_proba(self, X):
        return self.parts["classifier"].predict_proba(self.parts["encoder"].encode(X))

    def row_extras(self, X) -> dict[str, np.ndarray]:
        """Per-row reconstruction error — an anomaly signal in its own right."""
        X = np.asarray(X, dtype=np.float64)
        if not len(X):
            return {"recon_error": np.zeros(0)}
        rebuilt = self.parts["decoder"].decode(self.parts["encoder"].encode(X))
        return {"recon_error": ((rebuilt - X) ** 2).mean(axis=1)}

    def literal_extras(self) -> dict[str, str]:
        return {"latent_dim": f"CAST({int(self.setting('latent_dim'))} AS BIGINT)"}
