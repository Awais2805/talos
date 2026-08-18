"""Path B, autoencoder branch: constraint-consistent reconstruction, then a head.

Pretrain encoder+decoder on D_l with squared error on continuous columns,
cross-entropy per one-hot block, and a penalty on rows that break a declared
relation. Then swap the decoder for a classifier and fine-tune on D_s.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from talos.data.feature.featureset import L1
from talos.data.feature.vectorise import residuals
from talos.data.labelling.base import LABELLERS
from talos.data.labelling.behavioural.base import BehaviouralMethod
from talos.data.labelling.behavioural.training import (
    Epoch, binary_cross_entropy_flags, cross_entropy_blocks, minibatches, train,
)
from talos.parts.base import CLASSIFIERS, DECODERS, ENCODERS
from talos.parts.mlp import require_torch

#: log1p of any representable byte count is under 45; beyond this the untransform
#: overflows and the whole batch becomes NaN.
TRANSFORM_CEILING = 50.0

DEFAULTS = {
    "latent_dim": 16, "epochs": 5, "finetune_epochs": 5, "freeze_epochs": 2,
    "lr": 1e-3, "residual_weight": 0.1, "seed": 0,
    "encoder": "mlp", "decoder": "mlp", "classifier": "mlp",
    # Eq. 3 is plain unweighted cross-entropy. An audited D_s is near-balanced
    # by construction -- `CandidateSelector` draws per class up to a floor --
    # so the weighting that once compensated for an unaudited, 80%-benign D_s
    # is not needed when the audit gate is honoured.
    "dropout": 0.0, "weight_decay": 0.0, "class_weighting": False,
    "residual_norm": L1,
}


@LABELLERS.register
class AutoEncoderLabeller(BehaviouralMethod):
    """Reconstruct a flow; a representation that can rebuild one can describe one."""

    name: ClassVar[str] = "autoencoder"
    extras: ClassVar[tuple[str, ...]] = ("label_margin", "recon_error", "latent_dim")

    #: Emitted per row; the rest of `extras` are constants.
    row_extra_names: ClassVar[tuple[str, ...]] = ("recon_error",)

    def setting(self, key: str):
        return self.settings.get(key, DEFAULTS[key])

    # ------------------------------------------------------------- the parts

    def build_parts(self, width: int) -> dict:
        latent = int(self.setting("latent_dim"))
        # `dropout` was absent here while TabCL passed it -- one more asymmetry
        # between two branches Eq. 17 compares on confidence.
        shared = {"seed": int(self.setting("seed")),
                  "device": self.settings.get("device", "auto"),
                  "dropout": float(self.setting("dropout")),
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

        # Eq. 1 splits the loss by column TYPE: squared error over `x_cont`,
        # cross-entropy over `x_cat`. A 0/1 missingness flag is a categorical
        # field with a two-class softmax, so it is scored by the second term,
        # not the first. Every column is scored by exactly one of them --
        # leaving any unscored would let the decoder emit anything there and
        # `recon_error` averages over all of them.
        flags = sorted(spec.indicators.values())
        continuous = [i for i in spec.numeric if i not in set(flags)]
        loss = torch.nn.functional.mse_loss(X_hat[:, continuous], X[:, continuous])
        # Eq. 1 sums MSE and CE with no scalar between them.
        loss = loss + cross_entropy_blocks(X, X_hat, spec.categorical, torch)
        loss = loss + binary_cross_entropy_flags(X, X_hat, flags, torch)
        # Eq. 2's phi_m is applied per constraint, inside.
        return loss + self.residual_loss(X_hat, X, spec, torch)

    def residual_loss(self, X_hat, X, spec, torch):
        """Eq. 2: `sum_m phi_m * ||g_m(x_hat)||_1`, per declared relation.

        Applicability comes from the INPUT: whether a value was absent is a fact
        about the record, not about what the decoder produced. `phi_m` is
        per-constraint, as the paper writes it; a constraint declaring none
        falls back to the method's global `residual_weight`.
        """
        if not spec.constraints:
            return torch.zeros((), device=X.device)
        bounded = X_hat.clamp(min=-1.0, max=TRANSFORM_CEILING)
        found = residuals(bounded, spec, xp=torch, mask_from=X)
        reduction = str(self.setting("residual_norm"))
        weights = spec.constraint_weights
        default = float(self.setting("residual_weight"))
        total = torch.zeros((), device=X.device)
        for name, residual in found.items():
            term = (residual.mean_abs(xp=torch) if reduction == L1
                    else residual.mean_log_square(xp=torch))
            total = total + weights.get(name, default) * term
        return total

    # ------------------------------------------------------------- the stages

    def pretrain(self, batches, report, supervised=None, keyed=None,
                 duck=None) -> tuple:
        """`supervised`/`keyed`/`duck` are TabCL's; Eq. 1 is unsupervised."""
        return tuple(train(
            [self.parts["encoder"], self.parts["decoder"]], batches,
            self.reconstruction_loss, epochs=int(self.setting("epochs")),
            lr=float(self.setting("lr")), device=self.device,
            weight_decay=float(self.setting("weight_decay"))))

    def finetune(self, X, y, report) -> tuple:
        """Head first with the encoder frozen, then both together.

        Fine-tuning everything from the start lets a randomly-initialised head
        push large gradients through an encoder that took the whole pretraining
        stage to learn anything.
        """
        torch = require_torch()
        weight = self.class_weights(y)

        def loss_fn(features, target):
            logits = self.parts["classifier"].module(self.parts["encoder"].module(features))
            return torch.nn.functional.cross_entropy(logits, target, weight=weight)

        decay = float(self.setting("weight_decay"))
        batches = minibatches(self.batch, X, y, seed=int(self.setting("seed")))
        history: list[Epoch] = list(train(
            [self.parts["classifier"]], batches, loss_fn,
            epochs=int(self.setting("freeze_epochs")), lr=float(self.setting("lr")),
            device=self.device, also=[self.parts["encoder"]], weight_decay=decay))
        history += train(
            [self.parts["encoder"], self.parts["classifier"]], batches, loss_fn,
            epochs=int(self.setting("finetune_epochs")),
            lr=float(self.setting("lr")), device=self.device, weight_decay=decay)
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
