"""Path B, contrastive branch: two augmented views of a flow, pulled together.

Eslami & Hamouda §IV-C — class-conditioned replacement (Eq. 4-5), constraint
projection (Eq. 6), symmetric NT-Xent (Eq. 7-10) over two heads mixed by lambda
(Eq. 11). Departures from the paper are marked ADAPTED where they appear.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from talos.data.feature.vectorise import untransform
from talos.data.labelling.base import LABELLERS
from talos.data.labelling.behavioural.base import BehaviouralMethod
from talos.data.labelling.behavioural.training import (
    Epoch, as_tensor, minibatches, train,
)
from talos.data.feature.featureset import IDENTITY, LOG1P, RATIO
from talos.parts.base import CLASSIFIERS, ENCODERS, HEADS
from talos.parts.mlp import require_torch

#: Paper values, Table VII (self-generated / ISCX column).
DEFAULTS = {
    "latent_dim": 128, "projection_dim": 128, "hidden": [256, 128, 64],
    "replace_rate": 0.15,          # r, grid {0.05, 0.10, 0.15, 0.20}
    "tau_cont": 0.5, "tau_cat": 0.2,
    "lam": 0.5,                    # Eq. 11 mixing weight
    "epochs": 100, "batch": 256, "lr": 5e-4, "dropout": 0.10,
    "refresh_every": 10,           # pseudo-label refresh interval, in epochs
    "reservoir": 4096,             # ADAPTED: see `ClassMarginals`
    "freeze_epochs": 2, "finetune_epochs": 5, "seed": 0,
    "encoder": "mlp", "head": "mlp", "classifier": "mlp",
}


class ClassMarginals:
    """Per-class, per-feature empirical marginals to draw replacements from.

    ADAPTED. The paper draws from `P(x[j] | y = c)` over the whole pool (Eq. 4).
    An exact marginal over tens of millions of rows is unstorable, so this holds
    a bounded uniform reservoir per class and draws from that — the standard
    estimator of the same distribution, with the sample size declared.
    """

    def __init__(self, n_classes: int, width: int, size: int, seed: int = 0):
        self.size = int(size)
        self.rng = np.random.default_rng(seed)
        self.pools = [np.zeros((0, width)) for _ in range(n_classes)]

    def observe(self, X, labels) -> None:
        """Add a batch, keeping each class's reservoir at its cap."""
        for cls in np.unique(labels):
            rows = X[labels == cls]
            pool = np.vstack([self.pools[cls], rows])
            if len(pool) > self.size:
                keep = self.rng.choice(len(pool), self.size, replace=False)
                pool = pool[keep]
            self.pools[cls] = pool

    def ready(self) -> bool:
        return any(len(pool) for pool in self.pools)

    def draw(self, labels, columns, width: int):
        """One replacement value per (row, selected column), drawn independently.

        Independently per coordinate, as Eq. 4 specifies — this is a marginal
        draw, not a sampled row, so the result need not be a flow that existed.
        """
        out = np.zeros((len(labels), width))
        for cls in np.unique(labels):
            pool = self.pools[cls]
            if not len(pool):
                continue
            rows = labels == cls
            picks = self.rng.integers(0, len(pool), size=(rows.sum(), width))
            out[rows] = np.take_along_axis(pool, picks, axis=0)
        return np.where(columns, out, 0.0)


def project_constraints(X, spec):
    """Pi_G (Eq. 6): make a perturbed row satisfy the declared relations again.

    The paper's "single-pass update for ratio/sum identities": recompute each
    constrained target from its operands. Done in ORIGINAL units, because
    `rate = bytes / duration` is false after `log1p` — the same reason the
    residual penalty untransforms first.
    """
    X = np.array(X, dtype=np.float64, copy=True)
    for _name, form, target, operands in spec.constraints:
        left = untransform(X[:, operands[0]], spec.transforms.get(operands[0], IDENTITY))
        right = untransform(X[:, operands[1]], spec.transforms.get(operands[1], IDENTITY))
        if form == RATIO:
            value = np.divide(left, right, out=np.zeros_like(left), where=right != 0)
        else:
            value = left + right
        if spec.transforms.get(target, IDENTITY) == LOG1P:
            value = np.log1p(np.maximum(value, 0.0))
        X[:, target] = value
    return X


def nt_xent(first, second, temperature: float, torch):
    """Symmetric NT-Xent over 2N views (Eq. 7-10), cosine similarity.

    Negatives are every other view in the batch from this head; the two
    directions are averaged with the paper's 1/(2N) prefactor.
    """
    views = torch.nn.functional.normalize(torch.cat([first, second], dim=0), dim=1)
    similarity = views @ views.T / temperature
    n = first.shape[0]
    # A view is not its own negative; masking with -inf removes it from the sum.
    similarity.fill_diagonal_(float("-inf"))
    positives = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(views.device)
    return torch.nn.functional.cross_entropy(similarity, positives)


@LABELLERS.register
class TabCLLabeller(BehaviouralMethod):
    """Contrastive pretraining over tabular flows, then a head."""

    name: ClassVar[str] = "tabcl"
    extras: ClassVar[tuple[str, ...]] = ("label_margin", "view_agreement", "latent_dim")
    row_extra_names: ClassVar[tuple[str, ...]] = ("view_agreement",)

    def setting(self, key: str):
        return self.settings.get(key, DEFAULTS[key])

    # ------------------------------------------------------------- the parts

    def build_parts(self, width: int) -> dict:
        latent = int(self.setting("latent_dim"))
        projection = int(self.setting("projection_dim"))
        shared = {"seed": int(self.setting("seed")),
                  "device": self.settings.get("device", "auto"),
                  "dropout": float(self.setting("dropout")),
                  "hidden": self.setting("hidden")}
        return {
            "encoder": ENCODERS.get(self.setting("encoder"), input_dim=width,
                                    latent_dim=latent, **shared),
            # Both heads read the FULL latent. The paper's "continuous vs
            # categorical" specialisation is carried by the two temperatures and
            # the mixing weight, not by slicing z (§IV-C-2).
            "head_cont": HEADS.get(self.setting("head"), input_dim=latent,
                                   output_dim=projection, **shared),
            "head_cat": HEADS.get(self.setting("head"), input_dim=latent,
                                  output_dim=projection, **shared),
            "classifier": CLASSIFIERS.get(self.setting("classifier"),
                                          input_dim=latent,
                                          n_classes=self.space.n_classes, **shared),
        }

    @property
    def device(self) -> str:
        return getattr(self.parts["encoder"], "device", "cpu")

    # -------------------------------------------------------- augmentation

    def views(self, X, labels, marginals: ClassMarginals):
        """Two independently-augmented, constraint-projected views (Eq. 4-6)."""
        spec = self.vectoriser.columns()
        rate = float(self.setting("replace_rate"))
        width = X.shape[1]
        k = int(np.ceil(rate * width))

        out = []
        for _ in range(2):
            columns = np.zeros((len(X), width), dtype=bool)
            for row in range(len(X)):
                columns[row, marginals.rng.choice(width, k, replace=False)] = True
            replaced = marginals.draw(labels, columns, width)
            view = np.where(columns, replaced, X)
            out.append(project_constraints(view, spec))
        return out

    # -------------------------------------------------------------- stages

    def pretrain(self, batches, report) -> tuple:
        """Contrastive pretraining, with pseudo-labels refreshed periodically."""
        torch = require_torch()
        encoder = self.parts["encoder"]
        heads = (self.parts["head_cont"], self.parts["head_cat"])
        lam = float(self.setting("lam"))
        taus = (float(self.setting("tau_cont")), float(self.setting("tau_cat")))
        marginals = ClassMarginals(self.space.n_classes,
                                   self.vectoriser.columns().width,
                                   int(self.setting("reservoir")),
                                   int(self.setting("seed")))

        def loss_fn(X_tensor):
            X = X_tensor.detach().to("cpu").numpy().astype(np.float64)
            labels = self.pseudo_labels(X)
            marginals.observe(X, labels)
            if not marginals.ready():
                # First batch of the first epoch: nothing to draw from yet, so
                # the views would be identical and the loss meaningless.
                marginals.observe(X, labels)
            first, second = self.views(X, labels, marginals)
            z1 = encoder.module(as_tensor(first, self.device))
            z2 = encoder.module(as_tensor(second, self.device))
            total = None
            for head, tau, weight in zip(heads, taus, (lam, 1.0 - lam)):
                term = weight * nt_xent(head.module(z1), head.module(z2), tau, torch)
                total = term if total is None else total + term
            return total

        return tuple(train([encoder, *heads], batches, loss_fn,
                           epochs=int(self.setting("epochs")),
                           lr=float(self.setting("lr")), device=self.device))

    def pseudo_labels(self, X):
        """The current probe's opinion, used only to condition the augmentation.

        Computed per batch rather than materialised: at lake scale a stored
        pseudo-label column would be tens of millions of rows rewritten every
        refresh, and the value is derived anyway.
        """
        return np.asarray(self.predict_proba(X)).argmax(axis=1)

    def finetune(self, X, y, report) -> tuple:
        """Freeze the encoder, train the head, then both (Alg. 1 lines 8-9)."""
        torch = require_torch()

        def loss_fn(features, target):
            latent = self.parts["encoder"].module(features)
            return torch.nn.functional.cross_entropy(
                self.parts["classifier"].module(latent), target)

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

    # -------------------------------------------------------------- output

    def predict_proba(self, X):
        return self.parts["classifier"].predict_proba(self.parts["encoder"].encode(X))

    def row_extras(self, X) -> dict[str, np.ndarray]:
        """Do two augmented views of this row still get the same class?

        A row whose views disagree sits near a boundary the encoder has not
        settled; recorded per row so fusion and the audit can see it.
        """
        X = np.asarray(X, dtype=np.float64)
        if not len(X):
            return {"view_agreement": np.zeros(0)}
        labels = self.pseudo_labels(X)
        marginals = ClassMarginals(self.space.n_classes, X.shape[1],
                                   int(self.setting("reservoir")),
                                   int(self.setting("seed")))
        marginals.observe(X, labels)
        first, second = self.views(X, labels, marginals)
        agree = self.pseudo_labels(first) == self.pseudo_labels(second)
        return {"view_agreement": agree.astype(np.float64)}

    def literal_extras(self) -> dict[str, str]:
        return {"latent_dim": f"CAST({int(self.setting('latent_dim'))} AS BIGINT)"}
