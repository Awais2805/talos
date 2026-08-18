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
    "weight_decay": 0.0, "class_weighting": False,   # Eq. 3 is unweighted
    # Alg. 2 line 1's "light classifier". Table VII has no bootstrap column,
    # but it does publish an architecture for its OTHER simple-classifier role
    # -- the Confident Learning column -- so that is used rather than an
    # invented one: 2 hidden layers (128, 64), dropout 0.20, lr 1e-3, batch 64,
    # 30 epochs. Notably NOT the main classifier's (256,128,64), which would
    # memorise a D_s of a few thousand rows and poison c_i for the first span.
    "bootstrap_hidden": [128, 64], "bootstrap_dropout": 0.20,
    "bootstrap_lr": 0.001, "bootstrap_batch": 64, "bootstrap_epochs": 30,
    # Alg. 2 line 5. The paper names a tolerance and a round cap but publishes
    # neither value; these are ours and are declared as such.
    "probe_epochs": 10, "refresh_tolerance": 0.01, "max_refresh_rounds": 10,
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
        self.seen = [0] * n_classes

    def observe(self, X, labels) -> None:
        """Add a batch by Algorithm R, so the reservoir stays uniform.

        Subsampling `old ∪ new` down to the cap each call is NOT uniform: an
        item's survival decays as (S/(S+B))^t, so the pool drifts toward
        whatever the stream showed last. Batches arrive in table order, so that
        would skew the marginal toward the tail of the table.
        """
        for cls in np.unique(labels):
            rows = X[labels == cls]
            pool, seen = self.pools[cls], self.seen[cls]
            free = self.size - len(pool)
            if free > 0:
                pool = np.vstack([pool, rows[:free]])
                rows = rows[free:]
            for row in rows:
                seen += 1
                # Replace a uniformly-chosen slot with probability S/n.
                slot = self.rng.integers(0, len(pool) + seen)
                if slot < len(pool):
                    pool[slot] = row
            self.pools[cls] = pool
            self.seen[cls] = seen

    def ready(self) -> bool:
        return any(len(pool) for pool in self.pools)

    def draw(self, labels, units, width: int):
        """One donor per (row, selected unit), drawn independently per unit.

        A UNIT is one declared feature: a numeric column with its missingness
        indicator, or a whole one-hot block. Eq. 4's `d` counts features, not
        matrix columns -- replacing columns inside a one-hot block would leave
        two 1.0s or an all-zero block, and Pi_G only repairs continuous
        identities, so an invalid encoding would reach the encoder intact.
        """
        out = np.zeros((len(labels), width))
        for cls in np.unique(labels):
            pool = self.pools[cls]
            if not len(pool):
                continue
            rows = np.flatnonzero(labels == cls)
            # One donor row per (row, unit): within a unit the columns must
            # come from the SAME donor or the block stops being a distribution.
            picks = self.rng.integers(0, len(pool), size=(len(rows), len(units)))
            for u, unit in enumerate(units):
                out[np.ix_(rows, unit)] = pool[picks[:, u]][:, unit]
        return out


def view_indices(rng, n_units: int, k: int):
    """Eq. 4's two index sets for one anchor, with minimal overlap.

    The paper samples them "independently (preferably with minimal overlap)".
    Drawing 2k distinct units once and halving them makes the overlap exactly
    zero whenever 2k <= d, and is one draw rather than two. Where the rate is
    high enough that 2k exceeds d, no disjoint pair exists and it falls back to
    the literal independent draws.
    """
    if 2 * k <= n_units:
        both = rng.choice(n_units, 2 * k, replace=False)
        return both[:k], both[k:]
    return (rng.choice(n_units, k, replace=False),
            rng.choice(n_units, k, replace=False))


def feature_units(spec):
    """Column indices grouped into the DECLARED features they belong to.

    A numeric feature carries its missingness indicator with it; a categorical
    occupies a whole one-hot block. Replacing at column granularity would split
    both.
    """
    blocked = {i for block in spec.categorical for i in block.indices}
    paired = set(spec.indicators.values())
    units = [tuple(block.indices) for block in spec.categorical]
    for index in spec.numeric:
        if index in blocked or index in paired:
            continue
        indicator = spec.indicators.get(index)
        units.append((index,) if indicator is None else (index, indicator))
    return units


def project_constraints(X, spec, replaced=None):
    """Pi_G (Eq. 6): make a perturbed row satisfy the declared relations again.

    The paper's "single-pass update for ratio/sum identities": recompute each
    constrained target from its operands. Done in ORIGINAL units, because
    `rate = bytes / duration` is false after `log1p` — the same reason the
    residual penalty untransforms first.
    """
    X = np.array(X, dtype=np.float64, copy=True)
    for _name, form, target, operands in spec.constraints:
        left = spec.undo(operands[0], X[:, operands[0]])
        right = spec.undo(operands[1], X[:, operands[1]])
        if form == RATIO:
            value = np.divide(left, right, out=np.zeros_like(left), where=right != 0)
        else:
            value = left + right
        if spec.transforms.get(target, IDENTITY) == LOG1P:
            value = np.log1p(np.maximum(value, 0.0))
        # Standardisation was applied after the transform, so it is restored
        # last -- writing the raw value back would put the target on a scale
        # the encoder never saw.
        centre, scale = spec.scaling.get(target, (None, None))
        if centre is not None:
            value = (value - centre) / scale

        # Applicable where the relation is defined AND something in it moved.
        # Rewriting a row nothing touched is not a projection, it is a second
        # augmentation; and a ratio over a zero denominator is undefined, not
        # zero -- the same rule `residuals` already applies to the AE.
        applicable = np.ones(len(X), dtype=bool)
        if form == RATIO:
            applicable &= right != 0
        for index in (target, *operands):
            indicator = spec.indicators.get(index)
            if indicator is not None:
                applicable &= X[:, indicator] == 0.0
        if replaced is not None:
            # The TARGET counts too. Replacing `orig_rate` while leaving
            # `orig_bytes` and `duration` alone is precisely the inconsistency
            # Pi_G exists to repair; keying only on the operands skipped it.
            applicable &= replaced[:, [target, *operands]].any(axis=1)
        X[:, target] = np.where(applicable, value, X[:, target])
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
            # Alg. 2 line 1's "light classifier on D_s": reads RAW features,
            # not the latent, because it has to produce labels before the
            # encoder has learned anything to project into. Its own, smaller
            # architecture so it cannot memorise a D_s of a few thousand rows
            # and poison c_i for the whole first span.
            "bootstrap": CLASSIFIERS.get(
                self.setting("classifier"), input_dim=width,
                n_classes=self.space.n_classes,
                seed=int(self.setting("seed")),
                device=self.settings.get("device", "auto"),
                dropout=float(self.setting("bootstrap_dropout")),
                hidden=self.setting("bootstrap_hidden")),
        }

    @property
    def device(self) -> str:
        return getattr(self.parts["encoder"], "device", "cpu")

    # -------------------------------------------------------- augmentation

    def views(self, X, labels, marginals: ClassMarginals):
        """Two independently-augmented, constraint-projected views (Eq. 4-6)."""
        spec = self.vectoriser.columns()
        units = feature_units(spec)
        rate = float(self.setting("replace_rate"))
        width = X.shape[1]
        # Eq. 4's `d` is the number of FEATURES, not of matrix columns.
        k = max(1, int(np.ceil(rate * len(units))))

        picks = [view_indices(marginals.rng, len(units), k) for _ in range(len(X))]

        out = []
        for view in range(2):
            chosen = np.zeros((len(X), len(units)), dtype=bool)
            for row in range(len(X)):
                chosen[row, picks[row][view]] = True
            replaced = marginals.draw(labels, units, width)
            mask = np.zeros((len(X), width), dtype=bool)
            for u, unit in enumerate(units):
                mask[np.ix_(np.arange(len(X)), unit)] |= chosen[:, u, None]
            view = np.where(mask, replaced, X)
            out.append(project_constraints(view, spec, replaced=mask))
        return out

    # -------------------------------------------------------------- stages

    def pretrain(self, batches, report, supervised=None) -> tuple:
        """Contrastive pretraining with periodic pseudo-label refresh (Alg. 2).

        Training is split into spans of `refresh_every` epochs so lines 4-5 can
        run between them: freeze the encoder, fit a probe on D_s embeddings,
        reclassify. Refreshing stops once the label-change fraction falls below
        `refresh_tolerance` or `max_refresh_rounds` is reached; training does not.
        """
        torch = require_torch()
        encoder = self.parts["encoder"]
        heads = (self.parts["head_cont"], self.parts["head_cat"])
        lam = float(self.setting("lam"))
        taus = (float(self.setting("tau_cont")), float(self.setting("tau_cat")))
        decay = float(self.setting("weight_decay"))
        marginals = ClassMarginals(self.space.n_classes,
                                   self.vectoriser.columns().width,
                                   int(self.setting("reservoir")),
                                   int(self.setting("seed")))

        # Alg. 2 line 1. Without it `c_i` is the argmax of a randomly
        # initialised head, so Eq. 4 would draw replacements from arbitrary
        # buckets rather than from the class manifold -- which is the whole of
        # the TabCL principle it implements.
        if supervised is not None:
            self._bootstrap(*supervised)
            report.extra["bootstrapped"] = True

        def loss_fn(X_tensor):
            X = X_tensor.detach().to("cpu").numpy().astype(np.float64)
            labels = self.pseudo_labels(X)
            marginals.observe(X, labels)
            first, second = self.views(X, labels, marginals)
            z1 = encoder.module(as_tensor(first, self.device))
            z2 = encoder.module(as_tensor(second, self.device))
            total = None
            for head, tau, weight in zip(heads, taus, (lam, 1.0 - lam)):
                term = weight * nt_xent(head.module(z1), head.module(z2), tau, torch)
                total = term if total is None else total + term
            return total

        total_epochs = int(self.setting("epochs"))
        every = max(1, int(self.setting("refresh_every")))
        tolerance = float(self.setting("refresh_tolerance"))
        max_rounds = int(self.setting("max_refresh_rounds"))

        history: list[Epoch] = []
        churn: list[float] = []
        done, rounds = 0, 0
        # Churn is counted over D_l, which needs the keyed stream and a
        # connection to count on. Without either, refreshing still runs but the
        # stopping rule cannot fire and the round cap is the only bound.
        refreshing = supervised is not None
        while done < total_epochs:
            span = min(every, total_epochs - done)
            history += train([encoder, *heads], batches, loss_fn, epochs=span,
                             lr=float(self.setting("lr")), device=self.device,
                             weight_decay=decay)
            done += span
            if not (refreshing and done < total_epochs and rounds < max_rounds):
                continue
            self._refresh(*supervised)
            rounds += 1
            if keyed is not None and duck is not None:
                changed = self._reclassify(duck, keyed)
                churn.append(changed)
                if changed < tolerance:
                    refreshing = False
        report.extra["refresh_rounds"] = rounds
        report.extra["refresh_churn"] = churn
        return tuple(history)

    def _bootstrap(self, X, y) -> None:
        """Alg. 2 line 1: a light classifier on D_s, over raw features."""
        torch = require_torch()
        part = self.parts["bootstrap"]

        def loss_fn(features, target):
            return torch.nn.functional.cross_entropy(
                part.module(features), target, weight=self.class_weights(y))

        train([part],
              minibatches(int(self.setting("bootstrap_batch")), X, y,
                          seed=int(self.setting("seed"))),
              loss_fn, epochs=int(self.setting("bootstrap_epochs")),
              lr=float(self.setting("bootstrap_lr")), device=self.device,
              weight_decay=float(self.setting("weight_decay")))
        self._probe = "bootstrap"

    def _reclassify(self, duck, keyed) -> float:
        """Alg. 2 line 5: relabel all of D_l and count what changed.

        The fraction is over D_l, not D_s: the probe was just fitted on D_s, so
        churn there measures training fit and collapses toward zero whether or
        not the pseudo-labels have settled. Rows are matched by uid because a
        scan has no order guarantee, and the comparison is done in DuckDB so a
        54M-row pool never becomes a 54M-element Python object.
        """
        duck.sql("CREATE OR REPLACE TEMP TABLE probe_now (uid VARCHAR, y BIGINT)")
        for keys, X in keyed():
            if not len(X):
                continue
            labels = self.pseudo_labels(X)
            duck.con.executemany(
                "INSERT INTO probe_now VALUES (?, ?)",
                [(str(k), int(v)) for k, v in zip(keys, labels)])

        churn = 1.0
        if getattr(self, "_have_previous", False):
            total = duck.one("SELECT count(*) FROM probe_now")[0]
            if total:
                differing = duck.one(
                    "SELECT count(*) FROM probe_now n JOIN probe_was w "
                    "USING (uid) WHERE n.y <> w.y")[0]
                churn = differing / total
        duck.sql("CREATE OR REPLACE TEMP TABLE probe_was AS "
                 "SELECT * FROM probe_now")
        self._have_previous = True
        return churn

    def _refresh(self, X, y) -> None:
        """Alg. 2 lines 4-5: probe the frozen encoder on D_s embeddings."""
        torch = require_torch()
        encoder, classifier = self.parts["encoder"], self.parts["classifier"]

        def loss_fn(features, target):
            return torch.nn.functional.cross_entropy(
                classifier.module(encoder.module(features)), target,
                weight=self.class_weights(y))

        train([classifier],
              minibatches(self.batch, X, y, seed=int(self.setting("seed"))),
              loss_fn, epochs=int(self.setting("probe_epochs")),
              lr=float(self.setting("lr")), device=self.device,
              also=[encoder], weight_decay=float(self.setting("weight_decay")))
        self._probe = "encoder"

    def pseudo_labels(self, X):
        """The current probe's opinion, used only to condition the augmentation.

        Computed per batch rather than materialised: at lake scale a stored
        pseudo-label column would be tens of millions of rows rewritten every
        refresh, and the value is derived anyway.
        """
        if getattr(self, "_probe", None) == "bootstrap":
            return np.asarray(
                self.parts["bootstrap"].predict_proba(X)).argmax(axis=1)
        return np.asarray(self.predict_proba(X)).argmax(axis=1)

    def finetune(self, X, y, report) -> tuple:
        """Freeze the encoder, train the head, then both (Alg. 1 lines 8-9)."""
        torch = require_torch()
        weight = self.class_weights(y)

        def loss_fn(features, target):
            latent = self.parts["encoder"].module(features)
            return torch.nn.functional.cross_entropy(
                self.parts["classifier"].module(latent), target, weight=weight)

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
        # From here the classifier is trained, so it -- not the bootstrap --
        # is what `pseudo_labels` should ask.
        self._probe = "encoder"
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
