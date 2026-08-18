"""Out-of-sample probabilities for the noise estimate (Eq. 18-19).

F-fold cross-validation over the fused pool: each row is scored by a model that
never saw it. In-sample probabilities would make every self-confidence, every
threshold and every MAD optimistic, and the whole noise estimate with them.
"""

from __future__ import annotations

import numpy as np

from talos.data.feature.featureset import DEFAULT_FEATURES, FeatureSetLoader
from talos.data.feature.vectorise import build as build_vectoriser
from talos.data.labelling.base import LabellingError
from talos.data.labelling.behavioural.training import minibatches, train
from talos.parts.base import CLASSIFIERS
from talos.parts.mlp import require_torch, torch_available

#: Paper's CV classifier, Table VII (Confident Learning column).
DEFAULTS = {"folds": 5, "hidden": [128, 64], "dropout": 0.20, "lr": 1e-3,
            "batch": 64, "epochs": 30, "classifier": "mlp", "seed": 0}


class OutOfSampleProbe:
    """Trains F models, each scoring the fold it was not trained on."""

    def __init__(self, method, space, settings: dict):
        self.method = method
        self.space = space
        self.settings = {**DEFAULTS, **(settings or {})}
        self.features = FeatureSetLoader().load(
            self.settings.get("features", DEFAULT_FEATURES))
        self.vectoriser = build_vectoriser(
            self.settings.get("vectoriser", "passthrough"), self.features)

    def run(self, duck, report):
        """`(keys, probabilities, labels)`, all from the one scan that built them."""
        if not torch_available():
            raise LabellingError(
                "confident learning needs out-of-sample probabilities, which "
                "need a model to produce. PyTorch is not installed.")
        torch = require_torch()

        rel = duck.relation("SELECT * FROM fused ORDER BY uid")
        keys, X, raw = self.vectoriser.labelled_matrix(rel, "label_class")
        labels = np.array([self.space.index(str(c)) for c in raw], dtype=np.int64)

        folds = int(self.settings["folds"])
        if len(labels) < folds:
            raise LabellingError(
                f"{len(labels)} rows cannot be split into {folds} folds. Lower "
                f"`folds`, or fuse a larger pool.")

        assignment = self._folds(len(labels), folds)
        probabilities = np.zeros((len(labels), self.space.n_classes))
        for fold in range(folds):
            held = assignment == fold
            classifier = CLASSIFIERS.get(
                self.settings["classifier"], input_dim=X.shape[1],
                n_classes=self.space.n_classes, hidden=self.settings["hidden"],
                dropout=float(self.settings["dropout"]),
                seed=int(self.settings["seed"]) + fold, device="cpu")
            self._fit(classifier, X[~held], labels[~held], torch)
            probabilities[held] = classifier.predict_proba(X[held])

        report.extra["oos_folds"] = folds
        return keys, probabilities, labels

    def _folds(self, rows: int, folds: int):
        """Deterministic fold assignment, seeded from the declaration."""
        rng = np.random.default_rng(int(self.settings["seed"]))
        return rng.permutation(np.arange(rows) % folds)

    def _fit(self, classifier, X, y, torch) -> None:
        def loss_fn(features, target):
            return torch.nn.functional.cross_entropy(
                classifier.module(features), target)

        train([classifier],
              minibatches(int(self.settings["batch"]), X, y,
                          seed=int(self.settings["seed"])),
              loss_fn, epochs=int(self.settings["epochs"]),
              lr=float(self.settings["lr"]), device="cpu")
