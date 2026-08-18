"""Regression test for the `_predict_into` batch-truncation bug.

Found running `ae` against real cic-ids-2017 data: `duck.con.executemany`
inside the same loop that streams `target` via `fetchmany` silently ended the
stream after one batch, so a 2,119,182-row table was written as 65,536 rows
with no error. Every existing behavioural fixture uses far fewer rows than one
batch, so this bug class was invisible to them; this test forces several
batches, including a short final one, and checks every row survives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from talos.common.duck import DuckEngine
from talos.data.feature.featureset import FeatureSet, Numeric
from talos.data.feature.vectorise import PassthroughVectoriser
from talos.data.labelling.behavioural.base import BehaviouralMethod, BehaviouralReport

ROWS = 2_500
BATCH = 1_000                # 3 batches: 1000, 1000, 500 -- exercises a short tail


@dataclass
class FakeSpace:
    classes: tuple[str, ...] = ("benign", "attack")

    def __contains__(self, name):
        return name in self.classes

    def index(self, name):
        return self.classes.index(name)


class StubMethod(BehaviouralMethod):
    """Only what `_predict_into` touches -- no torch, no checkpoints, no I/O."""

    def build_parts(self, width):
        return {}

    def pretrain(self, batches, report):
        return ()

    def finetune(self, X, y, report):
        return ()

    def predict_proba(self, X):
        # Deterministic, non-uniform: alternates the argmax by row parity, so
        # `per_class` isn't trivially satisfied by every row landing in one class.
        n = X.shape[0]
        proba = np.tile([0.3, 0.7], (n, 1))
        proba[::2] = [0.7, 0.3]
        return proba


def _method() -> StubMethod:
    method = object.__new__(StubMethod)
    method.settings = {}
    method.space = FakeSpace()
    method.batch = BATCH
    method.parts = {}
    return method


def test_predict_into_covers_every_row_across_several_batches():
    duck = DuckEngine()
    duck.sql(f"CREATE TABLE flows AS "
             f"SELECT 'u' || i AS uid, i::DOUBLE AS x FROM range({ROWS}) t(i)")
    target = duck.relation("SELECT * FROM flows")

    features = FeatureSet(name="test-v1", numeric=(Numeric(name="x", expr="x"),),
                          categorical=(), constraints=(), sha="000000000000",
                          path=__file__)
    method = _method()
    method.vectoriser = PassthroughVectoriser(features)

    report = BehaviouralReport(dataset="toy", feature_space="test", source="s",
                               method="stub")
    method._predict_into(duck, target, report)

    written = duck.one("SELECT count(*) FROM preds")[0]
    assert written == ROWS, (
        f"wrote {written} of {ROWS} rows -- the batch-truncation bug is back")
    assert sum(n for _, n in report.per_class) == ROWS
    assert report.mean_confidence is not None
