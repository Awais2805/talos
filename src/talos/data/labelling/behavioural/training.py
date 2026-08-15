"""The torch training loop, shared by every behavioural branch.

Generic on purpose: it knows about parts, batches and a loss, and nothing about
what is being learned. The autoencoder's reconstruction objective and the
contrastive branch's NT-Xent are both just a `loss_fn` here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np

from talos.parts.mlp import require_torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Epoch:
    """One pass, for the run report."""

    index: int
    loss: float
    rows: int

    def describe(self) -> str:
        return f"  epoch {self.index:>3}  loss {self.loss:.6f}  rows {self.rows:,}"


def as_tensor(X, device: str):
    """numpy -> tensor. float32 for features; integers stay integers for targets."""
    torch = require_torch()
    X = np.asarray(X)
    dtype = np.int64 if np.issubdtype(X.dtype, np.integer) else np.float32
    return torch.as_tensor(X.astype(dtype), device=device)


def train(parts: Sequence, batches: Callable[[], Iterator], loss_fn: Callable,
          *, epochs: int = 5, lr: float = 1e-3, device: str = "cpu",
          also: Sequence = ()) -> list[Epoch]:
    """Adam over `parts`, `loss_fn(*batch_tensors) -> scalar`.

    `batches` is a callable, not an iterator: a DuckDB stream is consumed once,
    so each epoch has to ask for a fresh one. `also` runs in train mode without
    being optimised — that is what freezing an encoder means.
    """
    torch = require_torch()
    modules = [part.module for part in parts if getattr(part, "module", None)]
    parameters = [p for module in modules for p in module.parameters()]
    if not parameters:
        raise ValueError("nothing to train: no part exposed any parameters")

    optimiser = torch.optim.Adam(parameters, lr=lr)
    frozen = [part.module for part in also if getattr(part, "module", None)]
    history: list[Epoch] = []
    for index in range(1, epochs + 1):
        for module in modules:
            module.train()
        for module in frozen:
            module.eval()
        total, rows = 0.0, 0
        for batch in batches():
            arrays = batch if isinstance(batch, tuple) else (batch,)
            if not len(arrays[0]):
                continue
            optimiser.zero_grad()
            loss = loss_fn(*(as_tensor(a, device) for a in arrays))
            loss.backward()
            optimiser.step()
            total += float(loss) * len(arrays[0])
            rows += len(arrays[0])
        epoch = Epoch(index, total / max(rows, 1), rows)
        history.append(epoch)
        logger.info(epoch.describe())
    for module in modules:
        module.eval()
    return history


def minibatches(size: int, *arrays, shuffle: bool = True, seed: int = 0):
    """A callable yielding aligned minibatches of in-memory arrays."""
    def go():
        order = np.arange(len(arrays[0]))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, len(order), size):
            index = order[start:start + size]
            yield tuple(a[index] for a in arrays)
    return go


def cross_entropy_blocks(X, X_hat, blocks, torch) -> object:
    """Cross-entropy per one-hot block, summed.

    Squared error on a one-hot block trains the decoder to emit 0.4 of a
    category; a block is a distribution and is scored as one.
    """
    total = None
    for block in blocks:
        target = X[:, block.start:block.stop].argmax(dim=1)
        logits = X_hat[:, block.start:block.stop]
        term = torch.nn.functional.cross_entropy(logits, target)
        total = term if total is None else total + term
    return total if total is not None else torch.zeros((), device=X.device)
