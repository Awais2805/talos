"""Stage 2: the half that was blocked on a label, run before the ML pipeline.

Membership rule: a step belongs here if it CANNOT be computed without a label,
and still has to be settled before a model trains. `relevance` is both --
mutual information against `label_class` needs the label by definition, and a
feature's fate (keep / drop / hold) has to be on record before the training
schema is locked.

It reads the annotated declaration `prelabel/screen` emitted, not the raw one,
and adds its own verdict beside rather than over: `relevance:` wins where the
two disagree, because it runs later and knows strictly more (`Numeric.verdict`
in `featureset.py`). `hold` is not a failure -- high domain AUC *and* high task
relevance means only measured leave-one-domain-out performance can decide, and
that is the ML pipeline's job, not this stage's.

The domain probe needs at least two domains to discriminate between; with one
it reports `unmeasured` rather than a degenerate 0.5. That is why this stage is
per-POOL, where `prelabel` is per-dataset.

Which labelling method produced the labels it reads is part of the question,
not an implementation detail: relevance measured against schedule labels and
relevance measured against fused labels are different measurements, and the
`labelled` zone keys on `{method}` so they cannot be confused.
"""
