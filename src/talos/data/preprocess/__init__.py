"""Preprocessing, in the two stages the pipeline actually has.

Labelling sits in the middle of preprocessing, not before or after it, because
half of what has to be decided about a feature can be decided without a label
and half cannot:

    parquet -> [prelabel] -> labelling -> [postlabel] -> ML pipeline

`prelabel` is everything judgeable with no label in existence: rows that cannot
be true of a real flow (`verify`), and columns that carry nothing a model could
use -- zero variance, mostly-null, redundant with a neighbour (`screen`). It
also measures the standardisation constants, because those must be taken over
the rows a model will train on and that is knowable early.

`postlabel` is the half that was BLOCKED on a label and still has to finish
before any model trains: how relevant a feature is to the task (mutual
information against the label) and whether it merely encodes which domain the
traffic came from (`relevance`). Running it earlier is not a scheduling
preference, it is impossible.

Two things deliberately do NOT live here:

`stats` is shared measurement, used by both stages and owned by neither.

EDA is a SERVICE both stages call, not a member of either -- it profiles for
exploration and leakage detection rather than for feature selection, and it
runs twice (once per stage) over the same ruler. It lives in
`talos.data.profiling.eda` and depends on the feature set one way only; no
stage here imports it and it decides nothing about a feature's fate.

Every stage here MARKS, and nothing here deletes. A verdict is an annotation on
the declaration (`screen:`, `relevance:`); `FeatureSet.active()` is the only
thing that filters, and it is called by a consumer that has decided its own
policy. Row validity is the same shape: flagged, never dropped, because row
selection is experiment-scoped.
"""
