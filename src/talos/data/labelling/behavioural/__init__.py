"""Path B — behavioural labelling.

Path A needs a published attack schedule, so it can never touch a honeypot
capture, and on the CIC datasets its labels carry noise that is structured and
correlated with exactly the traffic of interest. Path B learns from the traffic
itself and then models the resulting label noise explicitly instead of trusting
it, following Eslami & Hamouda (arXiv:2509.23522).

Two pools, and keeping them apart is the whole foundation:

    D_l   large, unlabelled, what the SSL branches pretrain on
    D_s   small, hand-audited, what they fine-tune and select on

`pool.py` is why they cannot overlap. It lives here rather than at platform
level deliberately: these two pools are this method's vocabulary, and a training
split answers a different question (leave-one-domain-out folds, a validation
slice) that will get its own splitter.
"""

from talos.data.labelling.behavioural.pool import (
    PARTITIONS, Partition, PartitionLoader, Pool, PoolError,
)
from talos.data.labelling.behavioural.base import (
    BehaviouralMethod, BehaviouralReport, labelling_columns,
)
from talos.data.labelling.behavioural import autoencoder as _autoencoder  # noqa: F401
from talos.data.labelling.behavioural import tabcl as _tabcl              # noqa: F401

__all__ = [
    "PARTITIONS", "Partition", "PartitionLoader", "Pool", "PoolError",
    "BehaviouralMethod", "BehaviouralReport", "labelling_columns",
]
