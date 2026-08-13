"""How a row becomes numbers, for whichever stage needs a matrix.

`featureset.py` declares — columns, vocabulary, transforms, constraints — and
`vectorise.py` applies. Shared by labelling and by the detector, because a model
served on a different encoding than it trained on is train/serve skew.
"""

from talos.data.feature.featureset import (
    DEFAULT_FEATURES, FEATURE_SETS, FeatureError, FeatureSet, FeatureSetLoader,
)
from talos.data.feature.vectorise import (
    VECTORISERS, ColumnSpec, Residual, Vectoriser, VectoriserError, residuals,
)

__all__ = [
    "DEFAULT_FEATURES", "FEATURE_SETS", "FeatureError", "FeatureSet",
    "FeatureSetLoader", "VECTORISERS", "ColumnSpec", "Residual", "Vectoriser",
    "VectoriserError", "residuals",
]
