"""Preprocessing: the flow representation, and what can be judged without labels.

Split by what a step needs, not by when it happens to run. `screen` needs no
label and so runs before labelling; relevance scoring needs one and runs after.
Row validity flags rows, column screening drops columns -- different verbs.
"""
