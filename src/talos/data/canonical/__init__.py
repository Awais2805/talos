"""The canonical zone: one schema every domain is mapped onto.

`registry/` names and types the canonical features; `mapping/` records how each
dataset's columns reach them. Runs AFTER labelling — the feature selection it
serves scores relevance against the label, so it cannot run before there is one.
"""
