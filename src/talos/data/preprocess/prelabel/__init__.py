"""Stage 1: everything judgeable before any label exists.

Membership rule, and the only one: a step belongs here if it would give the
same answer on a table with the label columns deleted. `verify` tests physical
invariants of a flow; `screen` tests variance, missingness and redundancy of a
column. Neither reads `label_class`, and `screen` excludes the label columns
from its own candidate set explicitly so that it cannot start.

Order within the stage matters once: `verify` runs before `screen`, because a
near-zero duration dividing a real byte count gives ~1e18 B/s -- a *finite*
double no NaN or Inf check catches -- and that one row would otherwise set the
standardisation scale for its whole column.
"""
