"""Who is in D_l, who is in D_s, and how that is provable.

Path B needs two pools drawn from the same lake: a large unlabelled one to
pretrain on, and a small audited one to fine-tune and select on. The only thing
that must never happen is a flow appearing in both -- fine-tuning on rows the
encoder has already seen makes every number afterwards meaningless.

So membership is not sampled and recorded, it is COMPUTED:

    bucket(flow) = hash(seed || key) % buckets

Each pool owns a contiguous half-open range of buckets, and the ranges are
derived by one cumulative walk over the declared shares. Two pools therefore
cannot overlap -- not "were checked and did not", but cannot. At 137M rows that
matters twice over, because the alternative is materialising a uid list of
several gigabytes and trusting it stays in step with the lake.

Two consequences worth knowing.

**Stratification is free.** A hash is uniform within any subset, so a 1% share is
1% of every class, every capture and every dataset without anyone asking for it.
What is NOT free is a per-class FLOOR -- 1% of a 33-flow class is nothing -- and
that belongs to whoever builds D_s for auditing, not here.

**`partition_by` is a real choice.** Keying on `uid` gives fine-grained,
proportional pools. Keying on `capture` sends a whole capture to one pool, which
is what leakage-safe evaluation needs (W5.9), at the cost of balance: captures
differ in size by orders of magnitude.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from talos.common.plugins import ConfigPlugins, Declaration
from talos.common.provenance import Sha12

POOLS_DIR = Path(__file__).with_name("pools")

#: Resolution of the bucket space. A million gives a share of 0.0001 (~14k flows
#: on the full lake) a hundred buckets to sit in, so small pools are still
#: expressible without the rounding landing on zero.
DEFAULT_BUCKETS = 1_000_000

#: What a flow can be keyed on. `uid` is per-flow; `capture` keeps a capture
#: whole, which is the leakage-safe unit.
BY_UID = "uid"
BY_CAPTURE = "capture"
PARTITION_KEYS = (BY_UID, BY_CAPTURE)

#: `label_quality` values a pool may exclude. This is where the label space
#: finally bites: gate 9 flags an out-of-space flow, and it is dropped HERE.
QUALITY_EXCLUSIONS = ("out-of-space", "uncertain")

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Shares are floats and must total one. The tolerance absorbs the usual decimal
# representation error (0.85 + 0.14 + 0.01 != 1.0 exactly in binary) without
# admitting a declaration that is actually wrong.
_SHARE_TOLERANCE = 1e-9


class PoolError(Exception):
    pass


#: The pool plug-in point. A user declares their own split of their own lake.
PARTITIONS = ConfigPlugins("partition", built_in=POOLS_DIR, error=PoolError)


def _quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@dataclass(frozen=True)
class Pool:
    """One named slice of the bucket space."""

    name: str
    share: float
    lo: int                    # inclusive
    hi: int                    # exclusive
    buckets: int
    key: str
    seed: str
    role: str = ""
    description: str = ""

    @property
    def n_buckets(self) -> int:
        return self.hi - self.lo

    def expected_flows(self, total: int) -> int:
        """Roughly how many rows this pool holds, given the source's size."""
        return round(total * self.n_buckets / self.buckets)

    # ------------------------------------------------------------------- SQL

    def bucket_sql(self, alias: str = "") -> str:
        """The bucket expression. Seeded, so a new seed reshuffles every pool.

        The seed is PREPENDED to the key rather than added to the hash: adding
        would shift every flow by the same amount and leave the ordering intact,
        so two seeds would produce pools that are rotations of each other rather
        than independent draws.
        """
        column = f"{alias}.{self.key}" if alias else self.key
        return f"(hash({_quote(self.seed)} || CAST({column} AS VARCHAR)) % {self.buckets})"

    def predicate(self, alias: str = "") -> str:
        """Membership alone, without the partition's exclusions."""
        bucket = self.bucket_sql(alias)
        return f"({bucket} >= {self.lo} AND {bucket} < {self.hi})"

    def describe(self) -> str:
        pct = 100.0 * self.n_buckets / self.buckets
        role = f"  {self.role}" if self.role else ""
        return (f"{self.name:<10} {pct:>8.4f}%  buckets [{self.lo}, {self.hi})"
                f"{role}")


@dataclass(frozen=True)
class Partition:
    """A complete allocation of one source into disjoint pools."""

    name: str
    pools: Mapping[str, Pool]
    datasets: tuple[str, ...]
    labels: str
    exclude: tuple[str, ...]
    #: Drop rows no real flow could produce, per docs/paper-deviations.md. One
    #: rule for every dataset -- evidence is per-dataset, the predicate is not.
    exclude_invalid: bool
    key: str
    seed: str
    buckets: int
    sha: Sha12
    path: Path
    doc: Mapping[str, Any]

    def __getitem__(self, name: str) -> Pool:
        try:
            return self.pools[name]
        except KeyError:
            raise PoolError(
                f"partition {self.name!r} has no pool {name!r}. "
                f"Have: {', '.join(self.pools)}") from None

    def __iter__(self) -> Iterator[Pool]:
        return iter(self.pools.values())

    def __contains__(self, name: object) -> bool:
        return name in self.pools

    # ------------------------------------------------------------------- SQL

    def exclusion_sql(self, alias: str = "") -> str:
        """Rows this partition refuses to draw from, whatever pool they land in.

        `out-of-space` is the one that matters: gate 9 flags those flows and
        leaves them in the labelled table, and this is the point at which they
        stop being training data. Doing it here rather than at labelling time is
        what keeps the labelled table the whole dataset.
        """
        if not self.exclude:
            return ""
        column = f"{alias}.label_quality" if alias else "label_quality"
        listed = ", ".join(_quote(value) for value in self.exclude)
        # NULL-safe: a method that emits no quality column value must not be
        # silently dropped by a NOT IN against NULL.
        return f"({column} IS NULL OR {column} NOT IN ({listed}))"

    def where(self, pool: str, alias: str = "") -> str:
        """Full membership test for one pool: in its range, and not excluded."""
        tests = [self[pool].predicate(alias)]
        exclusion = self.exclusion_sql(alias)
        if exclusion:
            tests.append(exclusion)
        return " AND ".join(tests)

    def sources(self, lake, feature_space: str, cfg=None) -> list[str]:
        """The labelled tables this partition draws from, one glob per dataset."""
        return [
            lake.uri("labelled", dataset=dataset, feature_space=feature_space,
                     method=self.labels,
                     source=cfg.source_of(dataset) if cfg else None,
                     rel="conn.parquet")
            for dataset in self.datasets
        ]

    def relation(self, duck, pool: str, sources: Sequence[str], alias: str = "p"):
        """The rows of one pool, as a lazy relation."""
        listed = ", ".join(_quote(uri) for uri in sources)
        table = f"read_parquet([{listed}], union_by_name = true)"
        tests = [self.where(pool, alias)]
        if self.exclude_invalid:
            tests.append(self.validity_sql(duck, table, alias))
        return duck.relation(
            f"SELECT * FROM {table} {alias} WHERE {' AND '.join(tests)}")

    def validity_sql(self, duck, table: str, alias: str = "") -> str:
        """The declared row invariants, as a filter over this partition's tables.

        Read off the table so an invariant whose column is absent is skipped
        rather than rejecting every row -- the same rule the auditor applies.
        """
        from talos.data.preprocess.validity import valid_sql

        columns = {row[0]: row[1]
                   for row in duck.sql(f"DESCRIBE SELECT * FROM {table}")}
        return valid_sql(columns, alias=alias)

    # -------------------------------------------------------------- reporting

    def describe(self) -> str:
        lines = [
            f"partition     {self.name}  sha {self.sha}",
            f"  source      {', '.join(self.datasets)}  labelled by {self.labels}",
            f"  key         {self.key}   seed {self.seed}   "
            f"buckets {self.buckets:,}",
        ]
        if self.exclude:
            lines.append(f"  excluding   {', '.join(self.exclude)}")
        lines += [f"  {pool.describe()}" for pool in self]
        return "\n".join(lines)


class PartitionLoader:
    """Reads a partition declaration and refuses the ways it can be wrong."""

    def __init__(self, point: ConfigPlugins = PARTITIONS):
        self.point = point

    def load(self, name: str | Path) -> Partition:
        declaration = self.point.get(name)
        doc = declaration.doc
        declared = doc.get("pools") or {}
        buckets = int(doc.get("buckets", DEFAULT_BUCKETS))
        key = doc.get("partition_by", BY_UID)
        seed = str(doc.get("seed", ""))
        exclude = tuple(doc.get("exclude") or ())

        self._validate(declaration, declared, buckets, key, seed, exclude)
        pools = self._allocate(declared, buckets, key, seed)

        source = doc.get("source") or {}
        return Partition(
            name=doc.get("name", declaration.name), pools=pools,
            datasets=tuple(source.get("datasets") or ()),
            labels=source.get("labels", "schedule"),
            exclude=exclude,
            exclude_invalid=bool(doc.get("exclude_invalid", True)),
            key=key, seed=seed, buckets=buckets,
            sha=declaration.sha, path=declaration.path, doc=doc)

    # ------------------------------------------------------------ allocation

    @staticmethod
    def _allocate(declared: Mapping[str, Any], buckets: int, key: str,
                  seed: str) -> dict[str, Pool]:
        """Cumulative edges, so ranges are disjoint and exhaustive by construction.

        Boundaries are rounded from the RUNNING total rather than per pool. Doing
        it per pool lets rounding errors accumulate and leaves a gap or an overlap
        of a few buckets between neighbours -- which is exactly the kind of
        almost-right that no test would notice and that would put a handful of
        flows in two pools.
        """
        pools: dict[str, Pool] = {}
        cumulative = 0.0
        edge = 0
        for name, entry in declared.items():
            entry = entry or {}
            cumulative += float(entry.get("share", 0.0))
            nxt = round(cumulative * buckets)
            pools[name] = Pool(
                name=name, share=float(entry.get("share", 0.0)),
                lo=edge, hi=nxt, buckets=buckets, key=key, seed=seed,
                role=entry.get("role", ""),
                description=entry.get("description", ""))
            edge = nxt
        return pools

    # ------------------------------------------------------------ validation

    @staticmethod
    def _validate(declaration: Declaration, declared: Mapping[str, Any],
                  buckets: int, key: str, seed: str,
                  exclude: tuple[str, ...]) -> None:
        errs = []
        if not declared:
            errs.append("declares no pools, so it allocates nothing")
        if key not in PARTITION_KEYS:
            errs.append(f"partition_by {key!r} is not one of "
                        f"{', '.join(PARTITION_KEYS)}")
        if not seed:
            errs.append("no seed. Membership must be reproducible from the "
                        "declaration alone, and an unseeded hash makes the split "
                        "a property of the DuckDB build instead")
        if buckets < 100:
            errs.append(f"buckets {buckets} is too coarse to express a small pool; "
                        f"a share below 1/{buckets} rounds to nothing")

        for value in exclude:
            if value not in QUALITY_EXCLUSIONS:
                errs.append(
                    f"exclude {value!r} is not a label_quality value; known: "
                    f"{', '.join(QUALITY_EXCLUSIONS)}. A typo would silently "
                    f"exclude nothing")

        total = 0.0
        for name, entry in declared.items():
            if not _NAME.match(name):
                errs.append(f"pool name {name!r} must match [a-z0-9][a-z0-9_-]*")
            share = (entry or {}).get("share")
            if not isinstance(share, (int, float)) or isinstance(share, bool):
                errs.append(f"{name}: share must be a number between 0 and 1")
                continue
            if not 0 <= share <= 1:
                errs.append(f"{name}: share {share} is not between 0 and 1. "
                            f"Shares are fractions, not percentages")
            total += float(share)

        if declared and abs(total - 1.0) > _SHARE_TOLERANCE:
            errs.append(
                f"shares total {total:g}, not 1. Every flow has to land in exactly "
                f"one pool: a total below 1 silently discards the remainder, and "
                f"above 1 is unrepresentable, so neither is treated as a rounding "
                f"question")

        if errs:
            raise PoolError(f"{declaration.path}:\n  " + "\n  ".join(errs))
