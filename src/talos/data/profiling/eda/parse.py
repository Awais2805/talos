"""Unpack the aggregate rows, and say what the profile says, on the console.

The parse is positional: `build_stats_query` packs its aggregates into LIST
literals in spec order, and this walks the same order back out. That coupling
is deliberate (it is what keeps the result a couple of dozen columns instead of
a thousand) and it is why the two modules name each other in their docstrings.
"""

from __future__ import annotations

from talos.data.profiling.eda.spec import Spec


def parse_row(row, spec, num, cat, ident, nulls):
    """Unpack one group's aggregate row using the spec order it was built from."""
    it = iter(row)
    grp, n = next(it), int(next(it))
    numeric = {}
    for f in num:
        m, h, q = next(it), next(it), next(it)
        numeric[f.name] = {
            "n_undefined": int(m[0] or 0), "n_zero": int(m[1] or 0),
            "min": m[2], "max": m[3], "sum_t": m[4] or 0.0, "sumsq_t": m[5] or 0.0,
            "hist": [int(x) for x in h],
            "q": dict(zip((str(x) for x in spec.quantiles), list(q or []))),
        }
    xy = [float(v) for v in (next(it) or [])]

    # Unpack the packed 2-D keys back into sparse bins x bins surfaces, in the
    # same pair order as pairs_xy.
    joints = []
    for _ in Spec.pairs(num):
        joints.append({f"{int(k) // 1000},{int(k) % 1000}": int(v)
                       for k, v in (next(it) or {}).items()})

    categorical = {}
    for f in cat:
        counts = {str(k): int(v) for k, v in (next(it) or {}).items()}
        top = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:spec.top_k])
        categorical[f.name] = {
            "counts": top,
            "n_other": sum(counts.values()) - sum(top.values()),
            "n_distinct": len(counts),
        }
    distinct = ({c: int(v) for c, v in zip(ident, next(it))} if ident else {})
    nullc = ({c: int(v) for c, v in zip(nulls, next(it))} if nulls else {})
    return grp, {"n": n, "numeric": numeric, "pairs_xy": xy, "pairs_joint": joints,
                 "categorical": categorical, "distinct": distinct, "nulls": nullc}


def summarise(profile):
    """Console summary -- the same numbers the HTML leads with."""
    meta, by = profile["meta"], profile["by_class"]
    total = meta["rows"]
    if not total:
        print("\nno rows matched -- nothing to summarise")
        return
    print(f"\n{'class':<16}{'flows':>16}{'share':>9}")
    for cls, blk in sorted(by.items(), key=lambda kv: -kv[1]["n"]):
        print(f"{cls:<16}{blk['n']:>16,}{100 * blk['n'] / total:>8.3f}%")
    print(f"{'-' * 41}\n{'TOTAL':<16}{total:>16,}")
    atk = total - by.get(profile["benign_class"], {}).get("n", 0)
    print(f"\nattack ratio {100 * atk / total:.3f}%  "
          f"({len(by)} classes, {meta['numeric_features']} numeric features, "
          f"{meta['categorical_features']} categorical)")
    if meta["skipped_features"]:
        print("\nnot measurable on this source (missing columns):")
        for name, cols in meta["skipped_features"].items():
            print(f"  {name:<22} needs {', '.join(cols)}")
