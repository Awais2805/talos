"""The SQL one profiling pass runs: every heavy statistic, one GROUP BY.

Separated from the stage that runs it because generating this SQL and deciding
what to profile are different jobs with different failure modes -- a wrong
aggregate is a silent number, a wrong source is a silent population. Keeping
them in one file made the second invisible behind the first for a long time.
"""

from __future__ import annotations

from talos.data.profiling.eda.spec import Spec, _q


def scan_clause(src, sample, limit, valid=""):
    """The source expression both queries read from.

    LIMIT is pushed into the scan, so a smoke test stops after N rows instead of
    reading the whole zone -- the difference between validating the pipeline in
    five seconds and finding out twenty minutes into 2019 that a column is not
    the type the spec assumed. SAMPLE still reads everything; it trades
    aggregation work for statistical validity, which is the opposite trade.

    `valid` filters to the rows the training pools draw from, so a distribution
    the report shows is a distribution of what a model actually sees.

    `src` is a LIST of globs, because one profile may span several capture
    files; `union_by_name` because a capture need not carry every column.
    """
    listed = ", ".join(f"'{g}'" for g in ([src] if isinstance(src, str) else src))
    sql = f"SELECT * FROM read_parquet([{listed}], union_by_name = true)"
    if valid:
        sql += f" WHERE {valid}"
    if sample:
        sql += f" USING SAMPLE {sample} PERCENT (bernoulli)"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return sql


def build_stats_query(src, spec, num, cat, ident, nulls, group_expr, sample,
                      limit, valid=""):
    """One pass, one GROUP BY, every heavy statistic.

    The `f` CTE projects each feature exactly once (raw value + transformed
    value); the aggregate then references those aliases, so a 16-feature spec
    costs 16 expression evaluations per row rather than one per bin. Aggregates
    are packed into LIST literals -- `[sum(a), sum(b)]` -- which keeps the
    result to a couple of dozen columns instead of a thousand, and keeps the
    parse order tied to the spec order rather than to generated column names.
    """
    proj = ["%s AS grp" % group_expr]
    proj += [f"{f.raw} AS r{i}" for i, f in enumerate(num)]
    proj += [f"{f.trans} AS t{i}" for i, f in enumerate(num)]
    proj += [f"{f.bin_index(f'r{i}')} AS g{i}" for i, f in enumerate(num)]
    proj += [f"{f.expr} AS k{j}" for j, f in enumerate(cat)]
    proj += [f"{_q(c)} AS id{k}" for k, c in enumerate(ident)]
    proj += [f"{_q(c)} AS nz{k}" for k, c in enumerate(nulls)]

    qs = ", ".join(str(q) for q in spec.quantiles)

    agg = ["grp", "count(*) AS n"]
    for i, f in enumerate(num):
        agg.append(
            f"[count(*) FILTER (r{i} IS NULL)::DOUBLE, count(*) FILTER (r{i} = 0)::DOUBLE, "
            f"min(r{i})::DOUBLE, max(r{i})::DOUBLE, sum(t{i})::DOUBLE, "
            f"sum(t{i} * t{i})::DOUBLE] AS m{i}")
        agg.append("[" + ", ".join(f"({c})::DOUBLE" for c in f.bin_counts(f"r{i}")) + f"] AS h{i}")
        agg.append(f"approx_quantile(r{i}::DOUBLE, [{qs}]) AS q{i}")
    pairs = Spec.pairs(num)
    agg.append("[" + ", ".join(f"sum(t{a} * t{b})::DOUBLE" for a, b in pairs) + "] AS xy"
               if pairs else "[]::DOUBLE[] AS xy")
    # The joint bin histogram of EVERY pair, as one MAP per pair: both bin
    # indices packed into a single integer key. This is what makes a rank
    # correlation possible at all downstream -- true ranks depend on the whole
    # dataset and could never be merged across datasets, but joint bin counts
    # are plain counts, so they add exactly like everything else here. It also
    # means any pair can be drawn as a 2-D density surface, not just a
    # pre-chosen few. Sparse in practice: most cells of most pairs are empty.
    agg += [f"histogram(g{a} * 1000 + g{b}) AS j{a}_{b}" for a, b in pairs]
    agg += [f"histogram(k{j}) AS c{j}" for j in range(len(cat))]
    if ident:
        # Identity columns are counted, never valued: 2019's spoofed sources
        # alone would put millions of addresses in a histogram, and an IP is a
        # fact about the testbed, not a feature.
        agg.append("[" + ", ".join(f"approx_count_distinct(id{k})::DOUBLE"
                                   for k in range(len(ident))) + "] AS dst")
    if nulls:
        agg.append("[" + ", ".join(f"count(*) FILTER (nz{k} IS NULL)::DOUBLE"
                                   for k in range(len(nulls))) + "] AS nul")

    return (f"WITH src AS (\n  {scan_clause(src, sample, limit, valid)}\n"
            f"), f AS (\n  SELECT " + ",\n         ".join(proj) + "\n  FROM src\n)\n"
            "SELECT " + ",\n       ".join(agg) + "\nFROM f GROUP BY grp ORDER BY n DESC")


def build_capture_query(src, group_expr, sample, limit, valid=""):
    """Domain sub-structure: flows and time span per capture x class.

    Cheap next to the main pass -- parquet column pruning means it reads three
    columns, not twenty -- and it is what turns "this dataset" into "these ten
    capture days", which is the unit domain shift actually happens in.
    """
    return (f"WITH src AS (\n"
            f"  SELECT capture, ts, {group_expr} AS grp "
            f"FROM ({scan_clause(src, sample, limit, valid)})\n)\n"
            f"SELECT capture, grp, count(*)::DOUBLE AS n, "
            f"min(ts)::DOUBLE AS t0, max(ts)::DOUBLE AS t1 "
            f"FROM src GROUP BY 1, 2 ORDER BY 3 DESC")
