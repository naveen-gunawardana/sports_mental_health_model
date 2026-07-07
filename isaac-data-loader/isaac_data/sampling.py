"""Stratified, efficient sampling helpers.

Mirrors the ISAAC web app's behavior: a total sample budget is spread **equally
across the selected months** (capped by each month's size), and within a month
rows are drawn uniformly at random without replacement. For parquet, only the
row groups that contain sampled rows are read (over HTTP range requests, with
column projection) — so small samples don't download whole files.
"""
from __future__ import annotations

from typing import List, Optional, Sequence


def equal_quotas(caps: List[int], n: int) -> List[int]:
    """Distribute a total of ``n`` across buckets ``caps`` as evenly as possible.

    Equal-per-bucket "water-filling": each round gives every not-yet-full bucket
    an equal share, capped at its capacity, redistributing overflow — matching the
    web app's ``compute_per_file_quotas``. Returns per-bucket counts summing to
    ``min(n, sum(caps))``.
    """
    quotas = [0] * len(caps)
    active = [i for i, c in enumerate(caps) if c > 0]
    remaining = max(0, int(n))

    while active and remaining > 0:
        base = max(1, remaining // len(active))
        nxt = []
        for idx in active:
            left = caps[idx] - quotas[idx]
            if left <= 0:
                continue
            take = min(base, left, remaining)
            quotas[idx] += take
            remaining -= take
            if quotas[idx] < caps[idx]:
                nxt.append(idx)
            if remaining <= 0:
                break
        active = nxt

    # Hand out any rounding leftover to the buckets with the most spare capacity.
    if remaining > 0:
        for idx in sorted(range(len(caps)), key=lambda i: caps[i] - quotas[i], reverse=True):
            if remaining <= 0:
                break
            left = caps[idx] - quotas[idx]
            if left > 0:
                take = min(left, remaining)
                quotas[idx] += take
                remaining -= take
    return quotas


def sample_parquet(source, num_rows: int, k: int, columns: Optional[Sequence[str]], rng):
    """Uniformly sample ``k`` rows from a parquet file into a pandas DataFrame.

    Reads only the row groups containing sampled positions (plus column
    projection), so transfer is bounded by the selected columns of the hit row
    groups — never more than a full column-projected read, and much less when
    ``k`` is small.

    Args:
        source: local path or an open binary file object (e.g. an fsspec HTTP file).
        num_rows: total rows in the file (from the catalog).
        k: number of rows to sample (clamped to num_rows).
        columns: columns to read (None = all).
        rng: a numpy Generator (provides reproducibility).
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(source)
    md = pf.metadata
    sizes = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
    cols = list(columns) if columns else None

    k = min(int(k), int(num_rows))
    if k <= 0:
        return pa.table({c: pa.array([], type=pa.null()) for c in (cols or [])}).to_pandas()
    positions = np.sort(rng.choice(num_rows, size=k, replace=False).astype(np.int64))

    tables = []
    rg_start = 0
    p = 0
    for i, size in enumerate(sizes):
        rg_end = rg_start + size
        j = p
        while j < positions.size and positions[j] < rg_end:
            j += 1
        if j > p:
            local = positions[p:j] - rg_start
            table = pf.read_row_group(i, columns=cols)
            tables.append(table.take(pa.array(local)))
            p = j
        rg_start = rg_end
        if p >= positions.size:
            break

    if not tables:
        import pandas as pd
        return pd.DataFrame(columns=cols or [])
    return pa.concat_tables(tables).to_pandas()
