"""Memory-bounded dataset loading.

NF-UNSW-NB15-v2 is 2.4M rows and loads whole without trouble. NF-ToN-IoT-v2 is
16.9M and needs roughly 6 GB in pandas, which a 16 GB machine cannot spare
alongside the transformed feature arrays and a model.

Rows are capped by reading a subset of Parquet **row groups** rather than by
sampling rows. That matters for a graph project: a row group is a contiguous
block of the original file, so the traffic inside it keeps its local structure.
Random row sampling would thin every host's neighbourhood -- the same objection
D20 raises against downsampling attack flows, where a scanner reaching 5,161
peers would appear to reach 155.

Selection is **family-aware**, because a uniform stride is not safe here.
NF-ToN-IoT-v2 is ordered by capture scenario, so each attack family occupies a
narrow band of the file rather than being spread through it. A first attempt
taking every k-th group gave `backdoor` zero rows out of 16,809 and `ransomware`
33 out of 3,425 -- which would have broken leave-one-attack-out before it
started, since a held-out family has to exist to be held out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _family_coverage(pf: pq.ParquetFile, column: str) -> list[dict[str, int]]:
    """Per row group, how many rows of each family it holds.

    Reads one column per group, so this is cheap even on a 17M-row file.
    """
    out = []
    for i in range(pf.metadata.num_row_groups):
        col = pf.read_row_group(i, columns=[column]).column(0).to_numpy(
            zero_copy_only=False)
        vals, counts = np.unique(col.astype(str), return_counts=True)
        out.append({str(v): int(c) for v, c in zip(vals, counts)})
    return out


def _select_groups(
    coverage: list[dict[str, int]], rows_per_group: float, max_rows: int
) -> list[int]:
    """Choose row groups: guarantee every family, then spread across the file.

    Rare families are claimed first, since they have the fewest groups that
    could satisfy them and are the ones a uniform stride misses.
    """
    budget = max(1, int(max_rows / max(rows_per_group, 1)))

    totals: dict[str, int] = {}
    for g in coverage:
        for fam, n in g.items():
            totals[fam] = totals.get(fam, 0) + n

    chosen: set[int] = set()
    for fam in sorted(totals, key=lambda f: totals[f]):
        if any(coverage[i].get(fam, 0) > 0 for i in chosen):
            continue
        best = max(range(len(coverage)), key=lambda i: coverage[i].get(fam, 0))
        if coverage[best].get(fam, 0) > 0:
            chosen.add(best)

    # Fill the remaining budget with an even spread, so the sample still covers
    # every phase of the capture rather than clustering on the rare families.
    remaining = [i for i in range(len(coverage)) if i not in chosen]
    if remaining and len(chosen) < budget:
        take = budget - len(chosen)
        idx = np.linspace(0, len(remaining) - 1, min(take, len(remaining)))
        chosen.update(remaining[int(round(i))] for i in idx)

    return sorted(chosen)


def load_capped(
    path: Path,
    max_rows: int | None = None,
    columns: list[str] | None = None,
    stratify_column: str | None = "Attack",
) -> tuple[pd.DataFrame, dict]:
    """Load a Parquet dataset, optionally capped at ~max_rows.

    Returns the frame and a provenance dict recording what was actually read, so
    every downstream result can state the sampling it rests on.
    """
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    n_groups = pf.metadata.num_row_groups

    if max_rows is None or total <= max_rows:
        df = pd.read_parquet(path, columns=columns)
        return df, {
            "source_rows": int(total), "rows_loaded": int(len(df)),
            "row_groups_total": int(n_groups), "row_groups_read": int(n_groups),
            "capped": False,
        }

    rows_per_group = total / max(n_groups, 1)
    schema_names = {f.name for f in pf.schema_arrow}

    if stratify_column and stratify_column in schema_names:
        coverage = _family_coverage(pf, stratify_column)
        groups = _select_groups(coverage, rows_per_group, max_rows)
        covered = {f for i in groups for f in coverage[i]}
        missing = {f for g in coverage for f in g} - covered
    else:
        keep_n = max(1, int(max_rows / max(rows_per_group, 1)))
        step = max(1, n_groups // keep_n)
        groups = list(range(0, n_groups, step))[:keep_n]
        covered, missing = set(), set()

    df = pf.read_row_groups(groups, columns=columns).to_pandas()
    return df, {
        "source_rows": int(total), "rows_loaded": int(len(df)),
        "row_groups_total": int(n_groups), "row_groups_read": len(groups),
        "capped": True,
        "family_aware": bool(stratify_column and stratify_column in schema_names),
        "families_covered": sorted(covered),
        "families_missing": sorted(missing),
        # Windows are built inside the loaded frame, so a window straddling the
        # join between two non-adjacent row groups mixes distant traffic. At
        # ~400k rows per group and 10k-flow windows that is a small fraction of
        # windows; recorded rather than ignored.
        "note": "contiguous blocks; joins between kept groups are discontinuous",
    }
