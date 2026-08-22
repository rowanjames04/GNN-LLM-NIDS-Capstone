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
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def load_capped(
    path: Path, max_rows: int | None = None, columns: list[str] | None = None
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
    keep_n = max(1, int(max_rows / max(rows_per_group, 1)))
    step = max(1, n_groups // keep_n)
    # Spread across the whole capture rather than taking a prefix, so the sample
    # covers every phase of the traffic instead of only its opening.
    groups = list(range(0, n_groups, step))[:keep_n]

    df = pf.read_row_groups(groups, columns=columns).to_pandas()
    return df, {
        "source_rows": int(total), "rows_loaded": int(len(df)),
        "row_groups_total": int(n_groups), "row_groups_read": len(groups),
        "row_group_stride": int(step),
        "capped": True,
        # Windows are built inside the loaded frame, so a window straddling the
        # join between two non-adjacent row groups mixes distant traffic. At
        # ~1M rows per group and 10k-flow windows that is about one window in a
        # hundred; recorded rather than ignored.
        "note": "contiguous blocks; joins between kept groups are discontinuous",
    }
