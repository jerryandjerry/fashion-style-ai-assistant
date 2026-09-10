"""Behavioural v5 retrieval at serve time: chunked candidate stores, filtered reads.

The serve store is a directory of per-user-range parquet chunks. Loading whole chunks blows
the training container's 60GB memory cap, so ``candidates(uidx)`` reads ONLY the requested
user's rows via parquet predicate pushdown (chunks are written uidx-sorted, so pyarrow skips
all non-matching row groups). Unknown users fall back to popularity candidates derived once
from the first row group of chunk 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


class CandidateStore:
    """Lazy chunked store: serve_<strategy>/chunk_XXXX.parquet + meta.json."""

    def __init__(self, path: str | Path):
        self._dir = Path(path)
        meta = json.loads((self._dir / "meta.json").read_text())
        self._chunk_size = int(meta["chunk_size"])
        self._fallback: pd.DataFrame | None = None

    def _fallback_rows(self) -> pd.DataFrame:
        """Popularity candidates for unknown users, from chunk 0's first row group only."""

        if self._fallback is None:
            path = self._dir / "chunk_0000.parquet"
            if not path.exists():
                self._fallback = pd.DataFrame(columns=["uidx", "iidx"])
            else:
                sample = pq.ParquetFile(path).read_row_group(0).to_pandas()
                top = sample["iidx"].value_counts().head(500).index
                self._fallback = sample[sample["iidx"].isin(top)].drop_duplicates("iidx")
        return self._fallback

    def candidates(self, uidx: int | None) -> pd.DataFrame:
        if uidx is not None:
            path = self._dir / f"chunk_{int(uidx) // self._chunk_size:04d}.parquet"
            if path.exists():
                rows = pd.read_parquet(path, filters=[("uidx", "==", int(uidx))])
                if len(rows):
                    return rows.reset_index(drop=True)
        return self._fallback_rows().copy()
