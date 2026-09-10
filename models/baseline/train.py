"""Precompute the baseline's user profiles and save models/baseline/user_profiles.csv.

The baseline's one expensive lookup is the per-user preference profile, today rebuilt from the
raw 27M-row interactions frame on every question. This script computes ALL users' profiles once
-- the vectorized equivalent of ranking.build_user_profile (same joins, same recency decay,
same normalisation; verified identical) -- as one plain CSV:

    user_id, kind (0=category 1=color 2=style), key, value

That CSV is the baseline's whole artifact. Everything else ``build()`` needs is either cheap to
recompute (the TF-IDF index fits from products.csv in seconds) or already lives in the data
pool (users.csv, inventory.csv, reviews.csv).

    uv run python models/baseline/train.py            # full train/ (needs RAM -> instance)
    uv run python models/baseline/train.py --data-dir <dir> --out <path>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fashion_style_ai_assistant.data import load_catalog_bundle  # noqa: E402


def _kind_rows(joined: pd.DataFrame, column: str, kind: int, split: bool) -> pd.DataFrame:
    """(user_id, key, weighted) rows for one affinity kind, mirroring weighted_counts()."""

    if column not in joined.columns:
        return pd.DataFrame(columns=["user_id", "kind", "key", "w"])
    if split:
        # _split_tags semantics: per-row SET of comma/semicolon-separated tags
        tags = (
            joined[column]
            .fillna("")
            .astype(str)
            .str.replace(";", ",")
            .str.split(",")
            .apply(lambda parts: sorted({p.strip().lower() for p in parts if p.strip()}))
        )
        rows = pd.DataFrame({"user_id": joined["user_id"], "key": tags, "w": joined["weighted_event"]})
        rows = rows.explode("key").dropna(subset=["key"])
    else:
        # single-value semantics: str().strip().lower() when notna, skip empty
        key = joined[column].where(joined[column].notna()).astype(str).str.strip().str.lower()
        rows = pd.DataFrame({"user_id": joined["user_id"], "key": key, "w": joined["weighted_event"]})
        rows = rows.dropna(subset=["key"])
        rows = rows[rows["key"] != ""]
    rows["kind"] = np.int8(kind)
    return rows


def build_affinity(interactions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """All users' profiles at once -- vectorized build_user_profile (verified equivalent)."""

    joined = interactions.merge(products, on="product_id", how="left")
    # recency decay per user, exactly as build_user_profile: days behind the user's newest event
    if "event_timestamp" in joined.columns:
        user_max = joined.groupby("user_id")["event_timestamp"].transform("max")
        age_days = (user_max - joined["event_timestamp"]).dt.days.fillna(0)
        joined["recency_weight"] = np.exp(-age_days / 60.0)
    else:
        joined["recency_weight"] = 1.0
    joined["weighted_event"] = joined["event_weight"].astype(float) * joined["recency_weight"].astype(float)

    parts = [
        _kind_rows(joined, "category", 0, split=False),
        _kind_rows(joined, "color", 1, split=False),
        _kind_rows(joined, "style_tags", 2, split=True),
    ]
    rows = pd.concat(parts, ignore_index=True)
    summed = rows.groupby(["user_id", "kind", "key"], sort=False)["w"].sum()
    # normalize per (user, kind) by the sum of absolute weights, `or 1.0` like the original
    totals = summed.abs().groupby(level=[0, 1]).sum()
    totals = totals.where(totals != 0, 1.0)
    values = (summed / totals).rename("value").reset_index()
    values["key"] = values["key"].astype("category")
    values["kind"] = values["kind"].astype(np.int8)
    return values.set_index("user_id").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute baseline user profiles -> user_profiles.csv")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "processed" / "hm" / "train"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "user_profiles.csv"))
    args = parser.parse_args()

    t0 = time.time()
    bundle = load_catalog_bundle(args.data_dir)
    affinity = build_affinity(bundle.interactions, bundle.products)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    affinity.to_csv(out, index=True)
    print(
        f"Saved {out} "
        f"({len(affinity):,} rows, {affinity.index.nunique():,} users, {time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
