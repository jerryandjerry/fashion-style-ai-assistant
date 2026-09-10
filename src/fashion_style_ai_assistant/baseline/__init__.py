"""Baseline assistant: TF-IDF retrieval + weighted-sum rerank.

``build()`` is the submission factory (see scripts/eval/README.md section 2). Zero-arg call
composes the ready assistant from:

  - ``models/baseline/user_profiles.csv`` -- the precomputed per-user profiles (the one
    expensive lookup; built once by ``models/baseline/train.py``)
  - the data pool -- products.csv (TF-IDF fits from it in seconds), users.csv, inventory.csv,
    reviews.csv, all read from ``data/processed/hm/train``

With an explicit ``data_dir`` (demo/CLI), it fits live from that directory instead -- same
math, small data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..data import real_ids
from ..data import REVIEW_REQUIRED, _read_optional_csv
from .pipeline import FashionAssistant
from .retrieval import TfidfCatalogRetriever

PROFILES = Path("models/baseline/user_profiles.csv")
TRAIN_DIR = Path("data/processed/hm/train")


def build(data_dir: str | Path | None = None) -> FashionAssistant:
    """Return a ready-to-answer baseline. ``data_dir`` is a demo/CLI override only."""

    config = load_config("config.yaml")

    if data_dir is not None:
        assistant = FashionAssistant.from_data_dir(
            data_dir,
            max_features=config.max_features,
            ranking_config=config.ranking,
        )
        assistant.name = "baseline"
        assistant.version = "1.0"
        return assistant

    if not PROFILES.exists():
        raise SystemExit(
            f"No precomputed profiles at {PROFILES}. Build them first (full train/ -> instance):\n"
            f"  uv run python models/baseline/train.py"
        )

    # products/inventory normalized exactly as load_catalog_bundle does (without touching the
    # multi-GB interactions.csv -- the profiles CSV replaces it)
    products = pd.read_csv(TRAIN_DIR / "products.csv")
    products["product_id"] = real_ids(products["product_id"])
    products["price"] = pd.to_numeric(products["price"], errors="coerce").fillna(0.0)

    inventory = _read_optional_csv(TRAIN_DIR / "inventory.csv", {"product_id", "stock", "margin", "return_rate"})
    if not inventory.empty:
        inventory["product_id"] = real_ids(inventory["product_id"])
        inventory["stock"] = pd.to_numeric(inventory["stock"], errors="coerce").fillna(0)
        inventory["margin"] = pd.to_numeric(inventory["margin"], errors="coerce").fillna(0.0)
        inventory["return_rate"] = pd.to_numeric(inventory["return_rate"], errors="coerce").fillna(0.0)

    reviews = _read_optional_csv(TRAIN_DIR / "reviews.csv", REVIEW_REQUIRED)

    # tight dtypes: at full scale this is ~34M rows; categories keep it in the hundreds of MB
    affinity = pd.read_csv(
        PROFILES,
        dtype={"user_id": "category", "kind": np.int8, "key": "category", "value": np.float64},
    ).set_index("user_id")

    user_meta = _read_optional_csv(TRAIN_DIR / "users.csv", {"user_id"})
    if not user_meta.empty:
        user_meta["user_id"] = user_meta["user_id"].astype(str)
        keep = [c for c in ("preferred_style", "budget_tier") if c in user_meta.columns]
        user_meta = user_meta.set_index("user_id")[keep]

    weights = {
        "retriever": TfidfCatalogRetriever(products, max_features=config.max_features),
        "ranking_config": config.ranking,
        "affinity": affinity,
        "user_meta": user_meta,
        "inventory": inventory,
        "reviews": reviews,
    }
    assistant = FashionAssistant.from_weights(weights)
    assistant.name = "baseline"
    assistant.version = "1.0"
    return assistant
