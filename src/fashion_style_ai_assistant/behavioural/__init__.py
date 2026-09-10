"""Behavioural assistant v4: faithful Kaggle 1st-place port, pre-trained.

Trained offline by ``models/behavioural/train.py`` (see models/behavioural/SPEC.md).
``build()`` loads the trained weights + the precomputed candidate store and returns the
ready assistant.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from ..data import real_ids
from .pipeline import FashionAssistant
from .retrieval import CandidateStore

WEIGHTS = Path("models/behavioural/weights.pkl")
QUERY_SERVE = Path("models/behavioural/query_serve.pkl")
STORES = {"small": Path("models/behavioural/serve_small"),
          "large": Path("models/behavioural/serve_large")}
# catalog + inventory used only to decorate the human-facing response
TRAIN_DIR = Path("data/processed/hm/train")


def build(data_dir: str | Path | None = None) -> FashionAssistant:
    """Return a ready-to-answer behavioural v5 (silver-exact) assistant."""

    for path in (WEIGHTS, *[d / "meta.json" for d in STORES.values()]):
        if not path.exists():
            raise SystemExit(
                f"Missing trained artifact {path}. Train first (full train/ -> instance):\n"
                f"  python models/behavioural/train.py"
            )
    with WEIGHTS.open("rb") as fh:
        weights = pickle.load(fh)
    stores = {name: CandidateStore(p) for name, p in STORES.items() if name in weights["strategies"]}
    query_serve = None
    if "query" in weights["strategies"]:
        with QUERY_SERVE.open("rb") as fh:
            query_serve = pickle.load(fh)

    base = Path(data_dir) if data_dir is not None else TRAIN_DIR
    products = pd.read_csv(base / "products.csv")
    products["product_id"] = real_ids(products["product_id"])
    inventory = pd.DataFrame()
    inventory_path = base / "inventory.csv"
    if inventory_path.exists():
        inventory = pd.read_csv(inventory_path)
        inventory["product_id"] = real_ids(inventory["product_id"])
    return FashionAssistant(weights, stores, products, inventory, query_serve=query_serve)
