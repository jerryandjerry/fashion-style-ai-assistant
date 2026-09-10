"""Evaluation harness.

Asks each case's question of an object with ``recommend_dataframe()`` and grades the answer
against the held-out purchase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .metrics import (
    average_precision_at_k,
    category_diversity,
    inventory_hit_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    summarize_offline_metrics,
)


@dataclass(frozen=True)
class EvalSummary:
    """One evaluation of one assistant, ready for the leaderboard.

    ``name``/``version`` label who was evaluated. ``dataset`` is the comparability key
    (data + eval config); rows only compare within one.
    """

    name: str
    version: str
    dataset: str
    metrics: dict[str, float]


def _predicted_ids(prediction: Any, top_k: int) -> list[str]:
    """Pull ranked product_ids out of a frame, a response dict, or a plain id list."""

    if isinstance(prediction, pd.DataFrame):
        ids = prediction["product_id"].tolist()
    elif isinstance(prediction, dict):
        ids = [rec["product_id"] for rec in prediction.get("recommendations", [])]
    else:
        ids = list(prediction)
    return [str(pid) for pid in ids][:top_k]


def _ranked_frame(
    product_ids: list[str],
    category_by_id: dict[str, Any],
    stock_by_id: dict[str, Any] | None,
) -> pd.DataFrame:
    """Join category and stock onto the returned ids, for the business metrics.

    Lookups are dicts built once per run -- rebuilding them per case cost ~20ms on H&M.
    """

    stock = [1] * len(product_ids) if stock_by_id is None else [stock_by_id.get(pid, 0) for pid in product_ids]
    return pd.DataFrame(
        {
            "product_id": product_ids,
            "category": [category_by_id.get(pid) for pid in product_ids],
            "stock": stock,
        }
    )


def evaluate_assistant(
    assistant: Any,
    cases: pd.DataFrame,
    products: pd.DataFrame,
    inventory: pd.DataFrame,
    top_k: int = 5,
    sample_users: int | None = 100,
    sample_seed: int = 42,
) -> dict[str, float]:
    """Score ``assistant`` on ``cases`` -- each row is one held-out purchase to predict.

    ``cases`` carries the answer key (``user_id`` + ``product_id``); ``products`` phrases the
    question and supplies category, ``inventory`` supplies stock.
    """

    if sample_users is not None:
        if sample_users <= 0:
            raise ValueError("sample_users must be a positive integer")
        if len(cases) > sample_users:
            cases = cases.sample(n=sample_users, random_state=sample_seed).sort_values(
                ["user_id", "event_timestamp"]
            )

    answer = getattr(assistant, "recommend_dataframe", None)
    if not callable(answer):
        raise TypeError(
            f"{type(assistant).__name__} has no recommend_dataframe(query, user_id, top_k); "
            "see scripts/eval/README.md section 2."
        )

    by_id = products.set_index(products["product_id"].astype(str))
    category_by_id = by_id["category"].to_dict() if "category" in products.columns else {}
    stock_by_id = None
    if not inventory.empty and "stock" in inventory.columns:
        stock_by_id = dict(zip(inventory["product_id"].astype(str), inventory["stock"]))

    rows: list[dict[str, float]] = []
    for _, holdout in cases.iterrows():
        user_id = str(holdout["user_id"])
        target = str(holdout["product_id"])
        if target not in by_id.index:
            continue
        relevant = {target}
        product_row = by_id.loc[target]
        if isinstance(product_row, pd.DataFrame):
            product_row = product_row.iloc[0]
        query = (
            f"{product_row.get('style_tags', '')} {product_row.get('category', '')} "
            f"for {product_row.get('occasion_tags', '')}"
        )
        prediction = answer(query=query, user_id=user_id, top_k=top_k)
        recommended = _predicted_ids(prediction, top_k)
        recs = _ranked_frame(recommended, category_by_id, stock_by_id)
        rows.append(
            {
                "recall_at_k": recall_at_k(recommended, relevant, top_k),
                "precision_at_k": precision_at_k(recommended, relevant, top_k),
                "map_at_k": average_precision_at_k(recommended, relevant, top_k),
                "ndcg_at_k": ndcg_at_k(recommended, relevant, top_k),
                "inventory_hit_rate": inventory_hit_rate(recs, top_k),
                "category_diversity": float(category_diversity(recs, top_k)),
            }
        )
    return summarize_offline_metrics(rows)
