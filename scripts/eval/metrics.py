from __future__ import annotations

import math

import pandas as pd


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def average_precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for idx, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            hits += 1
            score += hits / idx
    return score / min(len(relevant), k)


def dcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    score = 0.0
    for idx, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            score += 1.0 / math.log2(idx + 1)
    return score


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg_at_k(recommended, relevant, k) / ideal if ideal > 0 else 0.0


def inventory_hit_rate(recommendations: pd.DataFrame, k: int) -> float:
    if recommendations.empty or "stock" not in recommendations.columns:
        return 0.0
    top = recommendations.head(k)
    return float((top["stock"] > 0).mean())


def category_diversity(recommendations: pd.DataFrame, k: int) -> int:
    if recommendations.empty or "category" not in recommendations.columns:
        return 0
    return int(recommendations.head(k)["category"].nunique())


def citation_coverage(response: dict[str, object]) -> float:
    recs = response.get("recommendations", [])
    if not isinstance(recs, list) or not recs:
        return 0.0
    with_evidence = 0
    for rec in recs:
        if isinstance(rec, dict) and rec.get("evidence"):
            with_evidence += 1
    return with_evidence / len(recs)


def summarize_offline_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "users_evaluated": 0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "map_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "inventory_hit_rate": 0.0,
            "category_diversity": 0.0,
        }

    df = pd.DataFrame(rows)
    return {
        "users_evaluated": float(len(df)),
        "recall_at_k": float(df["recall_at_k"].mean()),
        "precision_at_k": float(df["precision_at_k"].mean()),
        "map_at_k": float(df["map_at_k"].mean()),
        "ndcg_at_k": float(df["ndcg_at_k"].mean()),
        "inventory_hit_rate": float(df["inventory_hit_rate"].mean()),
        "category_diversity": float(df["category_diversity"].mean()),
    }
