from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import RankingConfig
from ..schemas import ParsedIntent


EVENT_WEIGHTS = {
    "view": 0.2,
    "cart": 0.6,
    "favorite": 0.8,
    "purchase": 1.0,
    "return": -1.0,
}

# split strings into a set of words
def _split_tags(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {part.strip().lower() for part in str(value).replace(";", ",").split(",") if part.strip()}


def build_user_profile(
    user_id: str,
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    users: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Build a compact user profile from prior interactions and optional user metadata."""

    profile: dict[str, object] = {
        "known_user": False,
        "category_affinity": {}, # {"shirt": 0.5, "pants": 0.3}
        "color_affinity": {}, # {"ivory": 0.4, "sand": 0.3}
        "style_affinity": {}, # {"casual": 0.5, "coastal": 0.4}
        "preferred_style_terms": set(), # {"minimal", "casual"}
        "budget_tier": None,
    }

    if users is not None and not users.empty and "user_id" in users.columns:
        row = users[users["user_id"].astype(str) == str(user_id)]
        if not row.empty:
            profile["known_user"] = True 
            if "preferred_style" in row.columns:
                profile["preferred_style_terms"] = _split_tags(row.iloc[0]["preferred_style"])
            if "budget_tier" in row.columns:
                profile["budget_tier"] = row.iloc[0]["budget_tier"]
    # history comes from interaction.csv
    history = interactions[interactions["user_id"].astype(str) == str(user_id)].copy()
    if history.empty:
        return profile

    profile["known_user"] = True
    if "event_weight" not in history.columns:
        history["event_weight"] = history["event_type"].map(EVENT_WEIGHTS).fillna(0.1)

    joined = history.merge(products, on="product_id", how="left")
    # Recency multiplier: newer rows get slightly more weight.
    if "event_timestamp" in joined.columns and joined["event_timestamp"].notna().any():
        max_ts = joined["event_timestamp"].max() # find the newest timestamp of the interaction
        age_days = (max_ts - joined["event_timestamp"]).dt.days.fillna(0) 
        joined["recency_weight"] = np.exp(-age_days / 60.0)
    else:
        # if no timestamp, treat everything as most recent
        joined["recency_weight"] = 1.0
    # final event weight = event_weight * recency_weight
    joined["weighted_event"] = joined["event_weight"].astype(float) * joined["recency_weight"].astype(float)

    def weighted_counts(column: str) -> dict[str, float]:
        counts: dict[str, float] = {}
        for _, row in joined.iterrows():
            weight = float(row.get("weighted_event", 0.0))
            if column in {"style_tags", "occasion_tags"}:
                values = _split_tags(row.get(column, ""))
            else: # for 'category' and 'color' columns
                values = {str(row.get(column, "")).strip().lower()} if pd.notna(row.get(column)) else set()
            for value in values:
                if value:
                    counts[value] = counts.get(value, 0.0) + weight
        total = sum(abs(v) for v in counts.values()) or 1.0
        return {k: v / total for k, v in counts.items()}

    profile["category_affinity"] = weighted_counts("category")
    profile["color_affinity"] = weighted_counts("color")
    profile["style_affinity"] = weighted_counts("style_tags")
    return profile

# this checks the product(rows) against user's profile(preference)
def _affinity_score(row: pd.Series, profile: dict[str, object]) -> float:
    # type: ignore[assignment] surpress type warning
    category_affinity: dict[str, float] = profile.get("category_affinity", {})  # type: ignore[assignment]
    color_affinity: dict[str, float] = profile.get("color_affinity", {})  # type: ignore[assignment]
    style_affinity: dict[str, float] = profile.get("style_affinity", {})  # type: ignore[assignment]
    preferred_style_terms: set[str] = profile.get("preferred_style_terms", set())  # type: ignore[assignment]

    category = str(row.get("category", "")).lower()
    color = str(row.get("color", "")).lower()
    style_tags = _split_tags(row.get("style_tags", ""))

    # precompute style score since each product has multiple styles. check use's preference for each style
    style_score = sum(style_affinity.get(tag, 0.0) for tag in style_tags)
    preferred_overlap = len(style_tags & preferred_style_terms) / max(len(style_tags | preferred_style_terms), 1)

    score = 0.40 * category_affinity.get(category, 0.0)
    score += 0.20 * color_affinity.get(color, 0.0)
    score += 0.25 * style_score
    score += 0.15 * preferred_overlap
    return float(max(0.0, min(1.0, score)))

# this checks the product(rows) against the intent
def _intent_match_score(row: pd.Series, intent: ParsedIntent) -> float:
    product_tags = set()
    for col in ["category", "color", "style_tags", "occasion_tags", "season", "description", "name"]:
        value = row.get(col, "")
        if col in {"style_tags", "occasion_tags"}:
            product_tags |= _split_tags(value)
        else:
            product_tags.add(str(value).strip().lower())

    wanted = set(intent.occasion_terms + intent.color_terms + intent.category_terms + intent.style_terms)
    if not wanted:
        return 0.0
    overlap = len(product_tags & wanted)
    # Also allow substring matches in the full product text.
    full_text = " ".join(str(row.get(col, "")).lower() for col in row.index)
    overlap += sum(1 for term in wanted if term in full_text and term not in product_tags)
    return min(1.0, overlap / max(len(wanted), 1))

# this checks the product price against the budget
def _price_score(price: float, budget_max: float | None) -> float:
    if budget_max is None:
        return 1.0
    if price <= budget_max:
        return 1.0
    overage = (price - budget_max) / max(budget_max, 1.0)
    return max(0.0, 1.0 - overage * 2.0)


def rank_candidates(
    candidates: pd.DataFrame,
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    inventory: pd.DataFrame,
    users: pd.DataFrame,
    user_id: str,
    intent: ParsedIntent,
    ranking_config: RankingConfig | None = None,
    profile: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Re-rank retrieved products with personalization and business signals.

    ``profile`` is the precomputed user profile from the trained artifact; when None (the
    live-data path used by demos/tests), it is built on the fly exactly as before.
    """

    if candidates.empty:
        return candidates.copy()

    weights = ranking_config or RankingConfig()

    # cached profile from the artifact, or build it live from interactions
    if profile is None:
        profile = build_user_profile(user_id, products, interactions, users)

    ranked = candidates.copy()
    if not inventory.empty:
        ranked = ranked.merge(inventory, on="product_id", how="left")
    else:
        ranked["stock"] = 1
        ranked["margin"] = 0.30
        ranked["return_rate"] = 0.10

    ranked["stock"] = ranked["stock"].fillna(0)
    # recommend more profitable product
    ranked["margin"] = ranked["margin"].fillna(ranked["margin"].median() if ranked["margin"].notna().any() else 0.30)
    ranked["return_rate"] = ranked["return_rate"].fillna(0.10)

    margin_min = ranked["margin"].min()
    margin_max = ranked["margin"].max()
    margin_denom = margin_max - margin_min if not math.isclose(margin_max, margin_min) else 1.0

    ranked["preference_score"] = ranked.apply(lambda row: _affinity_score(row, profile), axis=1)
    
    ranked["intent_match_score"] = ranked.apply(lambda row: _intent_match_score(row, intent), axis=1)
    ranked["price_score"] = ranked["price"].apply(lambda p: _price_score(float(p), intent.budget_max))
    ranked["inventory_score"] = (ranked["stock"] > 0).astype(float)
    ranked["margin_score"] = (ranked["margin"] - margin_min) / margin_denom
    ranked["return_score"] = 1.0 - ranked["return_rate"].clip(0.0, 1.0)

    # Retrieval remains dominant, but business and personalization signals can break ties.
    ranked["score"] = (
        weights.retrieval_weight * ranked["retrieval_score"].astype(float)
        + weights.intent_match_weight * ranked["intent_match_score"].astype(float)
        + weights.preference_weight * ranked["preference_score"].astype(float)
        + weights.price_weight * ranked["price_score"].astype(float)
        + weights.inventory_weight * ranked["inventory_score"].astype(float)
        + weights.margin_weight * ranked["margin_score"].astype(float)
        + weights.return_weight * ranked["return_score"].astype(float)
    )

    # Hard penalty for out-of-stock items. We keep them only if there are too few alternatives.
    ranked.loc[ranked["stock"] <= 0, "score"] -= weights.out_of_stock_penalty

    ranked = ranked.sort_values(
        ["score", "inventory_score", "retrieval_score", "price"], ascending=[False, False, False, True]
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked
