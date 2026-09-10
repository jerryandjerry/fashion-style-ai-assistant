from __future__ import annotations

import pandas as pd


def summarize_business_metrics(recommendations: pd.DataFrame) -> dict[str, float | int]:
    """Compute simple business-facing proxy metrics for a recommendation list."""

    if recommendations.empty:
        return {
            "num_recommendations": 0,
            "in_stock_rate": 0.0,
            "avg_margin": 0.0,
            "avg_return_rate": 0.0,
            "category_diversity": 0,
        }

    return {
        "num_recommendations": int(len(recommendations)),
        "in_stock_rate": float((recommendations["stock"] > 0).mean()) if "stock" in recommendations else 0.0,
        "avg_margin": float(recommendations["margin"].mean()) if "margin" in recommendations else 0.0,
        "avg_return_rate": float(recommendations["return_rate"].mean()) if "return_rate" in recommendations else 0.0,
        "category_diversity": int(recommendations["category"].nunique()) if "category" in recommendations else 0,
    }


def estimate_business_outcome(recommendations: pd.DataFrame) -> dict[str, str]:
    """Translate proxy metrics into plain-English business interpretation."""

    metrics = summarize_business_metrics(recommendations)
    notes: dict[str, str] = {}
    notes["availability"] = (
        "Strong: most recommended products are in stock."
        if metrics["in_stock_rate"] >= 0.8
        else "Risk: several recommended products are out of stock."
    )
    notes["margin"] = (
        "Healthy margin mix for a demo catalog."
        if metrics["avg_margin"] >= 0.40
        else "Lower margin mix; consider margin-aware ranking."
    )
    notes["return_risk"] = (
        "Return-risk proxy is acceptable."
        if metrics["avg_return_rate"] <= 0.10
        else "Return-risk proxy is elevated; add fit-risk modeling."
    )
    notes["diversity"] = (
        "Recommendation set covers multiple categories."
        if metrics["category_diversity"] >= 3
        else "Recommendation set may be too narrow."
    )
    return notes
