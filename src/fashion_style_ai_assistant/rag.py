from __future__ import annotations

import pandas as pd

from .schemas import RecommendationEvidence, RecommendationResult

# Human-readable labels for the fit_feedback values.
_FIT_LABELS = {"fit": "true-to-size", "large": "runs large", "small": "runs small"}


# Aggregate a product's reviews into count, average rating, a fit-feedback
# distribution, and a small deterministic sample of quotes (highest-rated first,
# review_id as tie-break).
def _summarize_reviews(product_id: str, reviews: pd.DataFrame, max_samples: int = 2) -> dict | None:
    if reviews.empty or "product_id" not in reviews.columns:
        return None
    rows = reviews[reviews["product_id"].astype(str) == str(product_id)].copy()
    if rows.empty:
        return None
    if "rating" in rows.columns:
        rows["rating"] = pd.to_numeric(rows["rating"], errors="coerce").fillna(0.0)
    else:
        rows["rating"] = 0.0

    fit_distribution: dict[str, float] = {}
    if "fit_feedback" in rows.columns:
        fb = rows["fit_feedback"].astype(str).str.strip().str.lower()
        fb = fb[fb != ""]
        if not fb.empty:
            # frequency desc, then label asc -> deterministic ordering
            for label, n in sorted(fb.value_counts().items(), key=lambda kv: (-kv[1], kv[0])):
                fit_distribution[label] = round(n / len(fb), 2)

    sort_cols, ascending = ["rating"], [False]
    if "review_id" in rows.columns:
        sort_cols.append("review_id")
        ascending.append(True)
    rows = rows.sort_values(sort_cols, ascending=ascending)

    return {
        "count": int(len(rows)),
        "avg_rating": round(float(rows["rating"].mean()), 1),
        "fit_distribution": fit_distribution,
        "samples": [rows.iloc[i] for i in range(min(max_samples, len(rows)))],
    }

def build_item_explanation(row: pd.Series, reviews: pd.DataFrame) -> RecommendationResult:
    product_id = str(row["product_id"])
    evidence: list[RecommendationEvidence] = []
    why: list[str] = []

    def add_evidence(source: str, source_id: str, field: str, value: object) -> None:
        if pd.isna(value):
            return
        text = str(value)
        if text:
            evidence.append(RecommendationEvidence(source=source, source_id=source_id, field=field, value=text))

    add_evidence("catalog", product_id, "name", row.get("name", ""))
    add_evidence("catalog", product_id, "category", row.get("category", ""))
    add_evidence("catalog", product_id, "color", row.get("color", ""))
    add_evidence("catalog", product_id, "style_tags", row.get("style_tags", ""))
    add_evidence("catalog", product_id, "occasion_tags", row.get("occasion_tags", ""))
    add_evidence("catalog", product_id, "description", row.get("description", ""))

    why.append(
        f"Matches the request through {row.get('category', 'item')} / {row.get('color', 'color')} / "
        f"{row.get('style_tags', 'style')} signals."
    )

    if float(row.get("preference_score", 0.0)) > 0.05:
        why.append("Aligns with the user's past interaction profile.")
    if float(row.get("inventory_score", 0.0)) > 0:
        why.append("Currently in stock, so it is safe to recommend in the demo flow.")
    else:
        why.append("Out-of-stock risk: keep as backup only, not a primary recommendation.")
    if float(row.get("return_rate", 0.0)) <= 0.08:
        why.append("Low return-risk proxy relative to the demo catalog.")

    summary = _summarize_reviews(product_id, reviews)
    if summary is not None:
        # Aggregate signals over ALL of this product's reviews.
        add_evidence("reviews", product_id, "count", summary["count"])
        add_evidence("reviews", product_id, "avg_rating", summary["avg_rating"])
        dist_str = ", ".join(
            f"{int(round(pct * 100))}% {_FIT_LABELS.get(label, label)}"
            for label, pct in summary["fit_distribution"].items()
        )
        if dist_str:
            add_evidence("reviews", product_id, "fit_distribution", dist_str)
        # A small, capped sample of the highest-rated quotes -- never the whole set.
        for sample in summary["samples"]:
            review_id = str(sample.get("review_id", f"review:{product_id}"))
            add_evidence("review", review_id, "review_text", sample.get("review_text", ""))
        fit_suffix = f"; fit {dist_str}" if dist_str else ""
        why.append(f"Reviews: avg {summary['avg_rating']}/5 across {summary['count']}{fit_suffix}.")
    else:
        # No reviews for this product -- keep the same format, just with zero/none.
        add_evidence("reviews", product_id, "count", 0)
        why.append("Reviews: none available.")

    business_signals = {
        "stock": int(row.get("stock", 0)),
        "margin": float(row.get("margin", 0.0)),
        "return_rate": float(row.get("return_rate", 0.0)),
        "retrieval_score": float(row.get("retrieval_score", 0.0)),
        "preference_score": float(row.get("preference_score", 0.0)),
    }

    return RecommendationResult(
        product_id=product_id,
        rank=int(row.get("rank", 0)),
        score=float(row.get("score", 0.0)),
        retrieval_score=float(row.get("retrieval_score", 0.0)),
        product=row.to_dict(),
        why=why,
        evidence=evidence,
        business_signals=business_signals,
    )


def render_response(
    query: str,
    user_id: str,
    items: list[RecommendationResult],
    clarification_question: str | None = None,
) -> dict[str, object]:
    """Render assistant-built ``RecommendationResult``s into the response object.

    Assistant-blind: any assistant that emits ``list[RecommendationResult]`` renders here.
    """

    response_items: list[dict[str, object]] = []
    for item in items:
        product = item.product
        response_items.append(
            {
                "rank": item.rank,
                "product_id": item.product_id,
                "name": product.get("name"),
                "category": product.get("category"),
                "color": product.get("color"),
                "price": float(product.get("price", 0.0)),
                "score": round(item.score, 4),
                "why": item.why,
                "evidence": [e.__dict__ for e in item.evidence],
                "business_signals": item.business_signals,
            }
        )

    return {
        "query": query,
        "user_id": user_id,
        "clarification_question": clarification_question,
        "recommendations": response_items,
    }

def response_to_markdown(response: dict[str, object]) -> str:
    """Format a response object for CLI/demo output."""

    lines: list[str] = []
    lines.append(f"# Recommendation result — user `{response['user_id']}`")
    lines.append("")
    lines.append(f"**Query:** `{response['query']}`")
    lines.append("")
    if response.get("clarification_question"):
        lines.append(f"> Clarification to ask if this were interactive: {response['clarification_question']}")
        lines.append("")

    recs = response.get("recommendations", [])
    if not isinstance(recs, list) or not recs:
        lines.append("No recommendations found.")
        return "\n".join(lines)

    for rec in recs:
        lines.append(f"## {rec['rank']}. {rec['name']} — ${rec['price']:.2f}")
        lines.append(f"Product ID: `{rec['product_id']}` | Score: `{rec['score']}`")
        lines.append("")
        lines.append("Why this item:")
        for reason in rec.get("why", []):
            lines.append(f"- {reason}")
        lines.append("")
        evidence_items = rec.get("evidence", [])
        review_samples = [e for e in evidence_items if e.get("source") == "review"]
        grounding = [e for e in evidence_items if e.get("source") != "review"]
        lines.append("Grounding evidence:")
        for evidence in grounding:
            lines.append(
                f"- `{evidence['source']}:{evidence['source_id']}.{evidence['field']}` → {evidence['value']}"
            )
        lines.append("- Sample top reviews:")
        if review_samples:
            for evidence in review_samples:
                lines.append(f"  - `{evidence['source_id']}.{evidence['field']}` → {evidence['value']}")
        else:
            lines.append("  - (none)")
        lines.append("")
        signals = rec.get("business_signals", {})
        if isinstance(signals, dict):
            lines.append(
                "Business signals: "
                f"stock={signals.get('stock')}, margin={signals.get('margin'):.2f}, "
                f"return_rate={signals.get('return_rate'):.2f}"
            )
        lines.append("")
    return "\n".join(lines)
