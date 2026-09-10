from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path, limit: int | None = None) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def normalize_amazon(reviews_jsonl: Path, meta_jsonl: Path, out_dir: Path, limit: int | None = 100000) -> None:
    reviews_raw = read_jsonl(reviews_jsonl, limit=limit)
    meta_raw = read_jsonl(meta_jsonl, limit=limit)
    if reviews_raw.empty or meta_raw.empty:
        raise ValueError("Amazon review/meta files appear empty")

    asin_col = "parent_asin" if "parent_asin" in meta_raw.columns else "asin"

    def _category(cats: object, main: object) -> str:
        # Prefer the specific `categories` leaf; fall back to real `main_category`.
        if isinstance(cats, list) and cats:
            return str(cats[-1])
        main = str(main) if main not in (None, "None") else ""
        return main or "unknown"

    def _join_desc(d: object) -> str:
        # Amazon `description` is a list of strings; join into one field.
        if isinstance(d, list):
            return " ".join(str(x).strip() for x in d if str(x).strip())
        return "" if d in (None, "None") else str(d)

    def _first_image(imgs: object) -> str:
        if isinstance(imgs, list) and imgs and isinstance(imgs[0], dict):
            return str(imgs[0].get("large") or imgs[0].get("thumb") or "")
        return ""

    n = len(meta_raw)
    cats_series = meta_raw["categories"] if "categories" in meta_raw.columns else pd.Series([[]] * n)
    main_series = meta_raw["main_category"] if "main_category" in meta_raw.columns else pd.Series([""] * n)
    products = pd.DataFrame(
        {
            "product_id": meta_raw[asin_col].astype(str),
            "name": meta_raw.get("title", "Unknown product"),
            "category": [_category(c, m) for c, m in zip(cats_series, main_series)],
            "color": "unknown",
            "style_tags": "",
            "occasion_tags": "",
            "season": "unknown",
            "price": pd.to_numeric(meta_raw.get("price", 0), errors="coerce").fillna(0),
            "description": meta_raw.get("description", pd.Series([""] * n)).apply(_join_desc),
            "brand": meta_raw.get("store", ""),
            "material": "",
            "gender": "unknown",
            "image_url": meta_raw.get("images", pd.Series([[]] * n)).apply(_first_image),
        }
    ).drop_duplicates("product_id")

    review_asin_col = "parent_asin" if "parent_asin" in reviews_raw.columns else "asin"
    reviews = pd.DataFrame(
        {
            "review_id": reviews_raw.get("rating_number", pd.Series(range(len(reviews_raw)))).astype(str),
            "product_id": reviews_raw[review_asin_col].astype(str),
            "rating": pd.to_numeric(reviews_raw.get("rating", 0), errors="coerce").fillna(0),
            "fit_feedback": "unknown",
            "review_text": reviews_raw.get("text", ""),
        }
    )

    interactions = pd.DataFrame(
        {
            "user_id": reviews_raw.get("user_id", "anonymous").astype(str),
            "product_id": reviews_raw[review_asin_col].astype(str),
            "event_type": "review",
            "event_timestamp": reviews_raw.get("timestamp", "1970-01-01"),
            "event_weight": pd.to_numeric(reviews_raw.get("rating", 0), errors="coerce").fillna(0) / 5.0,
        }
    )

    inventory = products[["product_id"]].copy()
    inventory["stock"] = 1
    inventory["margin"] = 0.35
    inventory["return_rate"] = 0.10
    users = interactions[["user_id"]].drop_duplicates().copy()
    users["age_band"] = "unknown"
    users["preferred_style"] = ""
    users["budget_tier"] = "unknown"

    out_dir.mkdir(parents=True, exist_ok=True)
    products.to_csv(out_dir / "products.csv", index=False)
    interactions.to_csv(out_dir / "interactions.csv", index=False)
    reviews.to_csv(out_dir / "reviews.csv", index=False)
    inventory.to_csv(out_dir / "inventory.csv", index=False)
    users.to_csv(out_dir / "users.csv", index=False)
    pd.DataFrame(
        columns=["outfit_id", "product_ids", "occasion_tags", "description", "compatibility_score"]
    ).to_csv(out_dir / "outfits.csv", index=False)
    print(f"Wrote normalized Amazon data to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews-jsonl", required=True, type=Path)
    parser.add_argument("--meta-jsonl", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100000)
    args = parser.parse_args()
    normalize_amazon(args.reviews_jsonl, args.meta_jsonl, args.out_dir, args.limit)


if __name__ == "__main__":
    main()
