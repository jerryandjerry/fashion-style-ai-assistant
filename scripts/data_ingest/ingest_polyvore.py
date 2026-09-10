from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_polyvore(raw_dir: Path, out_dir: Path) -> None:
    """Normalize Polyvore item metadata into the product schema (+ valid empties).

    Polyvore ships `polyvore_item_metadata.json` as a dict keyed by item_id, each value with
    url_name/title/description/semantic_category/category_id. We map those into products.csv.
    Fields Polyvore has no source for (color, style, occasion, price, ...) are left empty/dummy.
    Outfit-to-item membership is not in these files, so outfits.csv is header-only.
    """

    meta_path = raw_dir / "polyvore_item_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing expected file: {meta_path}")
    items = _load_json(meta_path)
    if not isinstance(items, dict):
        raise ValueError(f"{meta_path} is not a dict keyed by item_id")

    # Optional readable category names from categories.csv: id, fine_name, coarse_name (no header).
    cat_map: dict[str, str] = {}
    cat_csv = raw_dir / "categories.csv"
    if cat_csv.exists():
        cats = pd.read_csv(cat_csv, header=None, dtype=str).fillna("")
        for _, row in cats.iterrows():
            cat_map[str(row[0])] = (str(row[1]) or str(row[2]) or "").strip()

    def category_of(meta: dict) -> str:
        sem = str(meta.get("semantic_category", "") or "").strip()
        if sem:
            return sem
        return cat_map.get(str(meta.get("category_id", "") or ""), "") or "unknown"

    product_rows = []
    for item_id, meta in items.items():
        if not isinstance(meta, dict):
            continue
        product_rows.append(
            {
                "product_id": str(item_id),
                "name": str(meta.get("title") or meta.get("url_name") or "Polyvore item").strip(),
                "category": category_of(meta),
                "color": "unknown",
                "style_tags": "",
                "occasion_tags": "",
                "season": "unknown",
                "price": 0.0,
                "description": str(meta.get("description", "") or ""),
                "brand": "",
                "material": "",
                "gender": "unknown",
                "image_url": "",
            }
        )

    products = pd.DataFrame(product_rows)
    inventory = products[["product_id"]].copy()
    inventory["stock"] = 1
    inventory["margin"] = 0.35
    inventory["return_rate"] = 0.10

    out_dir.mkdir(parents=True, exist_ok=True)
    products.to_csv(out_dir / "products.csv", index=False)
    inventory.to_csv(out_dir / "inventory.csv", index=False)
    # Data Polyvore's metadata does not provide -> empty but schema-valid.
    pd.DataFrame(columns=["user_id", "product_id", "event_type", "event_timestamp", "event_weight"]).to_csv(
        out_dir / "interactions.csv", index=False
    )
    pd.DataFrame(columns=["review_id", "product_id", "rating", "fit_feedback", "review_text"]).to_csv(
        out_dir / "reviews.csv", index=False
    )
    pd.DataFrame(columns=["user_id", "age_band", "preferred_style", "budget_tier"]).to_csv(
        out_dir / "users.csv", index=False
    )
    pd.DataFrame(
        columns=["outfit_id", "product_ids", "occasion_tags", "description", "compatibility_score"]
    ).to_csv(out_dir / "outfits.csv", index=False)
    print(f"Wrote normalized Polyvore data to {out_dir}  ({len(products)} products)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    normalize_polyvore(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
