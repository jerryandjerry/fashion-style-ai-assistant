from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _first_present(raw: pd.DataFrame, columns: list[str], default: str = "") -> pd.Series:
    values = pd.Series("", index=raw.index, dtype="object")
    for column in columns:
        if column in raw.columns:
            candidate = raw[column].fillna("").astype(str)
            values = values.where(values.astype(str).str.len() > 0, candidate)
    return values.where(values.astype(str).str.len() > 0, default)


def _join_columns(raw: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [column for column in columns if column in raw.columns]
    if not present:
        return pd.Series("", index=raw.index, dtype="object")
    return raw[present].fillna("").astype(str).agg(
        lambda row: ",".join(dict.fromkeys(value for value in row if value and value != "nan")),
        axis=1,
    )


def normalize_fashion_images(raw_dir: Path, out_dir: Path, limit: int | None = None) -> None:
    styles_path = raw_dir / "styles.csv"
    if not styles_path.exists():
        raise FileNotFoundError(f"Missing Fashion Product Images file: {styles_path}")

    raw = pd.read_csv(styles_path, on_bad_lines="skip")
    if raw.empty:
        raise ValueError(f"No rows found in {styles_path}")
    if limit:
        raw = raw.head(limit).copy()

    image_dir = raw_dir / "images"
    if not image_dir.exists():
        image_dir = raw_dir / "myntradataset" / "images"

    product_id = raw["id"].astype(str)
    image_paths = [
        str(image_dir / f"{pid}.jpg") if (image_dir / f"{pid}.jpg").exists() else ""
        for pid in product_id
    ]
    name = _first_present(raw, ["productDisplayName"], default="Unknown product")
    category = _first_present(raw, ["articleType", "subCategory", "masterCategory"], default="fashion")
    color = _first_present(raw, ["baseColour"], default="unknown")
    season = _first_present(raw, ["season"], default="unknown")

    products = pd.DataFrame(
        {
            "product_id": product_id,
            "name": name,
            "category": category,
            "color": color,
            "style_tags": _join_columns(raw, ["gender", "masterCategory", "subCategory", "usage"]),
            "occasion_tags": _first_present(raw, ["usage"], default=""),
            "season": season,
            "price": 0.0,
            "description": raw[
                [
                    column
                    for column in [
                        "productDisplayName",
                        "gender",
                        "masterCategory",
                        "subCategory",
                        "articleType",
                        "baseColour",
                        "season",
                        "usage",
                    ]
                    if column in raw.columns
                ]
            ]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1),
            "brand": pd.Series("", index=raw.index),
            "material": pd.Series("", index=raw.index),
            "gender": _first_present(raw, ["gender"], default="unknown"),
            "image_url": image_paths,
        }
    ).drop_duplicates("product_id")

    inventory = products[["product_id"]].copy()
    inventory["stock"] = 1
    inventory["margin"] = 0.35
    inventory["return_rate"] = 0.10

    out_dir.mkdir(parents=True, exist_ok=True)
    products.to_csv(out_dir / "products.csv", index=False)
    pd.DataFrame(columns=["user_id", "product_id", "event_type", "event_timestamp", "event_weight"]).to_csv(
        out_dir / "interactions.csv", index=False
    )
    pd.DataFrame(columns=["review_id", "product_id", "rating", "fit_feedback", "review_text"]).to_csv(
        out_dir / "reviews.csv", index=False
    )
    inventory.to_csv(out_dir / "inventory.csv", index=False)
    pd.DataFrame(columns=["user_id", "age_band", "preferred_style", "budget_tier"]).to_csv(
        out_dir / "users.csv", index=False
    )
    pd.DataFrame(
        columns=["outfit_id", "product_ids", "occasion_tags", "description", "compatibility_score"]
    ).to_csv(out_dir / "outfits.csv", index=False)
    print(f"Wrote normalized Fashion Product Images data to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of style rows to keep")
    args = parser.parse_args()
    normalize_fashion_images(args.raw_dir, args.out_dir, args.limit)


if __name__ == "__main__":
    main()
