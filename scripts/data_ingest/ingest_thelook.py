from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _require_files(raw_dir: Path, names: list[str]) -> dict[str, Path]:
    paths = {name: raw_dir / name for name in names}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing theLook files: {missing}")
    return paths


def _age_band(age: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(age, errors="coerce")
    return pd.cut(
        numeric,
        bins=[0, 17, 24, 34, 44, 54, 64, 120],
        labels=["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype(str)


def _event_type_from_status(status: object) -> str:
    value = str(status).strip().lower()
    if value == "returned":
        return "return"
    if value == "cancelled":
        return "cancel"
    if value == "processing":
        return "cart"
    return "purchase"


def _event_weight_from_type(event_type: str) -> float:
    return {"purchase": 1.0, "cart": 0.6, "cancel": -0.4, "return": -1.0}.get(event_type, 0.1)


def normalize_thelook(raw_dir: Path, out_dir: Path, sample_users: int | None = None) -> None:
    paths = _require_files(
        raw_dir,
        [
            "products.csv",
            "users.csv",
            "order_items.csv",
            "inventory_items.csv",
        ],
    )

    raw_products = pd.read_csv(paths["products.csv"])
    raw_users = pd.read_csv(paths["users.csv"])
    order_items = pd.read_csv(paths["order_items.csv"])
    inventory_items = pd.read_csv(paths["inventory_items.csv"])

    if sample_users:
        user_counts = order_items["user_id"].value_counts().head(sample_users)
        order_items = order_items[order_items["user_id"].isin(user_counts.index)].copy()
        raw_users = raw_users[raw_users["id"].isin(user_counts.index)].copy()

    products = pd.DataFrame(
        {
            "product_id": raw_products["id"].astype(str),
            "name": raw_products["name"].fillna("Unknown product"),
            "category": raw_products["category"].fillna("unknown"),
            "color": "unknown",
            "style_tags": raw_products[["department", "category", "brand"]]
            .fillna("")
            .astype(str)
            .agg(lambda row: ",".join(dict.fromkeys(value for value in row if value)), axis=1),
            "occasion_tags": pd.Series("", index=raw_products.index),
            "season": "unknown",
            "price": pd.to_numeric(raw_products["retail_price"], errors="coerce").fillna(0.0),
            "description": raw_products[["name", "category", "brand", "department"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1),
            "brand": raw_products["brand"].fillna(""),
            "material": pd.Series("", index=raw_products.index),
            "gender": raw_products["department"].fillna("unknown"),
            "image_url": pd.Series("", index=raw_products.index),
        }
    )

    event_types = order_items["status"].apply(_event_type_from_status)
    interactions = pd.DataFrame(
        {
            "user_id": order_items["user_id"].astype(str),
            "product_id": order_items["product_id"].astype(str),
            "event_type": event_types,
            "event_timestamp": order_items["created_at"],
            "event_weight": event_types.apply(_event_weight_from_type),
        }
    )

    users = pd.DataFrame(
        {
            "user_id": raw_users["id"].astype(str),
            "age_band": _age_band(raw_users["age"]) if "age" in raw_users.columns else "unknown",
            "preferred_style": raw_users.get("gender", pd.Series("", index=raw_users.index)).fillna(""),
            "budget_tier": "unknown",
            "gender": raw_users.get("gender", pd.Series("unknown", index=raw_users.index)).fillna("unknown"),
            "city": raw_users.get("city", pd.Series("", index=raw_users.index)).fillna(""),
            "state": raw_users.get("state", pd.Series("", index=raw_users.index)).fillna(""),
            "country": raw_users.get("country", pd.Series("", index=raw_users.index)).fillna(""),
        }
    )

    stock_by_product = (
        inventory_items[inventory_items["sold_at"].isna()]
        .groupby("product_id")
        .size()
        .rename("stock")
        .reset_index()
    )
    returns_by_product = (
        order_items.assign(is_return=order_items["status"].astype(str).str.lower().eq("returned"))
        .groupby("product_id")["is_return"]
        .mean()
        .rename("return_rate")
        .reset_index()
    )
    inventory = raw_products[["id", "cost", "retail_price"]].rename(columns={"id": "product_id"}).copy()
    inventory = inventory.merge(stock_by_product, on="product_id", how="left")
    inventory = inventory.merge(returns_by_product, on="product_id", how="left")
    retail_price = pd.to_numeric(inventory["retail_price"], errors="coerce").fillna(0.0)
    cost = pd.to_numeric(inventory["cost"], errors="coerce").fillna(0.0)
    inventory["margin"] = ((retail_price - cost) / retail_price.where(retail_price > 0, 1.0)).clip(0.0, 1.0)
    inventory["stock"] = inventory["stock"].fillna(0).astype(int)
    inventory["return_rate"] = inventory["return_rate"].fillna(0.0)
    inventory = inventory[["product_id", "stock", "margin", "return_rate"]]
    inventory["product_id"] = inventory["product_id"].astype(str)

    out_dir.mkdir(parents=True, exist_ok=True)
    products.to_csv(out_dir / "products.csv", index=False)
    interactions.to_csv(out_dir / "interactions.csv", index=False)
    pd.DataFrame(columns=["review_id", "product_id", "rating", "fit_feedback", "review_text"]).to_csv(
        out_dir / "reviews.csv", index=False
    )
    inventory.to_csv(out_dir / "inventory.csv", index=False)
    users.to_csv(out_dir / "users.csv", index=False)
    pd.DataFrame(
        columns=["outfit_id", "product_ids", "occasion_tags", "description", "compatibility_score"]
    ).to_csv(out_dir / "outfits.csv", index=False)
    print(f"Wrote normalized theLook data to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-users", type=int, default=None, help="Number of active users to keep")
    args = parser.parse_args()
    normalize_thelook(args.raw_dir, args.out_dir, args.sample_users)


if __name__ == "__main__":
    main()
