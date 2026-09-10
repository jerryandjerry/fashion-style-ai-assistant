from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ARTICLE_COLS = {
    "article_id": "product_id",
    "prod_name": "name",
    "product_type_name": "category",
    "colour_group_name": "color",
    "index_name": "style_tags",
    "garment_group_name": "occasion_tags",
    "detail_desc": "description",
}


def normalize_hm(raw_dir: Path, out_dir: Path, sample_size: int | None = None) -> None:
    articles_path = raw_dir / "articles.csv"
    customers_path = raw_dir / "customers.csv"
    transactions_path = raw_dir / "transactions_train.csv"

    for path in [articles_path, customers_path, transactions_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing H&M file: {path}")

    articles = pd.read_csv(articles_path, dtype={"article_id": str})

    products = articles.rename(columns=ARTICLE_COLS)
    products = products[list(ARTICLE_COLS.values())].copy()
    products["product_id"] = products["product_id"].astype(str)
    products["name"] = products["name"].fillna("Unknown product")
    products["category"] = products["category"].fillna("unknown")
    products["style_tags"] = products["style_tags"].fillna("")
    products["occasion_tags"] = products["occasion_tags"].fillna("")
    products["season"] = "unknown"

    # Fold source distinctions that have no column of their own into their closest schema field, so
    # genuinely different garments stay distinct: fine shade -> color, pattern -> description.
    shade = articles["perceived_colour_value_name"].fillna("").astype(str).str.strip()
    color = products["color"].fillna("").astype(str).str.strip()
    products["color"] = (color + " " + shade).str.strip().replace("", "unknown")

    pattern = articles["graphical_appearance_name"].fillna("").astype(str).str.strip()
    pattern = pattern.where(~pattern.str.lower().isin(["", "unknown"]), "")
    desc = products["description"].fillna("").astype(str).str.strip()
    suffix = (" Pattern: " + pattern + ".").where(pattern.ne(""), "")
    products["description"] = (desc + suffix).str.strip()

    # Product identity is defined by OUR schema: rows identical on every descriptive column ARE the
    # same product. Collapse each such group to one canonical product_id (the smallest) and remap all
    # sales onto it. Price is a derived market attribute, not identity, so it is excluded from the key
    # and recomputed as the median across the merged product's transactions below.
    identity_cols = ["name", "category", "color", "style_tags", "occasion_tags", "description", "season"]
    identity = products[identity_cols].astype(str).agg("\x1f".join, axis=1)
    canonical = products.groupby(identity)["product_id"].transform("min")
    canonical_map = dict(zip(products["product_id"], canonical))
    products["product_id"] = canonical
    products = products.drop_duplicates("product_id").reset_index(drop=True)

    transactions = pd.read_csv(transactions_path, dtype={"article_id": str, "customer_id": str})
    if sample_size:
        # Keep a reproducible sample of active users, rather than random rows only.
        user_counts = transactions["customer_id"].value_counts().head(sample_size)
        transactions = transactions[transactions["customer_id"].isin(user_counts.index)].copy()

    # Remap sales onto canonical article_ids so duplicate SKUs' purchases consolidate.
    transactions["article_id"] = transactions["article_id"].map(canonical_map).fillna(transactions["article_id"])

    price_by_product = transactions.groupby("article_id")["price"].median().reset_index()
    price_by_product["price"] = price_by_product["price"].astype(float) * 1000.0
    products = products.merge(price_by_product, left_on="product_id", right_on="article_id", how="left")
    products["price"] = products["price"].fillna(products["price"].median()).round(2)
    products = products.drop(columns=["article_id"], errors="ignore")

    # Keep sales_channel_id and the per-transaction price in the pool -- the behavioural scheme's
    # features can use them; schemes that don't just ignore the extra columns.
    interactions = transactions.rename(
        columns={"customer_id": "user_id", "article_id": "product_id", "t_dat": "event_timestamp"}
    )[["user_id", "product_id", "event_timestamp", "price", "sales_channel_id"]]
    interactions["event_type"] = "purchase"
    interactions["event_weight"] = 1.0

    customers = pd.read_csv(customers_path, dtype={"customer_id": str})
    users = customers.rename(columns={"customer_id": "user_id"})
    if "age" in users.columns:
        users["age_band"] = pd.cut(
            users["age"], bins=[0, 17, 24, 34, 44, 54, 64, 120],
            labels=["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        ).astype(str)
    else:
        users["age_band"] = "unknown"
    # Derive preference features from each customer's own purchase history.
    # preferred_style: their top-3 most-purchased style_tags -- computed via a map + a vectorized
    # top-3 (no 31M-row merge, no per-group Python) so it scales to the full transaction log.
    style_map = products.set_index("product_id")["style_tags"]
    style_pairs = pd.DataFrame(
        {
            "user_id": transactions["customer_id"].to_numpy(),
            "style": transactions["article_id"].map(style_map).to_numpy(),
        }
    )
    style_pairs = style_pairs[style_pairs["style"].fillna("").astype(str).str.strip() != ""]
    counts = style_pairs.groupby(["user_id", "style"], sort=False).size().reset_index(name="n")
    counts = counts.sort_values(["user_id", "n"], ascending=[True, False])
    counts = counts[counts.groupby("user_id", sort=False).cumcount() < 3]
    preferred_style = counts.groupby("user_id", sort=False)["style"].agg(", ".join)

    median_price = transactions.groupby("customer_id")["price"].median()
    budget_tier = pd.qcut(median_price.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)

    users["preferred_style"] = users["user_id"].map(preferred_style).fillna("")
    users["budget_tier"] = users["user_id"].map(budget_tier).fillna("unknown")
    users = users[["user_id", "age_band", "preferred_style", "budget_tier"]]

    inventory = products[["product_id"]].copy()
    rng = np.random.default_rng(42)
    inventory["stock"] = rng.integers(0, 80, size=len(inventory))
    inventory["margin"] = rng.uniform(0.25, 0.55, size=len(inventory)).round(3)
    inventory["return_rate"] = rng.uniform(0.03, 0.20, size=len(inventory)).round(3)

    out_dir.mkdir(parents=True, exist_ok=True)
    products.to_csv(out_dir / "products.csv", index=False)
    interactions.to_csv(out_dir / "interactions.csv", index=False)
    users.to_csv(out_dir / "users.csv", index=False)
    inventory.to_csv(out_dir / "inventory.csv", index=False)
    pd.DataFrame(columns=["review_id", "product_id", "rating", "fit_feedback", "review_text"]).to_csv(
        out_dir / "reviews.csv", index=False
    )
    pd.DataFrame(
        columns=["outfit_id", "product_ids", "occasion_tags", "description", "compatibility_score"]
    ).to_csv(out_dir / "outfits.csv", index=False)
    print(f"Wrote normalized H&M data to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=None, help="Number of active users to keep")
    args = parser.parse_args()
    normalize_hm(args.raw_dir, args.out_dir, args.sample_size)


if __name__ == "__main__":
    main()
