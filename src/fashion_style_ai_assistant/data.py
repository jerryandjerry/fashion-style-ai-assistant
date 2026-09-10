from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import CatalogBundle


PRODUCT_REQUIRED = {
    "product_id",
    "name",
    "category",
    "color",
    "style_tags",
    "occasion_tags",
    "season",
    "price",
    "description",
}
INTERACTION_REQUIRED = {"user_id", "product_id", "event_type", "event_timestamp"}
REVIEW_REQUIRED = {"review_id", "product_id", "rating", "fit_feedback", "review_text"}
INVENTORY_REQUIRED = {"product_id", "stock", "margin", "return_rate"}
USER_REQUIRED = {"user_id"}
OUTFIT_REQUIRED = {"outfit_id", "product_ids", "occasion_tags", "description", "compatibility_score"}

def read_csv_checked(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    df = pd.read_csv(path)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def _read_optional_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=sorted(required_columns))
    return read_csv_checked(path, required_columns)


def load_catalog_bundle(data_dir: str | Path = "data/sample") -> CatalogBundle:
    """Load normalized catalog tables from a directory."""

    root = Path(data_dir)
    products = read_csv_checked(root / "products.csv", PRODUCT_REQUIRED)
    interactions = read_csv_checked(root / "interactions.csv", INTERACTION_REQUIRED)
    reviews = _read_optional_csv(root / "reviews.csv", REVIEW_REQUIRED)
    inventory = _read_optional_csv(root / "inventory.csv", INVENTORY_REQUIRED)
    users = _read_optional_csv(root / "users.csv", USER_REQUIRED)
    outfits = _read_optional_csv(root / "outfits.csv", OUTFIT_REQUIRED)

    products = products.copy()
    products["product_id"] = products["product_id"].astype(str)
    products["price"] = pd.to_numeric(products["price"], errors="coerce").fillna(0.0)

    interactions = interactions.copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["product_id"] = interactions["product_id"].astype(str)
    interactions["event_timestamp"] = pd.to_datetime(
        interactions["event_timestamp"], errors="coerce"
    )
    if "event_weight" not in interactions.columns:
        interactions["event_weight"] = interactions["event_type"].map(
            {"view": 0.2, "cart": 0.6, "favorite": 0.8, "purchase": 1.0, "return": -1.0}
        ).fillna(0.1)

    if not inventory.empty:
        inventory = inventory.copy()
        inventory["product_id"] = inventory["product_id"].astype(str)
        inventory["stock"] = pd.to_numeric(inventory["stock"], errors="coerce").fillna(0)
        inventory["margin"] = pd.to_numeric(inventory["margin"], errors="coerce").fillna(0.0)
        inventory["return_rate"] = pd.to_numeric(
            inventory["return_rate"], errors="coerce"
        ).fillna(0.0)

    return CatalogBundle(
        products=products,
        interactions=interactions,
        reviews=reviews,
        inventory=inventory,
        users=users,
        outfits=outfits,
        data_dir=root,
    )

# split interaction.csv into train/test, use last purchase as the test data
def _last_purchase_holdout(bundle: CatalogBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a tiny time-based holdout: last purchase per user is test data."""
    interactions = bundle.interactions.sort_values("event_timestamp").copy()
    purchases = interactions[interactions["event_type"].astype(str).str.lower() == "purchase"].copy()
    if purchases.empty:
        return interactions, purchases
    test_idx = purchases.groupby("user_id")["event_timestamp"].idxmax()
    test = purchases.loc[test_idx].copy()
    train = interactions.drop(index=test_idx).copy()
    return train, test


def real_ids(s: pd.Series) -> pd.Series:
    """Restore the catalog's real 10-digit ids: int inference strips leading zeros, but the
    eval compares ids exactly as read from CSV. Non-numeric ids (data/sample) pass through."""

    s = s.astype(str)
    numeric = s.str.isdigit()
    return s.where(~numeric, s.str.zfill(10))


def make_product_text(products: pd.DataFrame) -> pd.Series:
    """Create a single searchable text field for retrieval."""

    columns = [
        "name",
        "category",
        "color",
        "style_tags",
        "occasion_tags",
        "season",
        "brand",
        "material",
        "description",
    ]
    present = [col for col in columns if col in products.columns]
    return products[present].fillna("").astype(str).agg(" ".join, axis=1)
