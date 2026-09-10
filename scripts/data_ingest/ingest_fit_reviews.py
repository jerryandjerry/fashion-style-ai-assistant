from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_json_lines_or_array(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def normalize_fit_reviews(reviews_json: Path, out_dir: Path) -> None:
    rows = _read_json_lines_or_array(reviews_json)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError(f"No rows found in {reviews_json}")

    product_id_col = "item_id" if "item_id" in raw.columns else "product_id"
    reviews = pd.DataFrame(
        {
            "review_id": [f"fit_review_{i}" for i in range(len(raw))],
            "product_id": raw[product_id_col].astype(str),
            "rating": pd.to_numeric(raw.get("rating", 0), errors="coerce").fillna(0),
            "fit_feedback": raw.get("fit", "unknown"),
            "review_text": raw.get("review_text", raw.get("review_summary", "")),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(out_dir / "reviews.csv", index=False)
    print(f"Wrote fit reviews to {out_dir / 'reviews.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    normalize_fit_reviews(args.reviews_json, args.out_dir)


if __name__ == "__main__":
    main()
