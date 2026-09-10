"""Run the demo end-to-end on a dataset and write docs/sample demo result.md.

Only two inputs: --data-dir (the data source) and --config. It auto-picks the
#1 user in that dataset (so personalization always applies, on any data), uses a
fixed query and top_k=1, and takes everything else from config.yaml. One output:
the markdown file.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fashion_style_ai_assistant.cli import main as cli_main  # noqa: E402
from fashion_style_ai_assistant.config import load_config  # noqa: E402

QUERY = "I need a relaxed light outfit for a beach weekend under 90 dollars"
OUTPUT = ROOT / "docs" / "sample demo result.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate docs/sample demo result.md for a dataset.")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--data-dir", default=None, help="Dataset dir (default: from config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else config.data_dir

    # Pick the #1 user in this dataset (first interaction row) so personalization applies.
    user_ids = pd.read_csv(data_dir / "interactions.csv", usecols=["user_id"], nrows=1)["user_id"]
    user_id = str(user_ids.iloc[0]) if not user_ids.empty else "anonymous"

    # Reuse the CLI so formatting/output stays identical; only the data source varies.
    cli_main(
        [
            "--config", args.config,
            "--data-dir", str(data_dir),
            "demo",
            "--user-id", user_id,
            "--query", QUERY,
            "--top-k", "1",
            "--output", str(OUTPUT),
        ]
    )


if __name__ == "__main__":
    main()
