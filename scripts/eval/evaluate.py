"""Score one assistant and record it on the leaderboard.

The single entry point for every submission -- see README.md.

Usage:
    python scripts/eval/evaluate.py --config config.yaml \
        --data-dir data/processed/hm --assistant mypkg.mymodel:build
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval.leaderboard import (  # noqa: E402
    DatasetMismatch,
    append_entry,
    format_leaderboard,
)
from eval.report import write_report  # noqa: E402
from eval.runner import EvalSummary, evaluate_assistant  # noqa: E402
from fashion_style_ai_assistant.config import load_config  # noqa: E402


def load_assistant(spec: str) -> Any:
    """Import ``module:factory`` and call it with no arguments."""

    module_name, sep, factory_name = spec.partition(":")
    if not sep or not module_name or not factory_name:
        raise SystemExit(f"--assistant must be 'module:factory', got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"Cannot import {module_name!r}: {exc}") from exc
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise SystemExit(f"{module_name!r} has no callable {factory_name!r}")
    return factory()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an assistant and record the result")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML runtime config")
    parser.add_argument("--data-dir", default=None, help="Dataset dir containing test/")
    parser.add_argument(
        "--assistant",
        required=True,
        help="Zero-argument factory as 'module:factory', returning the object to evaluate",
    )
    parser.add_argument(
        "--leaderboard",
        default="artifacts/leaderboard.json",
        help="Leaderboard path; pass 'none' to score without recording",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else config.data_dir
    top_k = config.default_top_k
    sample_users = config.evaluation_sample_users
    sample_seed = config.evaluation_sample_seed

    test_dir = Path(data_dir) / "test"
    if not test_dir.is_dir():
        raise SystemExit(
            f"No test split at {test_dir}. Build one first:\n"
            f"  python scripts/data_ingest/split_hm.py --data-dir {data_dir}"
        )
    # Ids are read as strings. Inferred dtypes turn "0108775015" into 108775015, which silently
    # breaks every comparison against ids a submitter read from the same CSV.
    ids = {"user_id": str, "product_id": str}
    cases = pd.read_csv(test_dir / "interactions.csv", dtype=ids)
    products = pd.read_csv(test_dir / "products.csv", dtype=ids)
    inventory_path = test_dir / "inventory.csv"
    inventory = pd.read_csv(inventory_path, dtype=ids) if inventory_path.exists() else pd.DataFrame()

    assistant = load_assistant(args.assistant)
    metrics = evaluate_assistant(
        assistant,
        cases,
        products,
        inventory,
        top_k=top_k,
        sample_users=sample_users,
        sample_seed=sample_seed,
    )
    sampling = "all" if sample_users is None else f"{sample_users}|seed={sample_seed}"
    dataset = f"{Path(data_dir).name}|top_k={top_k}|cases={sampling}"
    summary = EvalSummary(
        name=getattr(assistant, "name", args.assistant),
        version=getattr(assistant, "version", "0"),
        dataset=dataset,
        metrics=metrics,
    )
    report_path = write_report(summary)

    entries = None
    if str(args.leaderboard).strip().lower() != "none":
        try:
            entries = append_entry(summary, args.leaderboard)
        except DatasetMismatch as exc:
            print(f"warning: not recorded on the leaderboard. {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        print(f"Offline metrics -- '{summary.name}' v{summary.version}")
        print("=" * 40)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        print(f"\nReport: {report_path}")
        if entries is not None:
            print(f"Leaderboard: {args.leaderboard}")
            print(format_leaderboard(entries))


if __name__ == "__main__":
    main()
