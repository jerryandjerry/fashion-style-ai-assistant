from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .rag import response_to_markdown

# Assistant folders under this package that expose a build() factory.
ASSISTANTS = ("baseline", "behavioural")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fashion Style AI Assistant CLI")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML runtime config")
    parser.add_argument("--data-dir", default=None, help="Override data directory from config")

    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run one recommendation query")
    demo.add_argument(
        "--assistant",
        default="baseline",
        choices=ASSISTANTS,
        help="Which assistant answers (each is a folder with a build() factory)",
    )
    demo.add_argument(
        "--query",
        default="I need a relaxed light outfit for a beach weekend under 90 dollars",
        help="Natural-language style request",
    )
    demo.add_argument("--user-id", default="u001", help="User ID from users/interactions tables")
    demo.add_argument("--top-k", type=int, default=None, help="Override recommendation count from config")
    demo.add_argument("--candidate-k", type=int, default=None, help="Override retrieval candidate pool size from config")
    demo.add_argument("--json", action="store_true", help="Print raw JSON instead of markdown")
    demo.add_argument("--output", default=None, help="Optional file path for markdown/JSON output")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    data_dir = Path(args.data_dir) if args.data_dir else config.data_dir
    top_k = args.top_k if args.top_k is not None else config.default_top_k

    if args.command == "demo":
        candidate_k = args.candidate_k if args.candidate_k is not None else config.candidate_k
        # dispatch to the chosen assistant's build() factory (same one the eval imports)
        module = importlib.import_module(f"fashion_style_ai_assistant.{args.assistant}")
        assistant = module.build(data_dir=data_dir)
        response = assistant.recommend(query=args.query, user_id=args.user_id, top_k=top_k, candidate_k=candidate_k)
        if args.json:
            output = json.dumps(response, indent=2, ensure_ascii=False)
        else:
            output = response_to_markdown(response)
            metrics = response.get("business_metrics", {})
            interpretation = response.get("business_interpretation", {})
            output += "\n## Business metrics\n\n"
            output += json.dumps(metrics, indent=2, ensure_ascii=False)
            output += "\n\n## Business interpretation\n\n"
            output += json.dumps(interpretation, indent=2, ensure_ascii=False)
            output += "\n"
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
        print(output)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
