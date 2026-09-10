.PHONY: install test demo evaluate lint

install:
	uv sync --all-extras

test:
	uv run pytest -q

demo:
	uv run fashion-assistant demo --user-id u001 --query "I need a relaxed light outfit for a beach weekend under 90 dollars"

evaluate:
	uv run python scripts/eval/evaluate.py --config config.yaml --data-dir data/processed/hm

lint:
	uv run ruff check src tests scripts
