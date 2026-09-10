"""Standalone per-run evaluation report."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .runner import EvalSummary


def write_report(summary: EvalSummary, out_dir: str | Path = "artifacts") -> Path:
    """Write this run's standalone JSON report to ``artifacts/eval_report_<name>_<dataset>.json``."""

    slug = re.sub(r"[^0-9a-zA-Z]+", "-", f"{summary.name}_{summary.dataset}").strip("-")
    path = Path(out_dir) / f"eval_report_{slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": summary.name,
        "version": summary.version,
        "dataset": summary.dataset,
        "metrics": summary.metrics,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
