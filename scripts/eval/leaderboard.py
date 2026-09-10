"""Append-only leaderboard for evaluated assistants.

One board holds runs for exactly one dataset+eval-config (encoded in ``EvalSummary.dataset``).
Mixing configs is refused: recall@5 on 5,000 H&M users is not comparable to recall@12 on the sample,
and a lax smoke run would otherwise sit permanently at the top.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .runner import EvalSummary


class DatasetMismatch(Exception):
    """Raised when a run would join a board built on a different dataset/eval-config."""

    def __init__(self, board_dataset: str, run_dataset: str, path: Path):
        self.board_dataset = board_dataset
        self.run_dataset = run_dataset
        super().__init__(
            f"leaderboard {path} scores '{board_dataset}'; this run scored '{run_dataset}'. "
            f"Rows only compare within one dataset+eval-config. Re-run against '{board_dataset}', "
            f"or pass a different --leaderboard path to keep a separate board."
        )


def append_entry(summary: EvalSummary, path: str | Path) -> list[dict[str, Any]]:
    """Append one evaluation to the JSON leaderboard and return all entries."""

    output = Path(path)
    entries: list[dict[str, Any]] = []
    if output.exists():
        loaded = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Leaderboard JSON at {output} must contain an array")
        entries = loaded

    existing = next((e["dataset"] for e in entries if isinstance(e, dict) and e.get("dataset")), None)
    if existing is not None and existing != summary.dataset:
        raise DatasetMismatch(existing, summary.dataset, output)

    next_id = max((e.get("id", 0) for e in entries if isinstance(e, dict)), default=0) + 1
    m = summary.metrics
    entries.append(
        {
            "id": next_id,
            "name": summary.name,
            "version": summary.version,
            "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": summary.dataset,
            "users_evaluated": int(m.get("users_evaluated", 0)),
            "recall_at_k": round(float(m.get("recall_at_k", 0.0)), 6),
            "precision_at_k": round(float(m.get("precision_at_k", 0.0)), 6),
            "map_at_k": round(float(m.get("map_at_k", 0.0)), 6),
            "ndcg_at_k": round(float(m.get("ndcg_at_k", 0.0)), 6),
            "inventory_hit_rate": round(float(m.get("inventory_hit_rate", 0.0)), 6),
            "category_diversity": round(float(m.get("category_diversity", 0.0)), 6),
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)
    output.with_suffix(".html").write_text(render_html(entries), encoding="utf-8")
    return entries


def _esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_html(entries: list[dict[str, Any]]) -> str:
    """Self-contained, theme-aware leaderboard page (works from file://), ranked by recall."""

    ranked = sorted(entries, key=lambda e: e.get("recall_at_k", 0.0), reverse=True)
    dataset = _esc(ranked[0]["dataset"]) if ranked else ""
    body = ""
    for rank, e in enumerate(ranked, start=1):
        lead = " class=lead" if rank == 1 else ""
        body += (
            f"<tr{lead}><td>{rank}</td><td>{_esc(e.get('name',''))}</td>"
            f"<td>{_esc(e.get('version',''))}</td>"
            f"<td class=num>{e.get('recall_at_k',0.0):.4f}</td>"
            f"<td class=num>{e.get('precision_at_k',0.0):.4f}</td>"
            f"<td class=num>{e.get('map_at_k',0.0):.4f}</td>"
            f"<td class=num>{e.get('ndcg_at_k',0.0):.4f}</td>"
            f"<td class=num>{e.get('inventory_hit_rate',0.0):.4f}</td>"
            f"<td class=num>{e.get('category_diversity',0.0):.2f}</td>"
            f"<td class=num>{e.get('users_evaluated',0)}</td>"
            f"<td class=mono>{_esc(e.get('run_at',''))}</td></tr>"
        )
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Recommendation leaderboard</title><style>
:root{{color-scheme:light dark;--bg:#fbfbfa;--card:#fff;--ink:#111;--mut:#777;--line:#e3e3dd;--lead:#2a78d6}}
@media(prefers-color-scheme:dark){{:root{{--bg:#141413;--card:#1e1e1c;--ink:#f2f2ef;--mut:#9a988f;--line:#333;--lead:#4c8fe0}}}}
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:36px 20px 64px}}
.wrap{{max-width:960px;margin:0 auto}}h1{{font-size:21px;margin:0 0 4px}}
.meta{{color:var(--mut);font-size:13px;margin:0 0 22px}}.meta code{{background:var(--card);border:1px solid var(--line);border-radius:5px;padding:1px 6px}}
table{{border-collapse:collapse;width:100%;font-size:13px;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}}
th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);padding:10px;border-bottom:1px solid var(--line)}}
td{{padding:9px 10px;border-bottom:1px solid var(--line)}}tr:last-child td{{border-bottom:0}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}.mono{{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--mut)}}
tr.lead td{{font-weight:650}}tr.lead td:first-child{{box-shadow:inset 3px 0 0 var(--lead)}}
</style></head><body><div class=wrap>
<h1>Recommendation leaderboard</h1>
<p class=meta><code>{dataset}</code> · {len(ranked)} run(s) · ranked by recall</p>
<table><thead><tr><th>#</th><th>name</th><th>ver</th><th class=num>recall</th><th class=num>prec</th>
<th class=num>MAP</th><th class=num>NDCG</th><th class=num>in-stock</th><th class=num>diversity</th>
<th class=num>users</th><th>run at</th></tr></thead><tbody>{body}</tbody></table>
</div></body></html>"""


def format_leaderboard(entries: list[dict[str, Any]]) -> str:
    """Plain-text board, ranked by recall_at_k descending."""

    if not entries:
        return "(leaderboard empty)"
    ranked = sorted(entries, key=lambda e: e.get("recall_at_k", 0.0), reverse=True)
    header = f"{'#':>2}  {'name':<18} {'ver':<5} {'recall':>9} {'map':>9} {'ndcg':>9}  users"
    lines = [header, "-" * len(header)]
    for rank, e in enumerate(ranked, start=1):
        lines.append(
            f"{rank:>2}  {str(e.get('name','')):<18} {str(e.get('version','')):<5} "
            f"{e.get('recall_at_k',0.0):>9.4f} {e.get('map_at_k',0.0):>9.4f} "
            f"{e.get('ndcg_at_k',0.0):>9.4f}  {e.get('users_evaluated',0)}"
        )
    return "\n".join(lines)
