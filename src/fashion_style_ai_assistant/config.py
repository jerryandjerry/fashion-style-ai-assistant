from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RankingConfig:
    retrieval_weight: float = 0.45
    intent_match_weight: float = 0.15
    preference_weight: float = 0.15
    price_weight: float = 0.10
    inventory_weight: float = 0.08
    margin_weight: float = 0.04
    return_weight: float = 0.03
    out_of_stock_penalty: float = 0.35


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = Path("data/sample")
    default_top_k: int = 5
    candidate_k: int = 30
    retrieval_method: str = "tfidf"
    max_features: int = 5000
    evaluation_sample_users: int | None = 100
    evaluation_sample_seed: int = 42
    ranking: RankingConfig = RankingConfig()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load a YAML config file, falling back to sensible defaults."""

    if config_path is None:
        return AppConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    retrieval = raw.get("retrieval", {}) or {}
    evaluation = raw.get("evaluation", {}) or {}
    ranking = raw.get("ranking", {}) or {}
    sample_users = evaluation.get("sample_users", AppConfig.evaluation_sample_users)
    return AppConfig(
        data_dir=Path(raw.get("data_dir", AppConfig.data_dir)),
        default_top_k=int(raw.get("default_top_k", AppConfig.default_top_k)),
        candidate_k=int(raw.get("candidate_k", AppConfig.candidate_k)),
        retrieval_method=str(retrieval.get("method", AppConfig.retrieval_method)),
        max_features=int(retrieval.get("max_features", AppConfig.max_features)),
        evaluation_sample_users=int(sample_users) if sample_users is not None else None,
        evaluation_sample_seed=int(evaluation.get("sample_seed", AppConfig.evaluation_sample_seed)),
        ranking=RankingConfig(
            retrieval_weight=float(ranking.get("retrieval_weight", RankingConfig.retrieval_weight)),
            intent_match_weight=float(ranking.get("intent_match_weight", RankingConfig.intent_match_weight)),
            preference_weight=float(ranking.get("preference_weight", RankingConfig.preference_weight)),
            price_weight=float(ranking.get("price_weight", RankingConfig.price_weight)),
            inventory_weight=float(ranking.get("inventory_weight", RankingConfig.inventory_weight)),
            margin_weight=float(ranking.get("margin_weight", RankingConfig.margin_weight)),
            return_weight=float(
                ranking.get("return_weight", ranking.get("return_risk_weight", RankingConfig.return_weight))
            ),
            out_of_stock_penalty=float(ranking.get("out_of_stock_penalty", RankingConfig.out_of_stock_penalty)),
        ),
    )
