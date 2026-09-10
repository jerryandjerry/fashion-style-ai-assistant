from pathlib import Path

import pandas as pd

from fashion_style_ai_assistant.config import RankingConfig, load_config
from fashion_style_ai_assistant.baseline.ranking import rank_candidates
from fashion_style_ai_assistant.schemas import ParsedIntent


def test_load_config_reads_ranking_weights(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data_dir: custom/data
default_top_k: 7
retrieval:
  method: tfidf
  max_features: 123
evaluation:
  sample_users: 11
  sample_seed: 99
ranking:
  retrieval_weight: 0.5
  intent_match_weight: 0.1
  preference_weight: 0.1
  price_weight: 0.1
  inventory_weight: 0.1
  margin_weight: 0.05
  return_weight: 0.04
  out_of_stock_penalty: 0.2
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data_dir == Path("custom/data")
    assert config.default_top_k == 7
    assert config.max_features == 123
    assert config.evaluation_sample_users == 11
    assert config.evaluation_sample_seed == 99
    assert config.ranking.margin_weight == 0.05
    assert config.ranking.out_of_stock_penalty == 0.2


def test_load_config_uses_default_evaluation_sampling():
    config = load_config(None)
    assert config.evaluation_sample_users == 100
    assert config.evaluation_sample_seed == 42


def test_rank_candidates_uses_ranking_config():
    products = pd.DataFrame(
        [
            {"product_id": "low", "category": "shirt", "color": "white", "style_tags": "", "price": 50},
            {"product_id": "high", "category": "shirt", "color": "white", "style_tags": "", "price": 50},
        ]
    )
    candidates = products.copy()
    candidates["retrieval_score"] = [1.0, 1.0]
    inventory = pd.DataFrame(
        [
            {"product_id": "low", "stock": 1, "margin": 0.1, "return_rate": 0.1},
            {"product_id": "high", "stock": 1, "margin": 0.9, "return_rate": 0.1},
        ]
    )

    ranked = rank_candidates(
        candidates=candidates,
        products=products,
        interactions=pd.DataFrame(columns=["user_id", "product_id", "event_type"]),
        inventory=inventory,
        users=pd.DataFrame(columns=["user_id"]),
        user_id="u001",
        intent=ParsedIntent(raw_query="shirt"),
        ranking_config=RankingConfig(
            retrieval_weight=0.0,
            intent_match_weight=0.0,
            preference_weight=0.0,
            price_weight=0.0,
            inventory_weight=0.0,
            margin_weight=1.0,
            return_weight=0.0,
            out_of_stock_penalty=0.0,
        ),
    )

    assert ranked.iloc[0]["product_id"] == "high"
