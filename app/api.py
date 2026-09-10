from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fashion_style_ai_assistant.config import load_config  # noqa: E402
from fashion_style_ai_assistant.baseline.pipeline import FashionAssistant  # noqa: E402


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=3)
    user_id: str = "anonymous"
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=30, ge=1, le=200)
    config_path: str = "config.yaml"
    data_dir: str | None = None


app = FastAPI(title="Fashion Style AI Assistant API", version="0.1.0")


@lru_cache(maxsize=4)
def get_assistant(config_path: str, data_dir: str | None) -> FashionAssistant:
    config = load_config(config_path)
    resolved_data_dir = data_dir or str(config.data_dir)
    return FashionAssistant.from_data_dir(
        resolved_data_dir,
        max_features=config.max_features,
        ranking_config=config.ranking,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend")
def recommend(request: RecommendRequest) -> dict[str, Any]:
    assistant = get_assistant(request.config_path, request.data_dir)
    return assistant.recommend(query=request.query, user_id=request.user_id, top_k=request.top_k, candidate_k=request.candidate_k)
