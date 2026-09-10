from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CatalogBundle:
    """All normalized tables used by the assistant."""

    products: pd.DataFrame
    interactions: pd.DataFrame
    reviews: pd.DataFrame
    inventory: pd.DataFrame
    users: pd.DataFrame
    outfits: pd.DataFrame
    data_dir: Path


@dataclass(frozen=True)
class ParsedIntent:
    raw_query: str
    occasion_terms: list[str] = field(default_factory=list)
    color_terms: list[str] = field(default_factory=list)
    category_terms: list[str] = field(default_factory=list)
    style_terms: list[str] = field(default_factory=list)
    budget_max: float | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None

    def as_retrieval_text(self) -> str:
        parts = [self.raw_query]
        parts.extend(self.occasion_terms)
        parts.extend(self.color_terms)
        parts.extend(self.category_terms)
        parts.extend(self.style_terms)
        if self.budget_max is not None:
            parts.append(f"budget under {self.budget_max:g}")
        return " ".join(str(x) for x in parts if x)


@dataclass(frozen=True)
class RecommendationEvidence:
    source: str
    source_id: str
    field: str
    value: str


@dataclass(frozen=True)
class RecommendationResult:
    product_id: str
    rank: int
    score: float
    retrieval_score: float
    product: dict[str, Any]
    why: list[str]
    evidence: list[RecommendationEvidence]
    business_signals: dict[str, Any]


