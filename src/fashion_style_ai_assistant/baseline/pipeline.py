from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..business import estimate_business_outcome, summarize_business_metrics
from ..config import AppConfig, RankingConfig
from ..data import load_catalog_bundle
from ..intent import parse_intent
from ..rag import build_item_explanation, render_response
from .ranking import _split_tags, rank_candidates
from .retrieval import TfidfCatalogRetriever
from ..schemas import CatalogBundle

class FashionAssistant:
    """End-to-end orchestrator for the assistant."""

    def __init__(
        self,
        bundle: CatalogBundle,
        max_features: int = 5000,
        ranking_config: RankingConfig | None = None,
    ):
        self.bundle = bundle # dataclass, provided by class method below
        self.ranking_config = ranking_config or RankingConfig()
        # TF-IDF retriever using only the product catalog
        self.retriever = TfidfCatalogRetriever(bundle.products, max_features=max_features)
        # live mode: no cached profiles; rank_candidates builds them from interactions per call
        self._affinity = None
        self._user_meta = None

    @classmethod # class method doesn't need instantiation
    def from_data_dir(
        cls,
        data_dir: str | Path = "data/sample",
        max_features: int = 5000,
        ranking_config: RankingConfig | None = None,
    ) -> "FashionAssistant":
        return cls(
            load_catalog_bundle(data_dir),
            max_features=max_features,
            ranking_config=ranking_config,
        )

    @classmethod
    def from_config(cls, config: AppConfig) -> "FashionAssistant":
        return cls.from_data_dir(
            config.data_dir,
            max_features=config.max_features,
            ranking_config=config.ranking,
        )

    @classmethod
    def from_weights(cls, weights: dict) -> "FashionAssistant":
        """Build from the trained artifact (models/baseline/weights.pkl) -- no raw interactions.

        The retriever comes fitted, user profiles come precomputed; the bundle carries only
        what serving still reads (products for ranking, reviews for evidence, inventory).
        """

        obj = cls.__new__(cls)
        obj.retriever = weights["retriever"]
        obj.ranking_config = weights["ranking_config"]
        obj.bundle = CatalogBundle(
            products=obj.retriever.products,
            interactions=pd.DataFrame(),
            reviews=weights["reviews"],
            inventory=weights["inventory"],
            users=pd.DataFrame(),
            outfits=pd.DataFrame(),
            data_dir=Path("models/baseline"),
        )
        obj._affinity = weights["affinity"]
        obj._user_meta = weights["user_meta"]
        return obj

    def _profile_for(self, user_id: str) -> dict[str, object] | None:
        """Cached profile lookup (artifact mode); None in live mode -> built from interactions."""

        if self._affinity is None:
            return None
        uid = str(user_id)
        profile: dict[str, object] = {
            "known_user": False,
            "category_affinity": {},
            "color_affinity": {},
            "style_affinity": {},
            "preferred_style_terms": set(),
            "budget_tier": None,
        }
        if self._user_meta is not None and uid in self._user_meta.index:
            meta = self._user_meta.loc[uid]
            profile["known_user"] = True
            profile["preferred_style_terms"] = _split_tags(meta.get("preferred_style", ""))
            profile["budget_tier"] = meta.get("budget_tier")
        if uid in self._affinity.index:
            rows = self._affinity.loc[[uid]]
            profile["known_user"] = True
            for kind, field in ((0, "category_affinity"), (1, "color_affinity"), (2, "style_affinity")):
                part = rows[rows["kind"] == kind]
                if not part.empty:
                    profile[field] = dict(zip(part["key"].astype(str), part["value"].astype(float)))
        return profile
    
    def recommend(
        self,
        query: str,
        user_id: str = "anonymous",
        top_k: int = 5,
        candidate_k: int = 30,
    ) -> dict[str, Any]:
        # intent -> retreival -> rerank -> respond
        intent = parse_intent(query)
        candidates = self.retriever.search(intent, top_k=candidate_k)
        ranked = rank_candidates(
            candidates=candidates,
            products=self.bundle.products,
            interactions=self.bundle.interactions,
            inventory=self.bundle.inventory,
            users=self.bundle.users,
            user_id=user_id,
            intent=intent,
            ranking_config=self.ranking_config,
            profile=self._profile_for(user_id),
        )
        top = ranked.head(top_k).copy()
        # the assistant supplies the evidence (RecommendationResult); shared rag only renders
        items = [build_item_explanation(row, self.bundle.reviews) for _, row in top.iterrows()]
        response = render_response(
            query=query,
            user_id=user_id,
            items=items,
            clarification_question=intent.clarification_question,
        )
        response["parsed_intent"] = intent.__dict__
        response["business_metrics"] = summarize_business_metrics(top)
        response["business_interpretation"] = estimate_business_outcome(top)
        return response

    def recommend_dataframe(
        self,
        query: str,
        user_id: str = "anonymous",
        top_k: int = 5,
        candidate_k: int = 30,
    ) -> pd.DataFrame:
        intent = parse_intent(query)
        candidates = self.retriever.search(intent, top_k=candidate_k)
        ranked = rank_candidates(
            candidates=candidates,
            products=self.bundle.products,
            interactions=self.bundle.interactions,
            inventory=self.bundle.inventory,
            users=self.bundle.users,
            user_id=user_id,
            intent=intent,
            ranking_config=self.ranking_config,
            profile=self._profile_for(user_id),
        )
        return ranked.head(top_k).copy()
