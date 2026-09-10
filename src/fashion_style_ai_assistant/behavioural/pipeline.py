"""Behavioural v4 assistant: the faithful Kaggle 1st-place port, serving side.

intent -> candidate-store lookup -> query features -> 12-model ensemble -> respond.
Emits the same ``list[RecommendationResult]`` every assistant emits; shared rag/business
render it (see STRUCTURE.md contract).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..business import estimate_business_outcome, summarize_business_metrics
from ..intent import parse_intent
from ..rag import render_response
from ..schemas import RecommendationEvidence, RecommendationResult
from .ranking import cust_blend, model_frame, query_codes, ranked_list
from .retrieval import CandidateStore


class FashionAssistant:
    """End-to-end v4 assistant over the trained weights + candidate store."""

    def __init__(self, weights: dict, stores: dict,
                 products: pd.DataFrame, inventory: pd.DataFrame, query_serve: dict | None = None):
        self.w = weights
        self.name = weights["name"]
        self.version = weights["version"]
        self._stores = stores
        self._qs = query_serve
        # hoisted once: rebuilding this 24M-entry Series per question cost ~2s/case
        self._ui = (pd.Series(query_serve["ui_val"], index=query_serve["ui_key"])
                    if query_serve is not None else None)
        self._u_map = {u: i for i, u in enumerate(weights["users_idx"])}
        self._items = weights["items_idx"]
        self._products_by_id = products.set_index(products["product_id"].astype(str))
        self._inventory_by_id = (
            inventory.set_index(inventory["product_id"].astype(str))
            if not inventory.empty else pd.DataFrame()
        )

    # ------------------------------------------------------------ query strategy assembly
    def _query_rows(self, uidx: int | None, qc: int, qs_: int, qo: int) -> pd.DataFrame:
        """Assemble candidate feature rows for the query strategy on the fly (it has no
        store -- candidates depend on the query). Item/user features come from frames built
        through the SAME join path as training; pair features computed here."""

        qsv = self._qs
        if qsv is None or (qc < 0 and qs_ < 0 and qo < 0):
            return pd.DataFrame()
        key = (qc, qs_, qo)
        recent = qsv["triples"]["recent"].get(key)
        alltime = qsv["triples"]["alltime"].get(key)
        if recent is None and alltime is None:
            return pd.DataFrame()
        recent = recent if recent is not None else np.array([], dtype=np.int32)
        alltime = alltime if alltime is not None else np.array([], dtype=np.int32)
        cand = pd.unique(np.concatenate([recent, alltime]))

        rows = qsv["item_frame"].reindex(cand).copy()
        rows["iidx"] = cand
        rows["uidx"] = -1 if uidx is None else int(uidx)
        # user-side features (cold users keep neutral defaults)
        for c in qsv["user_cols"]:
            rows[c] = 0.0
        if uidx is not None and int(uidx) in qsv["user_frame"].index:
            u = qsv["user_frame"].loc[int(uidx)]
            for c in qsv["user_cols"]:
                rows[c] = float(u[c])
        # pair features
        keys = (np.int64(-1 if uidx is None else int(uidx)) << 32) | cand.astype(np.int64)
        ui = self._ui.reindex(keys).fillna(0).to_numpy(np.float32)
        rows["ui_sale"] = ui
        rows["ui_sale_ratio"] = ui / (rows["i_full_sale"].to_numpy(np.float32) + 1e-6)
        rows["price_ratio"] = rows["item_price"].to_numpy(np.float32) / (
            rows["user_mean_price"].to_numpy(np.float32) + 1e-6)
        u_emb = (qsv["w2v_user"][int(uidx)] if uidx is not None and int(uidx) < len(qsv["w2v_user"])
                 else np.ones(qsv["w2v_item"].shape[1], np.float32) / qsv["w2v_item"].shape[1])
        rows["wv_similarity"] = (qsv["w2v_item"][cand] @ u_emb).astype(np.float32)
        if "prone_user" in qsv:  # v2.0 pair-level features (item_frame bakes them with uidx=0)
            pu, pi = qsv["prone_user"], qsv["prone_item"]
            p_emb = (pu[int(uidx)] if uidx is not None and int(uidx) < len(pu)
                     else np.zeros(pi.shape[1], np.float32))
            rows["prone_similarity"] = (pi[cand] @ p_emb).astype(np.float32)
            rows["u_age_diff"] = (rows["age_band_code"].to_numpy(np.float32)
                                  - rows["i_buyer_age_mean"].to_numpy(np.float32))
            rows["u_pa_diff"] = (rows["purchase_ability"].to_numpy(np.float32)
                                 - rows["i_buyer_pa_mean"].to_numpy(np.float32))
            uu = -1 if uidx is None else int(uidx)
            mi = pd.MultiIndex.from_arrays([np.full(len(cand), uu), rows["cat_code"].to_numpy()])
            rows["u_cat_affinity"] = qsv["u_cat"].reindex(mi).fillna(0).to_numpy(np.float32)
            mi = pd.MultiIndex.from_arrays([np.full(len(cand), uu), rows["fidx"].to_numpy()])
            rows["u_fam_affinity"] = qsv["u_fam"].reindex(mi).fillna(0).to_numpy(np.float32)
        # rule scores: 1/(rank+1) exactly as rules.query_match emits (norm=False strategy)
        for name, arr in (("QueryMatch_recent", recent), ("QueryMatch_alltime", alltime)):
            score = pd.Series(1.0 / (np.arange(len(arr)) + 1), index=arr)
            rows[name] = score.reindex(cand).fillna(0).to_numpy(np.float32)
        return rows.reset_index(drop=True)

    # ------------------------------------------------------------ core ranking
    def _rank(self, query: str, user_id: str, top_k: int) -> list[tuple[str, float]]:
        w = self.w
        qc, qs, qo = query_codes(w["vocab"], query)
        codes = w["item_query_codes"]
        uidx = self._u_map.get(str(user_id))

        lists: dict[str, list[int]] = {}
        for s_name, strat in w["strategies"].items():
            if s_name == "query":
                cand = self._query_rows(uidx, qc, qs, qo)
            else:
                cand = self._stores[s_name].candidates(uidx)
            if cand.empty:
                continue
            cand = cand.copy()
            cand["q_cat_eq"] = (codes["cat_code"].reindex(cand["iidx"]).fillna(-2).to_numpy() == qc).astype(np.float32)
            cand["q_style_eq"] = (codes["style_code"].reindex(cand["iidx"]).fillna(-2).to_numpy() == qs).astype(np.float32)
            cand["q_occ_eq"] = (codes["occ_code"].reindex(cand["iidx"]).fillna(-2).to_numpy() == qo).astype(np.float32)
            cand["q_match"] = cand["q_cat_eq"] + cand["q_style_eq"] + cand["q_occ_eq"]
            X = model_frame(cand, strat["feats"], strat["cat_feats"])
            for m_name, model in strat["models"].items():
                probs = model.predict_proba(X)[:, 1]
                lists[f"{s_name}_{m_name}"] = ranked_list(cand, probs)
        if not lists:
            return []
        blended = cust_blend(lists, w["blend_weights"], top_k)
        # blend score is the fusion weight sum; expose rank-order score for the response
        return [(str(self._items[i]), float(top_k - n) / top_k) for n, i in enumerate(blended)]

    # ------------------------------------------------------------ eval's view
    def recommend_dataframe(
        self, query: str, user_id: str = "anonymous", top_k: int = 5, candidate_k: int = 30
    ) -> pd.DataFrame:
        ranked = self._rank(query, user_id, top_k)
        return pd.DataFrame(
            {"product_id": [pid for pid, _ in ranked], "score": [s for _, s in ranked]}
        )

    # ------------------------------------------------------------ human's view
    def _item_result(self, rank: int, product_id: str, score: float) -> RecommendationResult:
        product: dict[str, Any] = {"product_id": product_id}
        if product_id in self._products_by_id.index:
            row = self._products_by_id.loc[product_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            product.update(row.to_dict())
        stock, margin, return_rate = 0, 0.0, 0.0
        if not self._inventory_by_id.empty and product_id in self._inventory_by_id.index:
            inv = self._inventory_by_id.loc[product_id]
            if isinstance(inv, pd.DataFrame):
                inv = inv.iloc[0]
            stock, margin, return_rate = int(inv.get("stock", 0)), float(inv.get("margin", 0.0)), float(inv.get("return_rate", 0.0))
        product.setdefault("price", 0.0)
        product["stock"], product["margin"], product["return_rate"] = stock, margin, return_rate

        evidence = [
            RecommendationEvidence(source="catalog", source_id=product_id, field=f, value=str(product.get(f)))
            for f in ("name", "category", "color", "style_tags", "occasion_tags")
            if str(product.get(f, "") or "")
        ]
        why = [
            "Retrieved by the two silver-solution recall strategies (rules + ALS/BPR/item-CF).",
            "Ranked by per-strategy LightGBM models fused with the silver cust_blend.",
        ]
        return RecommendationResult(
            product_id=product_id, rank=rank, score=round(score, 4), retrieval_score=round(score, 4),
            product=product, why=why, evidence=evidence,
            business_signals={"stock": stock, "margin": margin, "return_rate": return_rate,
                              "model_score": round(score, 4)},
        )

    def recommend(
        self, query: str, user_id: str = "anonymous", top_k: int = 5, candidate_k: int = 30
    ) -> dict[str, Any]:
        intent = parse_intent(query)
        ranked = self._rank(query, user_id, top_k)
        items = [self._item_result(r, pid, s) for r, (pid, s) in enumerate(ranked, start=1)]
        response = render_response(
            query=query, user_id=user_id, items=items,
            clarification_question=intent.clarification_question,
        )
        top = pd.DataFrame([item.product for item in items])
        response["parsed_intent"] = intent.__dict__
        response["business_metrics"] = summarize_business_metrics(top)
        response["business_interpretation"] = estimate_business_outcome(top)
        return response
