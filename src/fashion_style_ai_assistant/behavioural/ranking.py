"""Behavioural v5 ranking: query features, per-strategy LGB scoring, gen_submit blend.

``model_frame`` is shared with training so feature dtypes can never drift.
``cust_blend`` is the exact rank-fusion from the silver repo's gen_submit notebook:
each model's ranked list contributes W/(position+1) per item; sum, sort, cut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def model_frame(df: pd.DataFrame, feats: list[str], cat_feats: list[str]) -> pd.DataFrame:
    """The exact frame the boosters were fitted on: int categoricals, float32 continuous."""

    X = df.reindex(columns=feats)  # a rule column absent in this frame -> NaN -> filled below
    for c in cat_feats:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(-1).astype(np.int64)
    cont = [c for c in feats if c not in cat_feats]
    X[cont] = X[cont].astype(np.float32).fillna(0)
    return X


def match_code(vocab: np.ndarray, query_lower: str) -> int:
    """Code of the longest vocab value appearing verbatim in the query, else -1."""

    best, best_len = -1, 0
    for code, value in enumerate(vocab):
        v = str(value)
        if len(v) > best_len and v and v in query_lower:
            best, best_len = code, len(v)
    return best


def query_codes(vocab: dict[str, np.ndarray], query: str) -> tuple[int, int, int]:
    q = query.lower()
    return match_code(vocab["cat"], q), match_code(vocab["style"], q), match_code(vocab["occ"], q)


def ranked_list(cand: pd.DataFrame, probs: np.ndarray) -> list[int]:
    """Best-first deduplicated iidx list from one model's probabilities."""

    order = np.argsort(-probs)
    seen, out = set(), []
    for pos in order:
        iidx = int(cand["iidx"].iloc[pos])
        if iidx not in seen:
            seen.add(iidx)
            out.append(iidx)
    return out


def cust_blend(lists: dict[str, list[int]], weights: dict[str, float], top_k: int) -> list[int]:
    """gen_submit cell 18: score item = sum over lists of W/(position+1); sort desc."""

    res: dict[int, float] = {}
    for key, rec in lists.items():
        w = weights.get(key, 1.0)
        for n, v in enumerate(rec):
            res[v] = res.get(v, 0.0) + w / (n + 1)
    return [v for v, _ in sorted(res.items(), key=lambda kv: -kv[1])[:top_k]]
