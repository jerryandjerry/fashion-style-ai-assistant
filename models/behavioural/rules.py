"""Faithful ports of the reference retrieval rules (SPEC.md section 1).

Every function reproduces one rule class from
``reference/H-M-Fashion-RecSys/src/retrieval/rules.py`` with the same math and reference
parameters, renamed to our schema and made day-granular (SPEC adaptation 2):

    customer_id -> uidx (int32)   article_id -> iidx (int32)   t_dat -> day (int32)

``tx`` is always the encoded transactions frame STRICTLY BEFORE the round's label day.
Every rule returns a frame (uidx, iidx, method, score) — scores bigger-is-better, exactly as
the reference states.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

# OrderHistoryDecay / TimeHistoryDecay constants — reference rules.py lines 222, 1204
_DECAY_A, _DECAY_B, _DECAY_C, _DECAY_D = 2.5e4, 1.5e5, 2e-1, 1e3


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"uidx": pd.Series(dtype=np.int32), "iidx": pd.Series(dtype=np.int32),
                         "method": pd.Series(dtype=object), "score": pd.Series(dtype=np.float32)})


# ---------------------------------------------------------------- personal rules

def order_history(tx: pd.DataFrame, days: int, n: int | None = None, name: str = "1") -> pd.DataFrame:
    """OrderHistory: items the user bought within `days` of their own last purchase."""

    df = tx[["uidx", "iidx", "day"]]
    max_day = df.groupby("uidx")["day"].transform("max")
    diff = max_day - df["day"]
    df = df.assign(diff=diff).loc[diff < days]
    # most recent occurrence per (user, item); score = -days-ago (reference line 152)
    df = df.sort_values("diff").drop_duplicates(["uidx", "iidx"])
    if n is not None:
        df = df[df.groupby("uidx")["diff"].rank(method="first") <= n]
    return pd.DataFrame(
        {"uidx": df["uidx"], "iidx": df["iidx"], "method": f"OrderHistory_{name}",
         "score": -df["diff"].astype(np.float32)}
    )


def order_history_decay(tx: pd.DataFrame, days: int, n: int | None = None, name: str = "1") -> pd.DataFrame:
    """OrderHistoryDecay: recency-decayed value x period-sale quotient, threshold >150."""

    if tx.empty:
        return _empty()
    df = tx[["uidx", "iidx", "day"]].copy()
    last = int(df["day"].max())
    df["gap"] = (last - df["day"]).clip(lower=1)
    # period-sale quotient: sales of the item in the current `days`-period vs its own period
    df["period"] = (last - df["day"]) // days
    period_sale = df.groupby(["period", "iidx"]).size().rename("period_sale")
    df = df.join(period_sale, on=["period", "iidx"])
    targ = period_sale.xs(0, level="period") if 0 in period_sale.index.get_level_values(0) else pd.Series(dtype=float)
    df = df.join(targ.rename("period_sale_targ"), on="iidx")
    df["period_sale_targ"] = df["period_sale_targ"].fillna(0)
    df["quotient"] = df["period_sale_targ"] / df["period_sale"]

    x = df["gap"].to_numpy(dtype=np.float64)
    value = _DECAY_A / np.sqrt(x) + _DECAY_B * np.exp(-_DECAY_C * x) - _DECAY_D
    value[value < 0] = 0
    df["value"] = value * df["quotient"].to_numpy()

    df = df.groupby(["uidx", "iidx"], as_index=False)["value"].sum()
    df = df[df["value"] > 150]  # reference line 236
    if n is not None:
        df = df[df.groupby("uidx")["value"].rank(ascending=False, method="first") <= n]
    return pd.DataFrame(
        {"uidx": df["uidx"], "iidx": df["iidx"], "method": f"OrderHistoryDecay_{name}",
         "score": df["value"].astype(np.float32)}
    )


def item_pair(base: pd.DataFrame, name: str = "1") -> pd.DataFrame:
    """ItemPair, reference-exact: the pair table is built FROM the base rule's own output
    (``ItemPair(OrderHistory(...).retrieve())``) -- recent items only, never the full log."""

    if base.empty:
        return _empty()
    pairs_src = base[["uidx", "iidx"]].drop_duplicates()
    pair = pairs_src.merge(pairs_src.rename(columns={"iidx": "pair"}), on="uidx")
    pair = pair[pair["iidx"] != pair["pair"]]
    pair = pair.groupby(["iidx", "pair"]).size().rename("count").reset_index()
    # most frequent partner per item (reference sorts desc then groupby.first)
    pair = pair.sort_values("count", ascending=False).drop_duplicates("iidx")

    df = pairs_src.merge(pair, on="iidx", how="left")
    df = df.dropna(subset=["pair"]).drop_duplicates(["uidx", "pair"])
    return pd.DataFrame(
        {"uidx": df["uidx"], "iidx": df["pair"].astype(np.int32),
         "method": f"ItemPairRetrieve_{name}", "score": df["count"].astype(np.float32)}
    )


def item_cf_sim(hist_tx: pd.DataFrame, top_k: int = 10) -> dict[int, tuple[list[int], list[float]]]:
    """ItemCF similarity table — direction factor 1.0/0.9, distance factor 0.7^(d-1),
    popularity factor 1/log(1+len). Build once per round, reuse across user chunks."""

    hist = hist_tx.sort_values("day").groupby("uidx")["iidx"].apply(list)

    sim_item: dict[int, dict[int, float]] = {}
    for items in hist.values:
        ln = math.log(1 + len(items))
        for i, item in enumerate(items[:-1]):
            d = sim_item.setdefault(item, {})
            for j, relate in enumerate(items):
                if i == j:
                    continue
                loc_alpha = 1.0 if j > i else 0.9
                w = loc_alpha * 0.7 ** (abs(j - i) - 1)
                d[relate] = d.get(relate, 0.0) + w / ln

    top: dict[int, tuple[list[int], list[float]]] = {}
    for item, related in sim_item.items():
        if len(related) >= 5:  # reference line 822
            best = sorted(related.items(), key=lambda x: x[1], reverse=True)[:top_k]
            top[item] = ([x[0] for x in best], [x[1] for x in best])
    return top


def item_cf_predict(sim: dict, target_tx: pd.DataFrame, method: str) -> pd.DataFrame:
    """ItemCF prediction from a prebuilt similarity table (reference _predict)."""

    target = target_tx.groupby("uidx")["iidx"].apply(lambda s: list(dict.fromkeys(s)))
    users, items_out, scores = [], [], []
    for uid, items in target.items():
        for item in items:
            hit = sim.get(int(item))
            if hit is not None:
                users.extend([uid] * len(hit[0]))
                items_out.extend(hit[0])
                scores.extend(hit[1])
    if not users:
        return _empty()
    return pd.DataFrame(
        {"uidx": np.array(users, dtype=np.int32), "iidx": np.array(items_out, dtype=np.int32),
         "method": method, "score": np.array(scores, dtype=np.float32)}
    )


def item_cf(hist_tx, target_tx, top_k: int = 10, name: str = "1") -> pd.DataFrame:
    """ItemCF end-to-end (build sim + predict) — used by training rounds."""

    return item_cf_predict(item_cf_sim(hist_tx, top_k), target_tx, f"ItemCF_{name}")


def ug_item_cf_sims(hist_tx, groups: pd.Series, top_k: int = 10) -> dict:
    """Per-group ItemCF similarity tables (UserGroupItemCF, build-once half)."""

    h = hist_tx.assign(_g=hist_tx["uidx"].map(groups))
    return {g: item_cf_sim(h[h["_g"] == g], top_k) for g in h["_g"].dropna().unique()}


def ug_item_cf_predict(sims: dict, target_tx, groups: pd.Series, name: str = "1") -> pd.DataFrame:
    """UserGroupItemCF prediction from prebuilt per-group tables."""

    t = target_tx.assign(_g=target_tx["uidx"].map(groups))
    parts = [item_cf_predict(sims[g], t[t["_g"] == g], f"UGItemCF_{name}")
             for g in t["_g"].dropna().unique() if g in sims]
    if not parts:
        return _empty()
    return pd.concat(parts, ignore_index=True)


def ug_item_cf(hist_tx, target_tx, groups: pd.Series, top_k: int = 10, name: str = "1") -> pd.DataFrame:
    """UserGroupItemCF end-to-end — used by training rounds."""

    return ug_item_cf_predict(ug_item_cf_sims(hist_tx, groups, top_k), target_tx, groups, name)


def mf_train(model_cls, tx: pd.DataFrame, n_users: int, n_items: int, **kwargs):
    """Fit an implicit ALS/BPR model on dedup (u,i) — build once per round."""

    ui = tx[["uidx", "iidx"]].drop_duplicates()
    mat = coo_matrix(
        (np.ones(len(ui), dtype=np.float32), (ui["uidx"].to_numpy(), ui["iidx"].to_numpy())),
        shape=(n_users, n_items),
    ).tocsr()
    model = model_cls(random_state=42, **kwargs)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1, "blas"):
            model.fit(mat, show_progress=False)
    except ImportError:
        model.fit(mat, show_progress=False)
    return model, mat


def mf_predict(model, mat, customer_list: np.ndarray, pool: np.ndarray, n: int,
               method: str) -> pd.DataFrame:
    """Recommend top-n pool items per user from a prefit model (reference _predict)."""

    if len(pool) == 0 or len(customer_list) == 0:
        return _empty()
    uids = np.asarray(customer_list, dtype=np.int32)
    preds = np.zeros((len(uids), n), dtype=np.int64)
    scores = np.zeros((len(uids), n), dtype=np.float32)
    for start in range(0, len(uids), 10000):
        batch = uids[start:start + 10000]
        ids, ss = model.recommend(
            batch, mat[batch], N=n, items=pool, filter_already_liked_items=False
        )
        preds[start:start + 10000] = ids
        scores[start:start + 10000] = ss
    return pd.DataFrame(
        {"uidx": np.repeat(uids, n), "iidx": preds.reshape(-1).astype(np.int32),
         "method": method, "score": scores.reshape(-1)}
    )


def build_triple_tables(tx: pd.DataFrame, item_attrs: pd.DataFrame, k: int = 200) -> dict:
    """QueryMatch stores: (cat,style,occ) triple -> top-k items by recent-4w AND all-time
    popularity (both anchored as-of the round day). The dual ranking is data-driven: recent
    covers fresh targets, all-time covers old ones (see diag_transfer results in SPEC)."""

    last = int(tx["day"].max())
    tables = {}
    for name, pop in (("recent", tx[tx["day"] >= last - 27].groupby("iidx").size()),
                      ("alltime", tx.groupby("iidx").size())):
        a = item_attrs[["cat_code", "style_code", "occ_code"]].copy()
        a["pop"] = pop.reindex(a.index).fillna(0).to_numpy()
        a["iidx"] = a.index
        top = (a.sort_values("pop", ascending=False)
                .groupby(["cat_code", "style_code", "occ_code"]).head(k))
        tables[name] = {key: g["iidx"].to_numpy(np.int32)
                        for key, g in top.groupby(["cat_code", "style_code", "occ_code"])}
    return tables


def query_match(user_triples: dict, table: dict[tuple, np.ndarray], name: str) -> pd.DataFrame:
    """QueryMatch rule: for each user's query triple, its top-k table items.
    Score = 1/(rank+1) (best-first). user_triples: uidx -> (cat, style, occ) codes."""

    users, items_out, scores = [], [], []
    for uid, triple in user_triples.items():
        cands = table.get(triple)
        if cands is None or len(cands) == 0:
            continue
        users.extend([uid] * len(cands))
        items_out.extend(cands.tolist())
        scores.extend((1.0 / (np.arange(len(cands)) + 1)).tolist())
    if not users:
        return _empty()
    return pd.DataFrame(
        {"uidx": np.array(users, dtype=np.int32), "iidx": np.array(items_out, dtype=np.int32),
         "method": f"QueryMatch_{name}", "score": np.array(scores, dtype=np.float32)}
    )


def recent_pool(tx: pd.DataFrame, days: int = 5, size: int = 1000) -> np.ndarray:
    """Reference target_items: top-`size` items by sales in the last `days` days."""

    recent = tx[tx["day"] > tx["day"].max() - days]
    return recent["iidx"].value_counts().index[:size].to_numpy()


def als_train(tx, n_users, n_items, factors=200, iterations=25):
    from implicit.als import AlternatingLeastSquares
    if tx.empty:
        return None, None
    return mf_train(AlternatingLeastSquares, tx, n_users, n_items,
                    factors=factors, iterations=iterations)


def bpr_train(tx, n_users, n_items, factors=300, iterations=350):
    from implicit.bpr import BayesianPersonalizedRanking
    if tx.empty:
        return None, None
    return mf_train(BayesianPersonalizedRanking, tx, n_users, n_items,
                    factors=factors, iterations=iterations, learning_rate=0.05)


def w2v_i2i(neighbors: dict[int, tuple[np.ndarray, np.ndarray]], target_tx: pd.DataFrame,
            name: str = "1") -> pd.DataFrame:
    """w2v item2item (writeup retrieval strategy): each of the user's recent items pulls its
    precomputed word2vec nearest neighbors. `neighbors`: iidx -> (neighbor iidx, cosine)."""

    target = target_tx.groupby("uidx")["iidx"].apply(lambda s: list(dict.fromkeys(s)))
    users, items_out, scores = [], [], []
    for uid, items in target.items():
        for item in items:
            hit = neighbors.get(int(item))
            if hit is not None:
                users.extend([uid] * len(hit[0]))
                items_out.extend(hit[0])
                scores.extend(hit[1])
    return pd.DataFrame(
        {"uidx": np.array(users, dtype=np.int32), "iidx": np.array(items_out, dtype=np.int32),
         "method": f"W2V_{name}", "score": np.array(scores, dtype=np.float32)}
    )


def prone_u2i(user_embd: np.ndarray, item_embd: np.ndarray, customer_list: np.ndarray,
              pool: np.ndarray, n: int = 100, name: str = "1") -> pd.DataFrame:
    """ProNE user2item (writeup retrieval strategy): top-n pool items by embedding dot."""

    if len(pool) == 0 or len(customer_list) == 0:
        return _empty()
    uids = np.asarray(customer_list, dtype=np.int32)
    sims = user_embd[uids] @ item_embd[pool].T          # (n_users, pool)
    n = min(n, len(pool))
    top = np.argpartition(-sims, n - 1, axis=1)[:, :n]
    scores = np.take_along_axis(sims, top, axis=1)
    return pd.DataFrame(
        {"uidx": np.repeat(uids, n), "iidx": pool[top].reshape(-1).astype(np.int32),
         "method": f"ProNE_{name}", "score": scores.reshape(-1).astype(np.float32)}
    )


# ---------------------------------------------------------------- group / global rules

def _broadcast(per_key: pd.DataFrame, key: str, users: pd.DataFrame) -> pd.DataFrame:
    """Reference merge(): attach group-level (or global) item lists to each target user."""

    out = users.merge(per_key, on=key, how="inner") if key else users.merge(per_key, how="cross")
    return out[["uidx", "iidx", "method", "score"]]


def ug_time_history(tx_window, users, group_col, n=50, scale=False, name="1") -> pd.DataFrame:
    """UserGroupTimeHistory: top-n items per user group in the window (dedup (u,i) first)."""

    df = tx_window.drop_duplicates(["uidx", "iidx"])
    df = df.assign(_g=df["uidx"].map(users.set_index("uidx")[group_col]))
    cnt = df.groupby(["_g", "iidx"]).size().rename("count").reset_index()
    cnt = cnt[cnt.groupby("_g")["count"].rank(ascending=False, method="first") <= n]
    cnt["score"] = (cnt["count"] / cnt["count"].max()) if scale else cnt["count"]
    cnt["method"] = f"UGTimeHistory_{name}"
    cnt = cnt.rename(columns={"_g": group_col})[[group_col, "iidx", "method", "score"]]
    return _broadcast(cnt, group_col, users[["uidx", group_col]])


def ug_sale_trend(tx, users, group_col, days=7, n=50, t=0.8, name="1") -> pd.DataFrame:
    """UserGroupSaleTrend: per-group trend (recent `days` vs prior `days`), /(prev+1)."""

    last = tx["day"].max()
    gap = last - tx["day"]
    win = tx.loc[gap <= 2 * days - 1].assign(_g=lambda d: d["uidx"].map(users.set_index("uidx")[group_col]))
    b = win[last - win["day"] <= days - 1].groupby(["_g", "iidx"]).size().rename("count_x")
    a = win[last - win["day"] > days - 1].groupby(["_g", "iidx"]).size().rename("count_y")
    log = pd.concat([b, a], axis=1).fillna({"count_x": 0}).dropna(subset=["count_y"]).reset_index()
    log["trend"] = (log["count_x"] - log["count_y"]) / (log["count_y"] + 1)
    log = log[log["trend"] > t].sort_values(["count_x", "trend"], ascending=False)
    log = log[log.groupby("_g").cumcount() < n]
    log["method"] = f"UGSaleTrend_{name}"
    log["score"] = log["trend"]
    log = log.rename(columns={"_g": group_col})[[group_col, "iidx", "method", "score"]]
    return _broadcast(log, group_col, users[["uidx", group_col]])


def time_history(tx_window, users, n=50, name="1") -> pd.DataFrame:
    """TimeHistory: global top-n items by count in the window (dedup (u,i) first)."""

    df = tx_window.drop_duplicates(["uidx", "iidx"])
    cnt = df.groupby("iidx").size().rename("score").reset_index().nlargest(n, "score")
    cnt["method"] = f"TimeHistory_{name}"
    return _broadcast(cnt[["iidx", "method", "score"]], "", users[["uidx"]])


def time_history_decay(tx, users, days=7, n=50, name="1") -> pd.DataFrame:
    """TimeHistoryDecay: global decayed popularity, same decay + quotient as OrderHistoryDecay."""

    if tx.empty:
        return _empty()
    df = tx[["iidx", "day"]].copy()
    last = int(df["day"].max())
    df["gap"] = (last - df["day"]).clip(lower=1)
    df["period"] = (last - df["day"]) // days
    period_sale = df.groupby(["period", "iidx"]).size().rename("period_sale")
    df = df.join(period_sale, on=["period", "iidx"])
    targ = period_sale.xs(0, level="period") if 0 in period_sale.index.get_level_values(0) else pd.Series(dtype=float)
    df = df.join(targ.rename("t0"), on="iidx")
    df["quotient"] = df["t0"].fillna(0) / df["period_sale"]

    x = df["gap"].to_numpy(dtype=np.float64)
    value = _DECAY_A / np.sqrt(x) + _DECAY_B * np.exp(-_DECAY_C * x) - _DECAY_D
    value[value < 0] = 0
    df["value"] = value * df["quotient"].to_numpy()
    cnt = df.groupby("iidx")["value"].sum().rename("score").reset_index().nlargest(n, "score")
    cnt["method"] = f"TimeHistoryDecay_{name}"
    return _broadcast(cnt[["iidx", "method", "score"]], "", users[["uidx"]])


def sale_trend(tx, users, days=7, n=50, t=0.8, name="1") -> pd.DataFrame:
    """SaleTrend: global trending items — (recent - prior)/prior > t."""

    last = tx["day"].max()
    gap = last - tx["day"]
    win = tx.loc[gap <= 2 * days - 1]
    b = win[last - win["day"] <= days - 1].groupby("iidx").size().rename("count_x")
    a = win[last - win["day"] > days - 1].groupby("iidx").size().rename("count_y")
    log = pd.concat([b, a], axis=1).fillna({"count_x": 0}).dropna(subset=["count_y"]).reset_index()
    log["trend"] = (log["count_x"] - log["count_y"]) / log["count_y"]
    log = log[log["trend"] > t].sort_values(["count_x", "trend"], ascending=False).head(n)
    log["method"] = f"SaleTrend_{name}"
    log["score"] = log["trend"]
    return _broadcast(log[["iidx", "method", "score"]], "", users[["uidx"]])


def out_of_stock(tx: pd.DataFrame) -> np.ndarray:
    """OutOfStock filter: items whose last-30-day sales dropped >80% vs the prior 30 days,
    or hit zero (reference used the last two calendar months; day-granular analog)."""

    last = tx["day"].max()
    recent = tx[tx["day"] > last - 60]
    cur = recent[recent["day"] > last - 30].groupby("iidx").size()
    prev = recent[recent["day"] <= last - 30].groupby("iidx").size()
    both = pd.concat([cur.rename("cur"), prev.rename("prev")], axis=1).fillna(0)
    both = both[both["prev"] > 0]
    mask = ((both["cur"] - both["prev"]) / both["prev"] < -0.8) | (both["cur"] == 0)
    return both.index[mask].to_numpy()
