"""Collector + feature engineering — faithful ports (SPEC.md sections 1-2).

``collect``  ports reference collector.py: per-rule quantile score normalization,
             OutOfStock filtering, min_pos_rate pruning, pivot to per-rule score columns.
``cap_candidates`` ports the writeup's "retrieve 100 candidates per user" via the reference
             Recall 2 mechanism (min-max sum rank).
``build_feature_tables`` + ``join_features`` port the reference feature set
             (base_features.py + LGB Recall 1 cells 26-57) at day granularity, computed once
             per round and joined per user chunk (the winners' feature-store discipline).
             Every value uses ONLY transactions strictly before the round's label day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

MIN_POS_RATE = 0.006  # reference notebook cell 17


# ------------------------------------------------------------------- collector

def collect(rule_outputs: list[pd.DataFrame], labels: pd.DataFrame | None,
            rm_items: np.ndarray, min_pos_rate: float = MIN_POS_RATE,
            norm: bool = True) -> pd.DataFrame:
    """Union rule outputs into one candidate frame with per-rule score columns.

    ``labels`` (uidx, iidx) are the round's true purchases -- used for min_pos_rate pruning
    exactly as collector.py does; pass None for the serve round (no pruning, like week 0
    reusing week-1 thresholds).
    """

    kept = []
    rm = set(rm_items.tolist())
    for items in rule_outputs:
        if items.empty:
            continue
        items = items.copy()
        if norm:  # Recall 1 normalizes per rule; Recall 2 collects with norm=False
            items["score"] = QuantileTransformer(output_distribution="normal").fit_transform(
                items["score"].to_numpy().reshape(-1, 1)
            )
        items = items[~items["iidx"].isin(rm)]

        if labels is not None and len(labels):
            lab = labels.assign(label=1)
            tmp = items[items["uidx"].isin(lab["uidx"].unique())]
            tmp = tmp.merge(lab, on=["uidx", "iidx"], how="left")
            tmp["label"] = tmp["label"].fillna(0)
            pos_rate = tmp["label"].mean() if len(tmp) else 0.0
            if pos_rate < min_pos_rate:
                # trim the rule to its best top-rank (reference collector.py lines 96-122)
                tmp = tmp.sort_values("score", ascending=False)
                tmp["rank"] = tmp.groupby("uidx")["score"].rank(ascending=False)
                best_rate, best_rank = pos_rate, tmp["rank"].max()
                for rank in range(int(tmp["rank"].max()), 0, -1):
                    r = tmp.loc[tmp["rank"] <= rank, "label"].mean()
                    if r > best_rate:
                        best_rate, best_rank = r, rank
                    if r > min_pos_rate:
                        break
                if best_rate < min_pos_rate:
                    continue  # skip rule
                items = tmp[tmp["rank"] <= best_rank].drop(columns=["rank", "label"])
        kept.append(items)

    if not kept:
        return pd.DataFrame({"uidx": pd.Series(dtype="int32"), "iidx": pd.Series(dtype="int32")})
    union = pd.concat(kept, ignore_index=True)
    cand = union.pivot_table(index=["uidx", "iidx"], columns="method", values="score",
                             aggfunc="sum").reset_index()
    cand.columns.name = None
    # float32 everywhere but the ids -- the 60GB container cap punishes float64 pivots
    rule_cols = [c for c in cand.columns if c not in ("uidx", "iidx")]
    cand[rule_cols] = cand[rule_cols].astype(np.float32)
    return cand


def cap_candidates(cand: pd.DataFrame, n: int = 200, keep_score: bool = False) -> pd.DataFrame:
    """Reference Recall 2 cap: min-max normalize each rule column, sum, rank per user, keep
    top n. With keep_score=True the summed 'score' and per-user 'rank' stay as feature
    columns, exactly as Recall 2 keeps them."""

    rule_cols = [c for c in cand.columns if c not in ("uidx", "iidx")]
    if not rule_cols or cand.empty:
        return cand
    tmp = cand[rule_cols]
    lo, hi = tmp.min(), tmp.max()
    normed = (tmp - lo) / (hi - lo).replace(0, 1.0)
    score = normed.sum(axis=1)
    rank = score.groupby(cand["uidx"].to_numpy()).rank(ascending=False, method="first")
    out = cand.copy()
    if keep_score:
        out["score"], out["rank"] = score.to_numpy(np.float32), rank.to_numpy(np.float32)
    return out[rank.to_numpy() <= n].reset_index(drop=True)


# ------------------------------------------------------------------- features

def build_feature_tables(tx: pd.DataFrame, day: int, item_attrs: pd.DataFrame,
                         user_attrs: pd.DataFrame | None = None) -> dict:
    """All per-item / per-family / per-category / per-user aggregates as-of `day`, computed
    ONCE per round and joined per user chunk. Same math as the single-pass featurize --
    only hoisted out of the per-chunk path."""

    t: dict = {"day": day}
    txf = tx.merge(item_attrs[["fidx"]], left_on="iidx", right_index=True, how="left")

    # windowed sales 7/14/21/28d for item + family
    for days, tag in ((7, "1w"), (14, "2w"), (21, "3w"), (28, "4w")):
        win = txf[(txf["day"] >= day - days) & (txf["day"] < day)]
        for key, pfx in (("iidx", "i"), ("fidx", "p")):
            t[f"{pfx}_{tag}"] = win.groupby(key).size().astype("float32")
    # current week / prior week (+ unique buyer variants)
    curw = txf[(txf["day"] >= day - 7) & (txf["day"] < day)]
    lastw = txf[(txf["day"] >= day - 14) & (txf["day"] < day - 7)]
    for key, pfx in (("iidx", "i"), ("fidx", "p")):
        for frame, tag in ((curw, ""), (lastw, "lw_")):
            t[f"{tag}{pfx}_sale"] = frame.groupby(key).size().astype("float32")
            t[f"{tag}{pfx}_sale_uni"] = (
                frame.drop_duplicates(["uidx", key]).groupby(key).size().astype("float32")
            )
    # category-level
    txc = tx.merge(item_attrs[["cat_code"]], left_on="iidx", right_index=True, how="left")
    t["cat_sale"] = txc[(txc["day"] >= day - 7) & (txc["day"] < day)].groupby("cat_code").size().astype("float32")
    t["lw_cat_sale"] = txc[(txc["day"] >= day - 14) & (txc["day"] < day - 7)].groupby("cat_code").size().astype("float32")
    # repurchase ratios
    for key, pfx in (("iidx", "i"), ("fidx", "p")):
        per_user = txf.groupby([key, "uidx"]).size()
        buyers = per_user.groupby(level=0).size()
        multi = per_user[per_user > 1].groupby(level=0).size()
        t[f"{pfx}_repurchase_ratio"] = (multi.reindex(buyers.index).fillna(0) / (buyers + 1e-6)).astype("float32")
    # age / cumulative / popularity / time stats
    t["i_first"] = tx.groupby("iidx")["day"].min()
    t["f_first"] = txf.groupby("fidx")["day"].min()
    t["i_full"] = tx.groupby("iidx").size().astype("float32")
    t["p_full"] = txf.groupby("fidx").size().astype("float32")
    decay = 1.0 / (((day - 1) - txf["day"]) + 1)
    t["i_pop"] = decay.groupby(txf["iidx"]).sum().astype("float32")
    t["p_pop"] = decay.groupby(txf["fidx"]).sum().astype("float32")
    t["i_day_mean"] = tx.groupby("iidx")["day"].mean()
    ug = tx.groupby("uidx")["day"]
    t["u_last"], t["u_first"], t["u_mean"] = ug.max(), ug.min(), ug.mean()
    # price / channel
    t["i_last_price"] = tx.groupby("iidx")["price"].last()
    t["i_channel"] = tx.groupby("iidx")["channel"].agg(lambda s: s.mode().iloc[0] if len(s) else 0)
    t["u_mean_price"] = tx.groupby("uidx")["price"].mean()
    t["ui_sale"] = tx.groupby(["uidx", "iidx"]).size().astype("float32")
    # --- top1-solution feature blocks (behavioural v2.0) ---
    if user_attrs is not None:
        # buyer profile per item: who buys it (their age-diff / ratio principle)
        age = user_attrs["age_band_code"].reindex(tx["uidx"]).fillna(0).to_numpy("float32")
        pa = user_attrs["purchase_ability"].reindex(tx["uidx"]).fillna(0).to_numpy("float32")
        g = pd.Series(age).groupby(tx["iidx"].to_numpy())
        t["i_buyer_age_mean"] = g.mean().astype("float32")
        t["i_buyer_age_std"] = g.std().fillna(0).astype("float32")
        t["i_buyer_pa_mean"] = pd.Series(pa).groupby(tx["iidx"].to_numpy()).mean().astype("float32")
    # same-window-last-year sales (their "same week last year" count)
    ly = txf[(txf["day"] >= day - 372) & (txf["day"] < day - 358)]
    t["i_ly_sale"] = ly.groupby("iidx").size().astype("float32")
    t["p_ly_sale"] = ly.groupby("fidx").size().astype("float32")
    # time-weighted user-side affinities (their time-weighted counts)
    decay2 = 1.0 / (((day - 1) - tx["day"]) + 1)
    catc = item_attrs["cat_code"].reindex(tx["iidx"]).fillna(-1).to_numpy()
    famc = item_attrs["fidx"].reindex(tx["iidx"]).fillna(-1).to_numpy()
    t["u_cat"] = decay2.groupby([tx["uidx"].to_numpy(), catc]).sum().astype("float32")
    t["u_fam"] = decay2.groupby([tx["uidx"].to_numpy(), famc]).sum().astype("float32")
    # rank tables precomputed once (rank of every unit's sale within the window)
    for tag in ("1w", "2w", "3w", "4w"):
        for pfx in ("i", "p"):
            t[f"{pfx}_{tag}_rank"] = t[f"{pfx}_{tag}"].rank(ascending=False)
            t[f"{pfx}_{tag}_sum"] = max(float(t[f"{pfx}_{tag}"].sum()), 1e-6)
    return t


def join_features(t: dict, cand: pd.DataFrame, item_attrs: pd.DataFrame,
                  user_attrs: pd.DataFrame) -> pd.DataFrame:
    """Join the precomputed tables onto one chunk's candidates -- values identical to the
    previous single-pass featurize."""

    day = t["day"]
    c = cand.copy()
    c["fidx"] = item_attrs["fidx"].reindex(c["iidx"]).to_numpy()
    c["cat_code"] = item_attrs["cat_code"].reindex(c["iidx"]).to_numpy()

    def by(key_col, table, fill=0.0):
        return table.reindex(c[key_col]).fillna(fill).to_numpy(np.float32)

    for tag in ("1w", "2w", "3w", "4w"):
        for key_col, pfx in (("iidx", "i"), ("fidx", "p")):
            c[f"{pfx}_{tag}_sale"] = by(key_col, t[f"{pfx}_{tag}"])
            c[f"{pfx}_{tag}_sale_rank"] = by(key_col, t[f"{pfx}_{tag}_rank"], fill=len(t[f"{pfx}_{tag}"]) + 1)
            c[f"{pfx}_{tag}_sale_norm"] = c[f"{pfx}_{tag}_sale"] / t[f"{pfx}_{tag}_sum"]
    for key_col, pfx in (("iidx", "i"), ("fidx", "p")):
        for tag in ("", "lw_"):
            c[f"{tag}{pfx}_sale"] = by(key_col, t[f"{tag}{pfx}_sale"])
            c[f"{tag}{pfx}_sale_uni"] = by(key_col, t[f"{tag}{pfx}_sale_uni"])
    c["i_sale_ratio"] = c["i_sale"] / (c["p_sale"] + 1e-6)
    c["i_sale_uni_ratio"] = c["i_sale_uni"] / (c["p_sale_uni"] + 1e-6)
    c["lw_i_sale_ratio"] = c["lw_i_sale"] / (c["lw_p_sale"] + 1e-6)
    c["lw_i_sale_uni_ratio"] = c["lw_i_sale_uni"] / (c["lw_p_sale_uni"] + 1e-6)
    c["i_uni_ratio"] = c["i_sale"] / (c["i_sale_uni"] + 1e-6)
    c["p_uni_ratio"] = c["p_sale"] / (c["p_sale_uni"] + 1e-6)
    c["lw_i_uni_ratio"] = c["lw_i_sale"] / (c["lw_i_sale_uni"] + 1e-6)
    c["lw_p_uni_ratio"] = c["lw_p_sale"] / (c["lw_p_sale_uni"] + 1e-6)
    c["i_sale_trend"] = (c["i_sale"] - c["lw_i_sale"]) / (c["lw_i_sale"] + 1e-6)
    c["p_sale_trend"] = (c["p_sale"] - c["lw_p_sale"]) / (c["lw_p_sale"] + 1e-6)
    c["cat_sale"] = t["cat_sale"].reindex(c["cat_code"]).fillna(0).to_numpy(np.float32)
    c["lw_cat_sale"] = t["lw_cat_sale"].reindex(c["cat_code"]).fillna(0).to_numpy(np.float32)
    c["cat_sale_trend"] = (c["cat_sale"] - c["lw_cat_sale"]) / (c["lw_cat_sale"] + 1e-6)
    c["i_repurchase_ratio"] = by("iidx", t["i_repurchase_ratio"])
    c["p_repurchase_ratio"] = by("fidx", t["p_repurchase_ratio"])
    c["first_dat"] = (day - t["i_first"].reindex(c["iidx"]).fillna(day)).to_numpy(np.float32).clip(1)
    c["i_full_sale"] = by("iidx", t["i_full"])
    c["p_full_sale"] = by("fidx", t["p_full"])
    c["i_daily_sale"] = c["i_full_sale"] / c["first_dat"]
    fam_age = (day - t["f_first"].reindex(c["fidx"]).fillna(day)).to_numpy(np.float32).clip(1)
    c["p_daily_sale"] = c["p_full_sale"] / fam_age
    c["i_daily_sale_ratio"] = c["i_daily_sale"] / (c["p_daily_sale"] + 1e-6)
    c["i_w_full_sale_ratio"] = c["i_sale"] / (c["i_full_sale"] + 1e-6)
    c["i_2w_full_sale_ratio"] = c["i_2w_sale"] / (c["i_full_sale"] + 1e-6)
    c["p_w_full_sale_ratio"] = c["p_sale"] / (c["p_full_sale"] + 1e-6)
    c["p_2w_full_sale_ratio"] = c["p_2w_sale"] / (c["p_full_sale"] + 1e-6)
    c["i_week_above_daily_sale"] = c["i_sale"] / 7 - c["i_daily_sale"]
    c["i_2w_week_above_daily_sale"] = c["i_2w_sale"] / 14 - c["i_daily_sale"]
    c["p_2w_week_above_daily_sale"] = c["p_2w_sale"] / 14 - c["p_daily_sale"]
    c["i_pop"] = by("iidx", t["i_pop"])
    c["p_pop"] = by("fidx", t["p_pop"])
    c["article_time_mean"] = (day - t["i_day_mean"].reindex(c["iidx"]).fillna(day)).to_numpy(np.float32)
    c["customer_id_last_time"] = (day - t["u_last"].reindex(c["uidx"]).fillna(day)).to_numpy(np.float32)
    c["customer_id_first_time"] = (day - t["u_first"].reindex(c["uidx"]).fillna(day)).to_numpy(np.float32)
    c["customer_id_time_mean"] = (day - t["u_mean"].reindex(c["uidx"]).fillna(day)).to_numpy(np.float32)
    c["customer_id_gap"] = c["customer_id_first_time"] - c["customer_id_last_time"]
    c["item_price"] = t["i_last_price"].reindex(c["iidx"]).fillna(item_attrs["price"].reindex(c["iidx"])).fillna(0).to_numpy(np.float32)
    c["item_channel"] = t["i_channel"].reindex(c["iidx"]).fillna(0).to_numpy(np.float32)
    c["user_mean_price"] = t["u_mean_price"].reindex(c["uidx"]).fillna(0).to_numpy(np.float32)
    c["price_ratio"] = c["item_price"] / (c["user_mean_price"] + 1e-6)
    c["ui_sale"] = t["ui_sale"].reindex(pd.MultiIndex.from_arrays([c["uidx"], c["iidx"]])).fillna(0).to_numpy(np.float32)
    c["ui_sale_ratio"] = c["ui_sale"] / (c["i_full_sale"] + 1e-6)
    for col in ("color_code", "season_code", "article_gender", "season_type"):
        c[col] = item_attrs[col].reindex(c["iidx"]).fillna(0).to_numpy(np.int32)
    for col in ("age_band_code", "user_gender", "purchase_ability"):
        c[col] = user_attrs[col].reindex(c["uidx"]).fillna(0).to_numpy(np.int32)
    # --- top1-solution features (v2.0) ---
    if "i_buyer_age_mean" in t:
        c["i_buyer_age_mean"] = by("iidx", t["i_buyer_age_mean"])
        c["i_buyer_age_std"] = by("iidx", t["i_buyer_age_std"])
        c["i_buyer_pa_mean"] = by("iidx", t["i_buyer_pa_mean"])
        u_age = user_attrs["age_band_code"].reindex(c["uidx"]).fillna(0).to_numpy("float32")
        u_pa = user_attrs["purchase_ability"].reindex(c["uidx"]).fillna(0).to_numpy("float32")
        c["u_age_diff"] = u_age - c["i_buyer_age_mean"]
        c["u_pa_diff"] = u_pa - c["i_buyer_pa_mean"]
    c["i_ly_sale"] = by("iidx", t["i_ly_sale"])
    c["p_ly_sale"] = by("fidx", t["p_ly_sale"])
    c["i_ly_ratio"] = c["i_ly_sale"] / (c["i_2w_sale"] + 1e-6)
    c["u_cat_affinity"] = t["u_cat"].reindex(
        pd.MultiIndex.from_arrays([c["uidx"], c["cat_code"]])).fillna(0).to_numpy("float32")
    c["u_fam_affinity"] = t["u_fam"].reindex(
        pd.MultiIndex.from_arrays([c["uidx"], c["fidx"]])).fillna(0).to_numpy("float32")
    rule_cols = [x for x in cand.columns if x not in ("uidx", "iidx")]
    c[rule_cols] = c[rule_cols].fillna(0)
    return c


def add_similarities(c: pd.DataFrame, w2v_user, w2v_item, prone_user, prone_item) -> pd.DataFrame:
    """Similarity features (writeup: w2v cosine i2i, ProNE u2i) via calc_embd_similarity."""

    u, i = c["uidx"].to_numpy(), c["iidx"].to_numpy()
    c["wv_similarity"] = np.einsum("ij,ij->i", w2v_user[u], w2v_item[i]).astype(np.float32)
    c["prone_similarity"] = np.einsum("ij,ij->i", prone_user[u], prone_item[i]).astype(np.float32)
    return c


def add_query_features(c: pd.DataFrame, item_attrs: pd.DataFrame,
                       q_cat, q_style, q_occ) -> pd.DataFrame:
    """SPEC adaptation 1 -- the query as agreement features (user decision: eq-only)."""

    cat = item_attrs["cat_code"].reindex(c["iidx"]).fillna(-2).to_numpy()
    sty = item_attrs["style_code"].reindex(c["iidx"]).fillna(-2).to_numpy()
    occ = item_attrs["occ_code"].reindex(c["iidx"]).fillna(-2).to_numpy()
    c["q_cat_eq"] = (cat == q_cat).astype(np.float32)
    c["q_style_eq"] = (sty == q_style).astype(np.float32)
    c["q_occ_eq"] = (occ == q_occ).astype(np.float32)
    c["q_match"] = c["q_cat_eq"] + c["q_style_eq"] + c["q_occ_eq"]
    return c
