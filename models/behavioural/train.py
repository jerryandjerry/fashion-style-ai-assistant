"""Train behavioural v5 -- EXACT port of the silver-medal solution,
reference/H-M-Fashion-RecSys (target chosen 2026-08-10; see SPEC.md).

Architecture (their README + notebooks, verbatim configs):
  - TWO recall strategies, each its own candidate pipeline:
      "small" = LGB Recall 1: rule retrieval, per-rule quantile norm, min_pos_rate=0.006
      "large" = LGB Recall 2: ALS/BPR/ItemCF/UG rules, norm=False, min-max-sum cap
                rank<=200 stored / rank<=50 trained, score+rank kept as features
  - shared features (their base_features + notebook FE) + w2v similarity
    (their Embeddings.ipynb params: 128d, window 32, sg=1, sample=1e-3, negative=15,
    10 epochs, seed 1; user embd = normalized mean, default vector = ones/128)
  - per strategy: LGB binary (probes) [+ LGB rank + NN at the final stage]
  - final blend: gen_submit cust_blend, weights [large_rank 1.0, large_binary 1.3,
    small_rank 1.0, small_binary 1.3], score W/(position+1)

Forced omissions (documented in SPEC.md): DSSM/YouTubeDNN similarities (their training code
is not in the repo and the published files leak our test period).

Sanctioned adaptations: (1) query agreement features; (2) last-day label rounds
(--rounds last days of train/, newest = validation) instead of calendar weeks.

Feature store: rounds and serve stores are cached under cache/ and reused by every run.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rules  # noqa: E402
from features import (  # noqa: E402
    add_query_features, build_feature_tables, cap_candidates, collect, join_features,
)
from fashion_style_ai_assistant.data import load_catalog_bundle  # noqa: E402

VERSION = "2.0"
SEED = 42

FEMALE_CATEGORIES = [
    "Bra", "Underwear Tights", "Leggings/Tights", "Hair clip", "Hair string",
    "Hair/alice band", "Bikini top", "Skirt", "Dress", "Earring", "Alice band", "Straw hat",
    "Necklace", "Ballerinas", "Blouse", "Beanie", "Giftbox", "Pumps", "Bootie",
    "Heeled sandals", "Nipple covers", "Hair ties", "Underwear corset", "Bra extender",
    "Underdress", "Underwear set", "Sarong", "Leg warmers", "Hairband", "Tote bag",
    "Earrings", "Flat shoes", "Heels", "Cap", "Shoulder bag", "Headband", "Baby Bib",
    "Cross-body bag", "Bumbag",
]
SUMMER = ["Sunglasses", "Hat/brim", "Sandals", "Flat shoe", "Heeled sandals", "Polo shirt",
          "Dress", "T-shirt", "Skirt", "Vest top", "Swimwear top", "Swimsuit",
          "Swimwear bottom", "Bikini top", "Shorts"]
WINTER = ["Beanie", "Felt hat", "Outdoor overall", "Long John", "Pyjama bottom", "Hat/beanie",
          "Leggings/Tights", "Hoodie", "Underwear Tights", "Pyjama set", "Boots", "Cardigan",
          "Sweater", "Jacket", "Scarf", "Coat", "Gloves", "Outdoor Waistcoat"]


# ------------------------------------------------------------------ encoding

def _real_ids(s: pd.Series) -> pd.Series:
    """Restore the catalog's real 10-digit ids (int inference strips leading zeros; the eval
    compares ids as read from CSV). Non-numeric ids (e.g. data/sample) pass through."""

    s = s.astype(str)
    numeric = s.str.isdigit()
    return s.where(~numeric, s.str.zfill(10))


def encode(bundle):
    """Ids -> integer indexes; item/user attribute tables; query-attribute vocabularies."""

    tx = bundle.interactions.copy()
    tx["product_id"] = _real_ids(tx["product_id"])
    bundle.products["product_id"] = _real_ids(bundle.products["product_id"])
    tx = tx.dropna(subset=["event_timestamp"]).sort_values("event_timestamp")
    users_idx = pd.Index(pd.unique(pd.concat([tx["user_id"], bundle.users["user_id"].astype(str)])))
    items_idx = pd.Index(bundle.products["product_id"].astype(str).unique())
    u_map = {u: i for i, u in enumerate(users_idx)}
    i_map = {p: i for i, p in enumerate(items_idx)}

    enc = pd.DataFrame({
        "uidx": tx["user_id"].map(u_map),
        "iidx": tx["product_id"].map(i_map),
        # resolution-independent day number (pandas 3 stores datetime64 in us, not ns)
        "day": (tx["event_timestamp"].dt.floor("D") - pd.Timestamp("1970-01-01")).dt.days,
        "price": pd.to_numeric(tx.get("price", 0.0), errors="coerce").fillna(0),
        "channel": pd.to_numeric(tx.get("sales_channel_id", 0), errors="coerce").fillna(0),
    }).dropna(subset=["uidx", "iidx"])
    enc = enc.astype({"uidx": np.int32, "iidx": np.int32, "day": np.int32,
                      "price": np.float32, "channel": np.int8})

    prod = bundle.products.set_index(bundle.products["product_id"].astype(str)).reindex(items_idx)
    fam = items_idx.str[:-3]  # exact product_code recovery (SPEC 5)

    def codes(col):
        vals = prod[col].fillna("").astype(str).str.lower()
        code, vocab = pd.factorize(vals)
        return code.astype(np.int32), np.array([str(v) for v in vocab], dtype=object)

    cat_code, cat_vocab = codes("category")
    style_code, style_vocab = codes("style_tags")
    occ_code, occ_vocab = codes("occasion_tags")
    color_code, _ = codes("color")
    season_code, _ = codes("season")

    style_raw = prod["style_tags"].fillna("").astype(str)
    cat_raw = prod["category"].fillna("").astype(str)
    gender = np.zeros(len(prod), dtype=np.int8)
    gender[style_raw.str.contains("Menswear", case=False).to_numpy()] = 1
    is_female = style_raw.str.contains("Ladieswear", case=False).to_numpy() | cat_raw.isin(FEMALE_CATEGORIES).to_numpy()
    gender[is_female] = 2
    season_type = np.zeros(len(prod), dtype=np.int8)
    season_type[cat_raw.isin(SUMMER).to_numpy()] = 1
    season_type[cat_raw.isin(WINTER).to_numpy()] = 2

    item_attrs = pd.DataFrame({
        "fidx": pd.factorize(fam)[0].astype(np.int32),
        "cat_code": cat_code, "style_code": style_code, "occ_code": occ_code,
        "color_code": color_code, "season_code": season_code,
        "article_gender": gender, "season_type": season_type,
        "price": enc.groupby("iidx")["price"].median().reindex(range(len(items_idx))).fillna(0).to_numpy(np.float32),
    }, index=pd.RangeIndex(len(items_idx)))

    u = bundle.users.copy()
    u["uidx"] = u["user_id"].astype(str).map(u_map)
    u = u.dropna(subset=["uidx"]).set_index(u["uidx"].astype(np.int64))
    age_code = pd.factorize(u.get("age_band", pd.Series("", index=u.index)).astype(str))[0]
    user_attrs = pd.DataFrame({"age_band_code": age_code.astype(np.int32)}, index=u.index)

    g = enc.merge(item_attrs[["article_gender"]], left_on="iidx", right_index=True)
    ratio = g.groupby(["uidx", "article_gender"]).size().unstack(fill_value=0)
    ratio = ratio.div(ratio.sum(axis=1), axis=0)
    ug = pd.Series(0, index=ratio.index, dtype=np.int8)
    if 1 in ratio: ug[ratio[1] >= 0.8] = 1
    if 2 in ratio: ug[ratio[2] >= 0.8] = 2
    user_attrs["user_gender"] = ug.reindex(user_attrs.index).fillna(0).astype(np.int8)
    # Recall 2 cell 17: purchase_ability = qcut(user mean price, 5)
    mean_price = enc.groupby("uidx")["price"].mean()
    pa = pd.qcut(mean_price.rank(method="first"), 5, labels=False)
    user_attrs["purchase_ability"] = pa.reindex(user_attrs.index).fillna(-1).astype(np.int8)

    vocab = {"cat": cat_vocab, "style": style_vocab, "occ": occ_vocab}
    return enc, item_attrs, user_attrs, users_idx, items_idx, vocab


# ------------------------------------------------------------------ w2v (Embeddings.ipynb)

def w2v_embeddings(enc, n_users, n_items):
    """Exact Embeddings.ipynb: skip-gram 128d, window 32, sample 1e-3, negative 15,
    10 epochs, seed 1; user embd = L2-normalized mean; default vector = ones/128."""

    from gensim.models import Word2Vec
    seqs = enc.sort_values("day").groupby("uidx")["iidx"].apply(lambda s: [str(x) for x in s])
    model = Word2Vec(seqs.tolist(), vector_size=128, window=32, min_count=1, sg=1,
                     sample=1e-3, negative=15, workers=32, seed=1, epochs=10)
    item_embd = np.ones((n_items, 128), dtype=np.float32) / 128
    for key in model.wv.index_to_key:
        vec = model.wv[key]
        item_embd[int(key)] = vec / np.sqrt(np.sum(vec ** 2))
    user_embd = np.ones((n_users, 128), dtype=np.float32) / 128
    for uid, items in seqs.items():
        if len(items) > 1:
            vec = np.mean([model.wv[x] for x in items], axis=0)
        else:
            vec = model.wv[items[0]]
        user_embd[uid] = vec / np.sqrt(np.sum(vec ** 2))
    return user_embd, item_embd


def extra_embeddings(enc, item_embd, n_users, n_items):
    """v2.0 (top1 writeup): w2v item2item neighbor table + ProNE user/item embeddings.
    Neighbors: top-10 cosine among items with any transaction (blocked matmul).
    ProNE on the bipartite user-item graph (their user2item retrieval + similarity)."""

    active = np.sort(enc["iidx"].unique())
    vecs = item_embd[active]  # already L2-normalized
    neighbors = {}
    for b in range(0, len(active), 2048):
        sims = vecs[b:b + 2048] @ vecs.T
        for j in range(sims.shape[0]):
            row = sims[j]
            row[b + j] = -1.0
            top = np.argpartition(-row, 10)[:10]
            top = top[np.argsort(-row[top])]
            neighbors[int(active[b + j])] = (active[top].astype(np.int32), row[top].astype(np.float32))

    from scipy import sparse
    ui = enc.groupby(["uidx", "iidx"]).size()
    rows = ui.index.get_level_values(0).to_numpy()
    cols = ui.index.get_level_values(1).to_numpy() + n_users
    n = n_users + n_items
    data = ui.to_numpy(np.float32)
    adj = sparse.csr_matrix((np.concatenate([data, data]),
                             (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                            shape=(n, n))
    try:
        np.float_ = np.float64  # numpy2 shim for nodevectors
        from nodevectors import ProNE
        embd = ProNE(n_components=64).fit_transform(adj)
        print("prone: nodevectors ProNE", flush=True)
    except Exception as e:  # exact-ProNE unavailable -> spectral fallback, same graph
        print(f"prone: nodevectors failed ({e}); TruncatedSVD spectral fallback", flush=True)
        from sklearn.decomposition import TruncatedSVD
        embd = TruncatedSVD(n_components=64, random_state=SEED).fit_transform(adj)
    embd = embd / (np.linalg.norm(embd, axis=1, keepdims=True) + 1e-12)
    return neighbors, embd[:n_users].astype(np.float32), embd[n_users:].astype(np.float32)


def add_wv_similarity(c, user_embd, item_embd):
    u, i = c["uidx"].to_numpy(), c["iidx"].to_numpy()
    c["wv_similarity"] = np.einsum("ij,ij->i", user_embd[u], item_embd[i]).astype(np.float32)
    return c


# ------------------------------------------------------------------ strategies

class DayContext:
    """User-independent state for one round day, built once (feature-store discipline)."""

    def __init__(self, enc, day, item_attrs, user_attrs, n_users, n_items, extras=None):
        self.day = day
        self.extras = extras
        self.tx = enc[enc["day"] < day]
        tx = self.tx
        self.last = int(tx["day"].max())
        self.item_attrs, self.user_attrs = item_attrs, user_attrs
        self.last_week = tx[tx["day"] >= self.last - 6]
        self.last_3days = tx[tx["day"] >= self.last - 2]
        self.last_2week = tx[tx["day"] >= self.last - 13]
        self.last_60day = tx[tx["day"] >= self.last - 59]
        self.last_80day = tx[tx["day"] >= self.last - 79]
        t0 = time.time()
        # Recall 2 machinery (built once per day)
        self.cf_sims = {
            "1": rules.item_cf_sim(self.last_80day, top_k=10),
            "2": rules.item_cf_sim(self.last_60day, top_k=10),
            "3": rules.item_cf_sim(self.last_2week, top_k=10),
        }
        self.ug_cf_sims = {
            "1": rules.ug_item_cf_sims(self.last_80day, user_attrs["age_band_code"], top_k=10),
            "2": rules.ug_item_cf_sims(self.last_60day, user_attrs["age_band_code"], top_k=10),
            "3": rules.ug_item_cf_sims(self.last_2week, user_attrs["age_band_code"], top_k=10),
            "4": rules.ug_item_cf_sims(self.last_80day, user_attrs["purchase_ability"], top_k=10),
            "5": rules.ug_item_cf_sims(self.last_60day, user_attrs["purchase_ability"], top_k=10),
            "6": rules.ug_item_cf_sims(self.last_2week, user_attrs["purchase_ability"], top_k=10),
        }
        print(f"  ctx day {day}: itemcf sims ({time.time()-t0:.0f}s)", flush=True)
        self.als = rules.als_train(self.last_60day, n_users, n_items, factors=200, iterations=25)
        self.bpr = rules.bpr_train(self.last_80day, n_users, n_items, factors=300, iterations=350)
        print(f"  ctx day {day}: als+bpr ({time.time()-t0:.0f}s)", flush=True)
        self.pool = rules.recent_pool(tx)
        self.oos = rules.out_of_stock(tx)
        self.tables = build_feature_tables(tx, day, item_attrs, user_attrs)
        print(f"  ctx day {day}: tables ({time.time()-t0:.0f}s)", flush=True)

    def users_frame(self, customer_list):
        users = pd.DataFrame({"uidx": customer_list})
        for col in ("age_band_code", "user_gender", "purchase_ability"):
            users[col] = self.user_attrs[col].reindex(users["uidx"]).fillna(0).to_numpy()
        return users


class QueryDayContext:
    """Light context for the query strategy: feature tables + triple tables only
    (no ItemCF/ALS/BPR -- QueryMatch needs none of them). ~2 min/round vs ~6."""

    def __init__(self, enc, day, item_attrs, user_attrs, extras=None):
        self.day = day
        self.extras = extras
        self.tx = enc[enc["day"] < day]
        self.item_attrs, self.user_attrs = item_attrs, user_attrs
        self.triples = rules.build_triple_tables(self.tx, item_attrs, k=200)
        self.tables = build_feature_tables(self.tx, day, item_attrs, user_attrs)


def query_candidates(ctx, customer_list, labels):
    """Third strategy (evidence: diag_ceiling 0.271 -> 0.857; diag_transfer dual ranking).
    Candidates = top-200 of the query triple by recent-4w AND all-time popularity."""

    lab = labels.merge(ctx.item_attrs[["cat_code", "style_code", "occ_code"]],
                       left_on="iidx", right_index=True)
    first = lab.groupby("uidx").first()  # one query per user-round (mirrors q-features)
    user_triples = {int(u): (int(r["cat_code"]), int(r["style_code"]), int(r["occ_code"]))
                    for u, r in first.iterrows()}
    outputs = [
        rules.query_match(user_triples, ctx.triples["recent"], name="recent"),
        rules.query_match(user_triples, ctx.triples["alltime"], name="alltime"),
    ]
    return collect(outputs, labels, np.array([]), min_pos_rate=0.0, norm=False)


def small_candidates(ctx: DayContext, customer_list, labels):
    """LGB Recall 1 cell 17, verbatim rule list; quantile norm; min_pos_rate=0.006."""

    users = ctx.users_frame(customer_list)
    uset = set(np.asarray(customer_list).tolist())
    tx_users = ctx.tx[ctx.tx["uidx"].isin(uset)]
    oh3 = rules.order_history(tx_users, days=3, name="1")
    oh7 = rules.order_history(tx_users, days=7, name="2")
    ohd3 = rules.order_history_decay(tx_users, days=3, n=50, name="1")
    ohd7 = rules.order_history_decay(tx_users, days=7, n=50, name="2")
    outputs = [
        oh3, oh7, ohd3, ohd7,
        rules.item_pair(oh3, name="1"), rules.item_pair(oh7, name="2"),
        rules.item_pair(ohd3, name="3"), rules.item_pair(ohd7, name="4"),
        rules.ug_time_history(ctx.last_week, users, "age_band_code", n=50, name="1"),
        rules.ug_time_history(ctx.last_3days, users, "age_band_code", n=50, name="2"),
        rules.ug_sale_trend(ctx.tx, users, "age_band_code", days=7, n=50, name="1"),
        rules.time_history(ctx.last_week, users, n=50, name="1"),
        rules.time_history(ctx.last_3days, users, n=50, name="2"),
        rules.time_history_decay(ctx.tx, users, days=3, n=50, name="1"),
        rules.time_history_decay(ctx.tx, users, days=7, n=50, name="2"),
        rules.sale_trend(ctx.tx, users, days=7, n=50, name="1"),
    ]
    return collect(outputs, labels, ctx.oos, min_pos_rate=0.006, norm=True)


def large_candidates(ctx: DayContext, customer_list, labels, cap=200):
    """LGB Recall 2 cell 19, verbatim rule list; norm=False, min_pos_rate=0.0;
    min-max-sum cap rank<=200 with score+rank kept as features."""

    users = ctx.users_frame(customer_list)
    uset = set(np.asarray(customer_list).tolist())
    tx_users = ctx.tx[ctx.tx["uidx"].isin(uset)]
    target = tx_users[tx_users["day"] >= ctx.last - 13]  # their last_2week targets
    outputs = []
    if ctx.als[0] is not None:
        outputs.append(rules.mf_predict(ctx.als[0], ctx.als[1], customer_list, ctx.pool, 200, "ALS_1"))
    if ctx.bpr[0] is not None:
        outputs.append(rules.mf_predict(ctx.bpr[0], ctx.bpr[1], customer_list, ctx.pool, 200, "BPR_1"))
    outputs += [
        rules.ug_time_history(ctx.last_week, users, "age_band_code", n=200, scale=True, name="1"),
        rules.ug_time_history(ctx.last_week, users, "purchase_ability", n=200, scale=True, name="2"),
        rules.ug_time_history(ctx.last_week, users, "user_gender", n=200, scale=True, name="3"),
        rules.order_history(tx_users, days=35, n=200, name="3"),
        rules.order_history_decay(tx_users, days=7, n=200, name="3"),
        rules.time_history(ctx.last_week, users, n=200, name="3"),
        rules.item_cf_predict(ctx.cf_sims["1"], target, "ItemCF_1"),
        rules.item_cf_predict(ctx.cf_sims["2"], target, "ItemCF_2"),
        rules.item_cf_predict(ctx.cf_sims["3"], target, "ItemCF_3"),
    ]
    if ctx.extras is not None:  # v2.0: writeup w2v i2i + ProNE u2i retrieval
        outputs.append(rules.w2v_i2i(ctx.extras[0], target, name="1"))
        outputs.append(rules.prone_u2i(ctx.extras[1], ctx.extras[2], customer_list, ctx.pool,
                                       n=100, name="1"))
    for name in ("1", "2", "3"):
        outputs.append(rules.ug_item_cf_predict(
            ctx.ug_cf_sims[name], target, ctx.user_attrs["age_band_code"], name=name))
    for name in ("4", "5", "6"):
        outputs.append(rules.ug_item_cf_predict(
            ctx.ug_cf_sims[name], target, ctx.user_attrs["purchase_ability"], name=name))
    cand = collect(outputs, labels, np.array([]), min_pos_rate=0.0, norm=False)
    if cand.empty:
        return cand
    return cap_candidates(cand, n=cap, keep_score=True)


def finalize_round(ctx, cand, labels, item_attrs, user_attrs, emb):
    """Shared tail: features + label + query features (adaptation 1)."""

    cand = join_features(ctx.tables, cand, item_attrs, user_attrs)
    cand = add_wv_similarity(cand, emb[0], emb[1])
    if getattr(ctx, "extras", None) is not None:
        pu, pi = ctx.extras[1], ctx.extras[2]
        u, i = cand["uidx"].to_numpy(), cand["iidx"].to_numpy()
        cand["prone_similarity"] = np.einsum("ij,ij->i", pu[u], pi[i]).astype(np.float32)
    if labels is not None:
        cand = cand.merge(labels.assign(label=1), on=["uidx", "iidx"], how="left")
        cand["label"] = cand["label"].fillna(0).astype(np.int8)
        pos = labels.merge(item_attrs[["cat_code", "style_code", "occ_code"]],
                           left_on="iidx", right_index=True)
        q = pos.groupby("uidx").first()
        cand = add_query_features(
            cand, item_attrs,
            q["cat_code"].reindex(cand["uidx"]).fillna(-2).to_numpy(),
            q["style_code"].reindex(cand["uidx"]).fillna(-2).to_numpy(),
            q["occ_code"].reindex(cand["uidx"]).fillna(-2).to_numpy(),
        )
    return cand


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description="Train behavioural v5 (silver-exact)")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "processed" / "hm" / "train"))
    parser.add_argument("--rounds", type=int, default=14)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--skip-serve-store", action="store_true")
    parser.add_argument("--serve-chunk", type=int, default=100_000)
    parser.add_argument("--serve-users", type=int, default=None)
    parser.add_argument("--models", choices=("lgb", "ensemble", "top1"), default="lgb",
                        help="probes: LGB binary per strategy; ensemble: + LGB rank (final); "
                             "top1: 5 LGB + 7 CatBoost per strategy (writeup final recipe)")
    parser.add_argument("--max-round-users", type=int, default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    bundle = load_catalog_bundle(args.data_dir)
    enc, item_attrs, user_attrs, users_idx, items_idx, vocab = encode(bundle)
    print(f"encoded: {len(enc):,} tx, {len(users_idx):,} users, {len(items_idx):,} items "
          f"({time.time()-t0:.0f}s)", flush=True)

    emb_cache = cache_dir / "w2v.pkl"
    if emb_cache.exists():
        with emb_cache.open("rb") as fh:
            emb = pickle.load(fh)
        print("w2v embeddings [cached]", flush=True)
    else:
        emb = w2v_embeddings(enc, len(users_idx), len(items_idx))
        with emb_cache.open("wb") as fh:
            pickle.dump(emb, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"w2v embeddings done ({time.time()-t0:.0f}s)", flush=True)

    extras_cache = cache_dir / "emb2.pkl"
    if extras_cache.exists():
        with extras_cache.open("rb") as fh:
            extras = pickle.load(fh)
        print("v2 embeddings (neighbors+prone) [cached]", flush=True)
    else:
        extras = extra_embeddings(enc, emb[1], len(users_idx), len(items_idx))
        with extras_cache.open("wb") as fh:
            pickle.dump(extras, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"v2 embeddings (neighbors+prone) done ({time.time()-t0:.0f}s)", flush=True)

    max_day = int(enc["day"].max())
    strategy_fns = {"small": small_candidates, "large": large_candidates,
                    "query": query_candidates}
    frames = {s: [] for s in strategy_fns}
    for r in range(args.rounds):
        day = max_day - r
        caches = {s: cache_dir / f"{s}_day{day}.parquet" for s in strategy_fns}
        missing = [s for s in strategy_fns if not caches[s].exists()]
        for s in strategy_fns:
            if s not in missing:
                cand = pd.read_parquet(caches[s])
                cand["round"] = r
                frames[s].append(cand)
        if not missing:
            print(f"round {r} (day {day}): cached", flush=True)
            continue
        day_tx = enc[enc["day"] == day]
        if day_tx.empty:
            continue
        labels = day_tx[["uidx", "iidx"]].drop_duplicates()
        customer_list = np.sort(labels["uidx"].unique())
        # heavy context only if a heavy strategy is missing; query needs only the light one
        heavy_missing = [s for s in missing if s in ("small", "large")]
        ctx = (DayContext(enc, day, item_attrs, user_attrs, len(users_idx), len(items_idx),
                          extras=extras)
               if heavy_missing else None)
        qctx = (QueryDayContext(enc, day, item_attrs, user_attrs, extras=extras)
                if "query" in missing else None)
        if ctx is not None and qctx is not None:
            qctx.tables = ctx.tables  # same day -> identical feature tables, reuse
        for s in missing:
            use_ctx = qctx if s == "query" else ctx
            fn = strategy_fns[s]
            parts = []
            for b in range(0, len(customer_list), 10_000):  # bounded memory under the 60G cap
                sub = customer_list[b:b + 10_000]
                sub_labels = labels[labels["uidx"].isin(set(sub.tolist()))]
                cand = fn(use_ctx, sub, sub_labels)
                if cand.empty:
                    continue
                parts.append(finalize_round(use_ctx, cand, sub_labels, item_attrs, user_attrs, emb))
            if not parts:
                continue
            cand = pd.concat(parts, ignore_index=True)
            del parts
            cand.to_parquet(caches[s], index=False)
            print(f"round {r} (day {day}) {s}: {len(cand):,} rows, "
                  f"{int(cand['label'].sum()):,} pos ({time.time()-t0:.0f}s)", flush=True)
            cand["round"] = r
            frames[s].append(cand)
        del ctx, qctx
        import gc
        gc.collect()

    # run-specific user sampling (deterministic, independent of cache state)
    def sample_users(cand, r):
        if not args.max_round_users:
            return cand
        per_round = max(args.max_round_users // args.rounds, 1)
        round_users = np.sort(pd.unique(cand["uidx"]))
        if len(round_users) <= per_round:
            return cand
        keep = np.random.default_rng(SEED + r).choice(round_users, per_round, replace=False)
        return cand[cand["uidx"].isin(set(keep.tolist()))]

    strategies = {}
    for s in ("small", "large", "query"):
        rounds_s = [sample_users(f, int(f["round"].iloc[0])) for f in frames[s] if len(f)]
        full = pd.concat(rounds_s, ignore_index=True)
        if s == "large":  # Recall 2 trains on rank<=50 of the stored rank<=200
            full = full[full["rank"] <= 50]
        feats = [c for c in full.columns if c not in ("label", "round")]
        cat_feats = [c for c in ("uidx", "iidx", "fidx", "cat_code", "color_code",
                                 "season_code", "age_band_code", "article_gender",
                                 "user_gender", "season_type", "purchase_ability")
                     if c in feats]
        train_df = full[full["round"] > 0]
        valid_df = full[full["round"] == 0]
        print(f"[{s}] train {len(train_df):,} rows ({int(train_df['label'].sum()):,} pos) | "
              f"valid {len(valid_df):,}", flush=True)

        from fashion_style_ai_assistant.behavioural.ranking import model_frame
        X_tr, X_va = model_frame(train_df, feats, cat_feats), model_frame(valid_df, feats, cat_feats)
        import lightgbm as lgb_mod
        models = {}
        m = LGBMClassifier(objective="binary", max_depth=8, num_leaves=128, learning_rate=0.03,
                           n_estimators=300, random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(X_tr, train_df["label"], eval_set=[(X_va, valid_df["label"])], eval_metric="auc",
              categorical_feature=cat_feats,
              callbacks=[lgb_mod.early_stopping(30, verbose=False)])
        models["binary"] = m
        print(f"[{s}] lgb binary done ({time.time()-t0:.0f}s)", flush=True)
        if args.models == "top1":
            # writeup final recipe: 5 lightgbm + 7 catboost, different seeds AND different
            # training data (negative downsampling); ranks fused by cust_blend at serve
            from catboost import CatBoostClassifier
            models.pop("binary")
            pos_df = train_df[train_df["label"] == 1]
            neg_df = train_df[train_df["label"] == 0]
            for k in range(12):
                rng = np.random.default_rng(SEED + 100 + k)
                keep = rng.random(len(neg_df)) < 0.8
                sub = pd.concat([pos_df, neg_df[keep]], ignore_index=False)
                X_k = model_frame(sub, feats, cat_feats)
                y_k = sub["label"]
                if k < 5:
                    mk = LGBMClassifier(objective="binary", max_depth=8, num_leaves=128,
                                        learning_rate=0.03, n_estimators=300,
                                        random_state=SEED + k, n_jobs=-1, verbose=-1)
                    mk.fit(X_k, y_k, eval_set=[(X_va, valid_df["label"])], eval_metric="auc",
                           categorical_feature=cat_feats,
                           callbacks=[lgb_mod.early_stopping(30, verbose=False)])
                    models[f"lgb{k}"] = mk
                else:
                    cat_idx = [X_k.columns.get_loc(c) for c in cat_feats]
                    mk = CatBoostClassifier(depth=8, learning_rate=0.03, iterations=300,
                                            random_seed=SEED + k, eval_metric="AUC",
                                            early_stopping_rounds=30, verbose=False,
                                            allow_writing_files=False)
                    mk.fit(X_k, y_k, eval_set=(X_va, valid_df["label"]), cat_features=cat_idx)
                    models[f"cbt{k - 5}"] = mk
                print(f"[{s}] top1 model {k + 1}/12 done ({time.time()-t0:.0f}s)", flush=True)
        if args.models == "ensemble":
            # their "ranker": same params trained on week-grouped data (see notebook cell 63)
            m2 = LGBMClassifier(objective="binary", max_depth=8, num_leaves=128,
                                learning_rate=0.03, n_estimators=300, random_state=SEED + 1,
                                n_jobs=-1, verbose=-1)
            m2.fit(X_tr, train_df["label"], eval_set=[(X_va, valid_df["label"])],
                   eval_metric="auc", categorical_feature=cat_feats,
                   callbacks=[lgb_mod.early_stopping(30, verbose=False)])
            models["rank"] = m2
            print(f"[{s}] lgb rank done ({time.time()-t0:.0f}s)", flush=True)
        strategies[s] = {"models": models, "feats": feats, "cat_feats": cat_feats}

    name = "behavioural"
    if args.max_round_users:
        name += f"_u{max(args.max_round_users // 1000, 1)}k"
    if args.models == "lgb":
        name += "_lgb"
    strat_w = {"small": 0.3, "large": 0.3, "query": 4.0}  # v2.0 tuned blend
    blend = {f"{s}_{m}": strat_w[s] / len(d["models"])
             for s, d in strategies.items() for m in d["models"]}
    weights = {
        "name": name, "version": VERSION, "strategies": strategies,
        "blend_weights": blend,
        "users_idx": users_idx.to_numpy(), "items_idx": items_idx.to_numpy(),
        "vocab": vocab,
        "item_query_codes": item_attrs[["cat_code", "style_code", "occ_code"]],
    }
    with (out_dir / "weights.pkl").open("wb") as fh:
        pickle.dump(weights, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"weights saved ({time.time()-t0:.0f}s)", flush=True)

    # ---- query-serve artifacts (third strategy has no store; serving assembles rows)
    qserve_path = out_dir / "query_serve.pkl"
    if not qserve_path.exists():
        qctx = QueryDayContext(enc, max_day + 1, item_attrs, user_attrs, extras=extras)
        n_i, n_u = len(items_idx), len(users_idx)
        # item-side + user-side frames through the REAL join path (zero drift):
        all_items = pd.DataFrame({"uidx": np.zeros(n_i, np.int32),
                                  "iidx": np.arange(n_i, dtype=np.int32)})
        item_frame = join_features(qctx.tables, all_items, item_attrs, user_attrs)
        user_cols = ["customer_id_last_time", "customer_id_first_time",
                     "customer_id_time_mean", "customer_id_gap", "user_mean_price",
                     "age_band_code", "user_gender", "purchase_ability"]
        all_users = pd.DataFrame({"uidx": np.arange(n_u, dtype=np.int32),
                                  "iidx": np.zeros(n_u, np.int32)})
        user_frame = join_features(qctx.tables, all_users, item_attrs, user_attrs)[["uidx"] + user_cols]
        ui = enc.groupby(["uidx", "iidx"]).size().astype(np.float32)
        ui_key = (ui.index.get_level_values(0).to_numpy(np.int64) << 32) |                  ui.index.get_level_values(1).to_numpy(np.int64)
        qserve = {
            "triples": qctx.triples,
            "item_frame": item_frame.set_index(all_items["iidx"].to_numpy()),
            "user_frame": user_frame.set_index("uidx"),
            "user_cols": user_cols,
            "ui_key": ui_key, "ui_val": ui.to_numpy(np.float32),
            "w2v_user": emb[0], "w2v_item": emb[1],
            "prone_user": extras[1], "prone_item": extras[2],
            "u_cat": qctx.tables["u_cat"], "u_fam": qctx.tables["u_fam"],
        }
        with qserve_path.open("wb") as fh:
            pickle.dump(qserve, fh, protocol=pickle.HIGHEST_PROTOCOL)
        del qserve, item_frame, user_frame, qctx
        print(f"query_serve artifacts saved ({time.time()-t0:.0f}s)", flush=True)
    else:
        print("query_serve artifacts exist, reusing", flush=True)

    # ---- serve stores: day 0, all users, per strategy, cached
    if not args.skip_serve_store:
        import gc
        import json
        need = {s: out_dir / f"serve_{s}" for s in ("small", "large")}
        done = all((d / "meta.json").exists() for d in need.values())
        if done:
            print("serve stores exist, reusing")
        else:
            all_users = np.arange(len(users_idx), dtype=np.int32)
            if args.serve_users:
                all_users = all_users[: args.serve_users]
            ctx = DayContext(enc, max_day + 1, item_attrs, user_attrs,
                             len(users_idx), len(items_idx), extras=extras)
            print(f"serve context ready ({time.time()-t0:.0f}s)", flush=True)
            for s, fn in (("small", small_candidates), ("large", large_candidates)):
                sdir = need[s]
                if (sdir / "meta.json").exists():
                    continue
                sdir.mkdir(parents=True, exist_ok=True)
                n_rows = 0
                for b in range(0, len(all_users), args.serve_chunk):
                    idx = b // args.serve_chunk
                    part_path = sdir / f"chunk_{idx:04d}.parquet"
                    if part_path.exists():
                        continue
                    chunk = all_users[b:b + args.serve_chunk]
                    cand = fn(ctx, chunk, None)
                    if cand.empty:
                        continue
                    cand = finalize_round(ctx, cand, None, item_attrs, user_attrs, emb)
                    cand.to_parquet(part_path, index=False)
                    n_rows += len(cand)
                    del cand
                    gc.collect()
                    print(f"serve {s} chunk {idx}: written ({time.time()-t0:.0f}s)", flush=True)
                (sdir / "meta.json").write_text(json.dumps(
                    {"chunk_size": args.serve_chunk, "n_users": int(len(users_idx)),
                     "rows": n_rows}))
                print(f"serve store {s}: {n_rows:,} rows -> {sdir} ({time.time()-t0:.0f}s)", flush=True)

    print(f"DONE behavioural v{VERSION} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
