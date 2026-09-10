# Behavioural v5 — EXACT port of the silver-medal solution

**Target: the silver-medal solution (decision 2026-08-10).** The 1st place
published no code, making "exact" impossible against it; the silver repo
`reference/H-M-Fashion-RecSys/` is complete and runnable, so it is the match target.

**Silver architecture, verbatim:**
- **Two recall strategies**, separate pipelines:
  - `small` = LGB Recall 1 notebook: rule list of cell 17 (OrderHistory 3/7, OrderHistoryDecay
    3/7 n=50, ItemPair ×4, UGTimeHistory age ×2 n=50, UGSaleTrend age, TimeHistory ×2 n=50,
    TimeHistoryDecay 3/7 n=50, SaleTrend 7d), per-rule quantile norm, min_pos_rate=0.006,
    OutOfStock filter
  - `large` = LGB Recall 2 notebook: cell 19 (ALS last-60d n=200 it=25, BPR last-80d n=200
    it=350, UGTimeHistory last-week × age/purchase_ability/user_gender n=200 scale=True,
    OrderHistory 35d n=200, OrderHistoryDecay 7d n=200, TimeHistory last-week n=200,
    ItemCF 80/60/14d → last-2-week targets k=10 ×3, UGItemCF age ×3 + purchase_ability ×3),
    norm=False, min_pos_rate=0.0, min-max-sum cap rank≤200 stored (score+rank kept as
    features), rank≤50 at training
- **w2v** (their Embeddings.ipynb, exact): skip-gram 128d, window 32, sample 1e-3,
  negative 15, 10 epochs, seed 1; user embd = normalized mean; default vector ones/128;
  used as the `wv_similarity` FEATURE (not a retrieval rule)
- **Models per strategy**: LGB binary (probes; user's staging rule) + LGB rank + NN at the
  final stage. LGB params: binary/auc, depth 8, 128 leaves, lr 0.03, 300 rounds, early stop 30
- **Final blend**: gen_submit `cust_blend` — item score = Σ W/(position+1) over the ranked
  lists, weights [large_rank 1.0, large_binary 1.3, small_rank 1.0, small_binary 1.3]

**Forced omissions (documented, not choices):** DSSM / YouTubeDNN similarity features — their
training code is NOT in the repo (only pretrained .npy downloads, which were fitted on data
overlapping our test period = disqualifying leakage). ProNE and w2v-as-retrieval (1st-place
writeup items) are NOT in silver and are therefore dropped.

**Sanctioned adaptations — exactly two:**
1. **Query as additional feature.** Kaggle had no queries; our eval sends one built from the
   held-out product's `style_tags / category / occasion_tags`. Training mirrors that
   construction from the label item (train/ data only).
2. **Last-purchase labels, not week labels.** Kaggle predicted a fixed calendar week for
   everyone; our eval predicts each user's last purchase date. Their "6 weeks of labels" →
   our **last 6 purchase DATES per user**, each an event labeled with everything bought that
   date, features computed only from strictly earlier data (point-in-time, per the writeup's
   causal discipline).

Everything else follows the reference. Deviations beyond the two above are defects.

---

## 1. Retrieval (writeup: "recent popularity, repurchase, itemcf, embedding retrieval")

Rules ported 1:1 from `reference/H-M-Fashion-RecSys/src/retrieval/rules.py`, parameters from
the notebooks (`LGB Recall 1` cell 17 + `LGB Recall 2` cell 19), target ≈100 candidates/user
(writeup: "retrieve 100 candidates per user"):

| writeup strategy | rules (reference params) |
|---|---|
| repurchase | `OrderHistory(days=3)`, `OrderHistory(days=7)`, `OrderHistoryDecay(days=3, n=50)`, `OrderHistoryDecay(days=7, n=50)` — decay formula `a/√x + b·e^(−cx) − d`, a=2.5e4 b=1.5e5 c=0.2 d=1e3, ×period-sale quotient, threshold >150 |
| bought-together | `ItemPair` over each OrderHistory variant (most-frequent co-purchase partner) |
| recent popularity | `TimeHistory(last_week, n=50)`, `TimeHistory(last_3days, n=50)`, `TimeHistoryDecay(days=3, n=50)`, `TimeHistoryDecay(days=7, n=50)`, `SaleTrend(days=7, t=0.8)` |
| segment popularity | `UserGroupTimeHistory(['age_bins'], last_week/last_3days, n=50)`, `UserGroupSaleTrend(['age_bins'], days=7)` — age_bins = our `age_band` (already binned) |
| itemcf | `ItemCF(top_k=10)` with direction factor 1.0/0.9, distance factor 0.7^(d−1), popularity factor 1/log(1+len) — history windows 80/60/14 days as in Recall 2 |
| embedding retrieval | w2v item2item (train word2vec on purchase sequences, gensim); ProNE user2item — see §5 flag |
| filter | `OutOfStock`: items whose last-month sales dropped >80% vs prior month or hit 0 |

Collector semantics from `collector.py`: per-rule **quantile normalization** of scores,
`min_pos_rate=0.006` pruning (a rule whose candidates' positive rate on the label events is
below threshold gets trimmed to its best top-n or skipped), union pivoted so **each rule's
score becomes a feature column** (`aggfunc=sum`).

## 2. Features (writeup table, mechanics from `base_features.py` + notebook cells 26–57)

Item-unit = both `product_id` (their article) and product family (their `product_code` — §5
flag). All computed per event-date with only prior data:

- **Count:** `period_sale` at 14/21/28-day windows (+rank +norm), `week_sale` current/last
  (+unique variants), `full_sale` cumulative, sale ratios (item/family, unique ratios),
  `i_sale_trend`/`p_sale_trend`, per-`category` sales & trend
- **Time:** `first_dat` (item age in days), `article_time_mean`, user first/last/mean
  event-time, `customer_id_gap`
- **Popularity:** time-decayed `popularity` (Σ 1/(days+1))
- **Repurchase:** `repurchase_ratio` per item and family
- **Mean/Max/Min & Difference/Ratio:** price/sales-channel aggregations from last event window
  (`merge_week_data` trans_info), user-vs-item price ratio
- **Similarity:** itemcf score (rule column), w2v user·item cosine; ProNE u2i (§5)
- **Categorical:** `customer_id`, `product_id`, family, `category`, `color`, `season`,
  `age_band`, `article_gender` + `user_gender` (ported gender heuristics from
  `datahelper._base_features`, driven by our `category`/`style_tags` values), `season_type`
  (summer/winter category lists), `purchase_ability` (qcut-5 of user mean price)
- **Query (adaptation 1):** `cat_eq`, `style_eq`, `occ_eq`, query-match fraction — candidate
  attrs vs query attrs; at train time query attrs = label item's attrs, as the eval builds them

## 2a. The full feature table (one row = one (user, candidate) pair; ~105 columns)

User decision (2026-08-09): the query enters as the **agreement features only** (H) — the
query's raw attribute codes are deliberately NOT features.

**A. Retrieval-rule scores** (quantile-normalized; 0 = rule didn't retrieve this candidate):
`OrderHistory_1/2/3` (3d/7d/35d own-history) · `OrderHistoryDecay_1/2` (3d/7d decayed
repurchase value) · `ItemPairRetrieve_1..4` (top co-purchase partner) · `UGTimeHistory_1..4`
(age-band 7d/3d, spend-tier, gender popularity) · `UGSaleTrend_1` · `TimeHistory_1/2` (global
7d/3d) · `TimeHistoryDecay_1/2` · `SaleTrend_1` · `ItemCF_1/2/3` (80/60/14d) · `UGItemCF_1..6`
(age-band ×3, spend-tier ×3) · `ALS_1` · `BPR_1` · `W2V_1` · `ProNE_1`

**B. Identity & categoricals** (GBDT categorical features): `uidx` `iidx` `fidx` `cat_code`
`color_code` `season_code` `article_gender` `season_type` `age_band_code` `user_gender`
`purchase_ability`

**C. Windowed sales** (item `i_` + family `p_`, windows 7/14/21/28d, each with `_sale`,
`_sale_rank`, `_sale_norm`): `i_1w_* p_1w_* i_2w_* p_2w_* i_3w_* p_3w_* i_4w_* p_4w_*`

**D. Current-vs-prior week**: `i_sale i_sale_uni lw_i_sale lw_i_sale_uni p_sale p_sale_uni
lw_p_sale lw_p_sale_uni` · ratios `i_sale_ratio i_sale_uni_ratio lw_i_sale_ratio
lw_i_sale_uni_ratio i_uni_ratio p_uni_ratio lw_i_uni_ratio lw_p_uni_ratio` · trends
`i_sale_trend p_sale_trend` · category level `cat_sale lw_cat_sale cat_sale_trend`

**E. Repurchase / age / cumulative**: `i_repurchase_ratio p_repurchase_ratio first_dat
i_full_sale p_full_sale i_daily_sale p_daily_sale i_daily_sale_ratio i_w_full_sale_ratio
i_2w_full_sale_ratio p_w_full_sale_ratio p_2w_full_sale_ratio i_week_above_daily_sale
i_2w_week_above_daily_sale p_2w_week_above_daily_sale i_pop p_pop`

**F. User-item time & price**: `article_time_mean customer_id_last_time customer_id_first_time
customer_id_time_mean customer_id_gap item_price item_channel user_mean_price price_ratio
ui_sale ui_sale_ratio`

**G. Embedding similarities**: `wv_similarity` (w2v user·item) · `prone_similarity` (ProNE)

**H. Query agreement (adaptation 1)**: `q_cat_eq q_style_eq q_occ_eq` (candidate attribute ==
query attribute) · `q_match` (0-3 sum). At train time the query = the label item's attributes
(mirroring the eval's query construction); at serve time parsed from the query string.

**Label** (train only): 1 iff the candidate is what the user bought on the round day.

## 3. Labels & downsampling (writeup: "6 weeks", "1M–2M negatives, seed 42")

User's translation (2026-08-09): Kaggle "uses week1-97 to predict w98 and so forth for the
last 6 weeks; for us it is last-DAY prediction — use the last 14 days as label rounds and all
previous days to predict."

- Rounds: the last **14 calendar days** of train/, one round per day. Round day D: candidates
  + features from data **strictly before D** (point-in-time); labels = purchases on D by that
  day's buyers. Kaggle's sliding weekly window, at day granularity.
- The most recent day is the **validation round** (their "last week as validation"); the other
  13 are training rounds.
- Negative downsampling per the writeup: keep all positives, sample negatives with seed 42 —
  ~150k/round ≈ their 1–2M per-training-set budget at our scale.
- Serving ("test round", day 0): candidates + features as-of the day after train/ ends, for
  all users — the Kaggle gen_submit batch-inference pattern, precomputed at train time.

## 4. Model (writeup: "5 LightGBM + 7 CatBoost classifiers")

- LightGBM: `objective=binary, metric=auc, max_depth=8, num_leaves=128, learning_rate=0.03`,
  300 rounds, early stopping 30 on the validation round (silver params — writeup gives none)
- Ensemble: **5 LGB + 7 CatBoost**, seeds varied, equal-weight mean probability
- Serving: same retrieval union for the (user, query), same features, ensemble score, top-k

## 5. Data-pool mapping — RESOLVED (user decisions 2026-08-09)

| Kaggle input | resolution |
|---|---|
| `FN, Active, club_member_status, fashion_news_frequency, postal_code` | **dropped — user: "not important."** Not in the pool, not used. |
| numeric `age` | use `age_band` directly (their age_bins ≈ same binning) |
| `product_code` (article family) | **exact recovery**: family = `product_id` minus last 3 digits (H&M id structure); dedup remapped transactions so family aggregates are unchanged |
| embedding retrieval | **faithful — install whatever is needed** (user's standing instruction): w2v item2item via `gensim`, ProNE user2item via pip (`nodevectors`) or paper implementation. No substitutions. |

## v2.0 (2026-08-12) — top1-solution adoption ("adapt behaviour scheme using the top1 kaggle solution principle and design")

Evidence first (validation round day 18526, 271 purchases): union candidate coverage 0.605,
eval conversion 0.46 of ceiling → both levers live, features first.

**Feature blocks added (top1 writeup)** — computed in `build_feature_tables`/`join_features`, all strategies:
- Buyer profile (their Difference/Ratio): `i_buyer_age_mean/std`, `i_buyer_pa_mean`, `u_age_diff`, `u_pa_diff`
- Same-window-last-year (their seasonal count): `i_ly_sale`, `p_ly_sale` (day−372..day−358), `i_ly_ratio`
- Time-weighted user affinities (their time-weighted counts): `u_cat_affinity`, `u_fam_affinity` (1/(gap+1) decayed)
- `prone_similarity` (their ProNE u2i cosine) on all candidate rows

**Retrieval added to strategy "large" (their retrieval list):** `w2v_i2i` (top-10 cosine neighbors of
last-2wk items, w2v item vectors) + `prone_u2i` (top-100 of recent pool by 64-d spectral embedding of
the bipartite user-item graph; nodevectors ProNE with TruncatedSVD fallback — fallback used locally).

**Ranker = their final ensemble:** per strategy **5 LightGBM + 7 CatBoost** (`--models top1`), each
with a different seed AND different training data (negatives independently downsampled to 80% per
model — their negative sampling); fused at serve by cust_blend with the strategy weight split evenly
across the 12 models (rank-mean ≙ their blend-of-submissions). Strategy weights tuned on the
validation round: {small 0.3, large 0.3, query 4.0}. weights.pkl 2.5 GB.

**Serving:** query-strategy row assembly overwrites the pair-level features per case
(`prone_similarity`, `u_age_diff`, `u_pa_diff`, `u_cat_affinity`, `u_fam_affinity`); qserve carries
prone arrays + u_cat/u_fam tables. v1.0 artifacts preserved in `v1.0_backup/`.

**Board (10k cohort, all cases): v2.0 recall@5 = 0.3216, map 0.1958, ndcg 0.2270** (v1.0: 0.2763/0.1565/0.1861).
Intermediate probe (features only, single LGB per strategy) measured 0.3007 — the ensemble adds +2.1pp.
NOTE: the 97-user validation round cannot resolve ~2pp effects (ensemble tied there, won on the full board).
Eval ~3.6 h at ~1.2 s/case (36 predicts/case); long evals launch DETACHED (Start-Process) — session-tied
background evals die on session restart.
