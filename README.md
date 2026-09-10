# Fashion Style AI Assistant

An applied recommendation system for fashion product discovery that combines:

- catalog search, vector-style retrieval, and purchase-history candidate generation
- user preference ranking through a weighted baseline or trained behavioural ensembles
- grounded RAG-style product explanations
- business metrics such as stock availability, margin proxy, return-risk proxy, and recommendation diversity
- a reproducible CLI, optional Streamlit/FastAPI apps, and tests

The repo includes a runnable sample-data baseline and a trained behavioural assistant evaluated on a 10,000-user H&M holdout. Dataset ingestion scripts also support normalized outputs from Polyvore, RentTheRunway/ModCloth, Amazon Reviews, Fashion Product Images, and theLook eCommerce.

---

## 1. Quickstart

```bash
uv sync --all-extras
source .venv/bin/activate # optional mac activation
.venv\Scripts\activate.bat # optional ws activation
deactivate # optional deactivate
```


Run a recommendation query with the sample-data baseline:

```bash
uv run fashion-assistant --config config.yaml demo --user-id u001 --top-k 1 --query "I need a relaxed light outfit for a beach weekend under 90 dollars"
```

Run the offline evaluation after preparing the H&M split and the selected assistant's artifacts:

```bash
uv run python scripts/eval/evaluate.py --config config.yaml --data-dir data/processed/hm \
  --assistant fashion_style_ai_assistant.baseline:build
uv run python scripts/eval/evaluate.py --config config.yaml --data-dir data/processed/hm \
  --assistant fashion_style_ai_assistant.behavioural:build
```

The baseline requires `models/baseline/user_profiles.csv`; the behavioural assistant requires its
trained weights, query-serving artifact, and candidate stores. See [`STRUCTURE.md`](STRUCTURE.md)
for artifact preparation and [`models/behavioural/SPEC.md`](models/behavioural/SPEC.md) for the
behavioural training design. The `--models top1` training option builds the 5-LightGBM/7-CatBoost
ensemble per strategy; the trainer defaults to a single LightGBM model per strategy.

Evaluation `top_k` comes from the top-level `default_top_k` setting in `config.yaml`;
`sample_users` and `sample_seed` come from its `evaluation` block. These settings are not
overridable from the evaluation command line. The checked-in configuration evaluates all cases
at K=5, and the leaderboard rejects runs with a different dataset/configuration key.
See [`scripts/eval/README.md`](scripts/eval/README.md) for the full submission spec.

`config.yaml` is the runtime config for data location, baseline retrieval settings, and baseline
ranking weights. The CLI defaults to `--assistant baseline`. Once behavioural artifacts exist,
run `uv run fashion-assistant --data-dir data/processed/hm/train demo --assistant behavioural --user-id <hm-user-id> --query "..."`
with a real H&M user ID.

Run tests:

```bash
uv run pytest -q
```

Optional Streamlit app (baseline assistant):

```bash
uv run streamlit run app/streamlit_app.py
```

Optional API (baseline assistant):

```bash
uv run uvicorn app.api:app --reload
```

---

## 2. System overview

The system follows a retrieval, ranking, and explanation workflow:

1. **Intent parsing**: extract occasion, color, style, category, and budget hints from a natural-language request. The baseline uses these hints for retrieval and ranking; the behavioural ranker separately matches category/style/occasion vocabulary from the query.
2. **Candidate retrieval**: the baseline retrieves catalog items with TF-IDF. The behavioural assistant combines three strategies: `small` (recent purchases, co-purchases, and popularity), `large` (ALS/BPR, item collaborative filtering, and embedding candidates), and `query` (popular items matching query attributes). Behavioural serving reads precomputed candidate features from Parquet stores and assembles query-dependent candidates at request time.
3. **Personalized ranking**: the baseline combines retrieval, intent match, user preferences, price, inventory, margin, and return-risk scores. The behavioural assistant scores user–candidate features with trained models and combines their ranked lists using weighted reciprocal-rank fusion; its `--models top1` training mode uses five LightGBM and seven CatBoost classifiers per strategy.
4. **Grounded explanation**: generate product explanations from retrieved catalog/review/inventory evidence rather than unsupported claims.
5. **Metric loop**: evaluate each assistant through `recommend_dataframe(query, user_id, top_k)` against one held-out purchase per user. Report Recall@K, Precision@K, MAP@K, NDCG@K, stock availability, and category diversity; the leaderboard ranks by Recall@K.

Stored results on all 10,000 H&M holdout cases at K=5: baseline v1.0 **Recall@5 0.97%**,
behavioural v1.0 **27.63%**, and behavioural v2.0 **32.16%**. The v2.0 run also records
**MAP@5 0.195822** and **NDCG@5 0.226995**. These are saved offline measurements; see
[`artifacts/leaderboard.html`](artifacts/leaderboard.html) and its
[`JSON records`](artifacts/leaderboard.json).

---

## 3. Repo layout

```text
fashion-style-ai-assistant/
├── app/                         # Optional Streamlit and FastAPI surfaces over baseline
├── artifacts/                   # Saved evaluation reports and leaderboard
├── config.yaml                  # Runtime config used by CLI/app/API
├── data/
│   ├── sample/                  # Tiny sample data that makes the repo runnable
│   ├── external/                # Place real downloaded datasets here; gitignored
│   └── processed/hm/            # Normalized H&M full/train/test data and split manifest
├── examples/                    # Example CLI output
├── models/
│   ├── baseline/                # Profile precomputation and generated user_profiles.csv
│   └── behavioural/             # Training, features, rules, weights, and candidate stores
├── scripts/
│   ├── data_ingest/             # Dataset normalization and H&M splitting
│   ├── eval/                    # Fixed evaluator, metrics, reports, and leaderboard
│   └── run_demo.py              # Sample demo entry point
├── src/fashion_style_ai_assistant/
│   ├── baseline/                # build() factory, pipeline, TF-IDF retrieval, weighted ranking
│   ├── behavioural/             # build() factory, pipeline, candidate stores, model ranking
│   ├── business.py              # Business KPI proxy helpers
│   ├── cli.py                   # CLI entry point
│   ├── config.py                # Config loading
│   ├── data.py                  # Dataset loaders and validation
│   ├── intent.py                # Query parsing
│   ├── rag.py                   # Grounded recommendation explanations
│   └── schemas.py               # Typed dataclasses
└── tests/                       # Unit tests
```

---

## 4. Public datasets

Use the small sample dataset for the demo. H&M supplies the current purchase-retrieval benchmark;
the other datasets provide additional catalog, review, outfit, or business data:

| Dataset | Best use in this project | Link |
|---|---|---|
| H&M Personalized Fashion Recommendations | Main catalog + user transactions + recommendation holdout evaluation | https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data |
| Polyvore Outfits Dataset | Outfit compatibility, complete-the-look, multimodal retrieval | https://mariya.fyi/polyvore |
| UCSD Clothing Fit Data: RentTheRunway / ModCloth | Fit-risk prediction, review-grounded size/fit explanations | https://cseweb.ucsd.edu/~jmcauley/datasets.html |
| Amazon Reviews 2023 | Review RAG, product metadata, bought-together links, product search | https://amazon-reviews-2023.github.io/ |
| Fashion Product Images Small / Myntra | Fast CV/multimodal search demo with ~44K product images | https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small |
| BigQuery theLook eCommerce | Business KPI layer: orders, inventory, events, returns, funnel SQL | https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce |

See [`DATASETS.md`](DATASETS.md) for integration notes and caveats.

The recorded H&M split starts from **31,788,324 transactions** and selects **10,000 users**,
stratified by age band and budget tier with preference for users with more purchase dates.
It retains **2,208,989 training interactions**, holds out one purchase from each user's last
purchase date, and discards the other **19,212 purchases** on those dates. See
[`split_manifest.json`](data/processed/hm/split_manifest.json). The evaluator constructs each query
from the held-out product's style, category, and occasion fields, so Recall@5 measures
query-assisted retrieval of that purchase.

---

Copyright © 2026 Jerry Huang. All rights reserved.
