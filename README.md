# Fashion Style AI Assistant

An applied recommendation system for fashion product discovery that combines:

- catalog search and vector-style retrieval
- user preference ranking
- grounded RAG-style product explanations
- business metrics such as stock availability, margin proxy, return-risk proxy, and recommendation diversity
- a reproducible CLI, optional Streamlit/FastAPI apps, and tests

The repo runs end-to-end on a small sample dataset and can be pointed at normalized outputs from larger public datasets such as H&M, Polyvore, RentTheRunway/ModCloth, Amazon Reviews, Fashion Product Images, and theLook eCommerce.

---

## 1. Quickstart

```bash
uv sync --all-extras
source .venv/bin/activate # optional mac activation
.venv\Scripts\activate.bat # optional ws activation
deactivate # optional deactivate
```


Run a recommendation query:

```bash
uv run fashion-assistant --config config.yaml demo --user-id u001 --top-k 1 --query "I need a relaxed light outfit for a beach weekend under 90 dollars"
```

Run the offline evaluation:

```bash
uv run python scripts/eval/evaluate.py --config config.yaml --data-dir data/processed/hm
```

`top_k`, `sample_users`, and `sample_seed` come from the `evaluation` block in `config.yaml` and
are not overridable from the command line, so every run lands on one comparable leaderboard.
See [`scripts/eval/README.md`](scripts/eval/README.md) for the full submission spec.

`config.yaml` is the runtime config for data location, retrieval settings, and ranking weights.

Run tests:

```bash
uv run pytest -q
```

Optional Streamlit app:

```bash
uv run streamlit run app/streamlit_app.py
```

Optional API:

```bash
uv run uvicorn app.api:app --reload
```

---

## 2. System overview

The system follows a retrieval, ranking, and explanation workflow:

1. **Intent parsing**: extract occasion, color, style, category, and budget hints from a natural-language request.
2. **Candidate retrieval**: retrieve catalog items using a text embedding baseline. The default implementation uses TF-IDF so the project works offline; a production upgrade could swap in CLIP/SentenceTransformers + FAISS/Qdrant.
3. **Personalized ranking**: re-rank candidates using user purchase history, inventory status, margin proxy, and return-risk proxy.
4. **Grounded explanation**: generate product explanations from retrieved catalog/review/inventory evidence rather than unsupported claims.
5. **Metric loop**: evaluate recommendation quality and business-facing proxies.

---

## 3. Repo layout

```text
fashion-style-ai-assistant/
├── app/                         # Optional Streamlit and FastAPI surfaces
├── config.yaml                  # Runtime config used by CLI/app/API
├── data/
│   ├── sample/                  # Tiny sample data that makes the repo runnable
│   └── external/                # Place real downloaded datasets here; gitignored
├── examples/                    # Example CLI output
├── scripts/                     # Demo, evaluation, and dataset ingestion scripts
├── src/fashion_style_ai_assistant/
│   ├── business.py              # Business KPI proxy helpers
│   ├── cli.py                   # CLI entry point
│   ├── config.py                # Config loading
│   ├── data.py                  # Dataset loaders and validation
│   ├── intent.py                # Query parsing
│   ├── metrics.py               # Ranking/business/RAG metrics
│   ├── pipeline.py              # End-to-end assistant orchestration
│   ├── rag.py                   # Grounded recommendation explanations
│   ├── ranking.py               # Personalization and business-aware re-ranking
│   ├── retrieval.py             # TF-IDF retrieval baseline
│   └── schemas.py               # Typed dataclasses
└── tests/                       # Unit tests
```

---

## 4. Recommended public datasets

Use the small sample dataset for a working MVP, then replace or enrich it with one or more of these:

| Dataset | Best use in this project | Link |
|---|---|---|
| H&M Personalized Fashion Recommendations | Main catalog + user transactions + recommendation holdout evaluation | https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data |
| Polyvore Outfits Dataset | Outfit compatibility, complete-the-look, multimodal retrieval | https://mariya.fyi/polyvore |
| UCSD Clothing Fit Data: RentTheRunway / ModCloth | Fit-risk prediction, review-grounded size/fit explanations | https://cseweb.ucsd.edu/~jmcauley/datasets.html |
| Amazon Reviews 2023 | Review RAG, product metadata, bought-together links, product search | https://amazon-reviews-2023.github.io/ |
| Fashion Product Images Small / Myntra | Fast CV/multimodal search demo with ~44K product images | https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small |
| BigQuery theLook eCommerce | Business KPI layer: orders, inventory, events, returns, funnel SQL | https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce |

See [`DATASETS.md`](DATASETS.md) for integration notes and caveats.

---

## 5. Limitations

This repository ships with a small sample dataset so it can run quickly and safely. For serious offline evaluation, replace the sample data with real public dataset outputs and report measured results from a time-based holdout split.

The default retriever is TF-IDF, not a deep vector model. That is deliberate for reproducibility. The retrieval interface can be replaced with CLIP, SentenceTransformers, FAISS, Qdrant, pgvector, or a hosted embedding endpoint.

---

Copyright © 2026 Jerry Huang. All rights reserved.
