# Project structure

The agreed layout. Three layers, one direction of pull:

- **`src/`** — the assistants (serving layer). Each assistant lives in its own folder and is
  the same kind of object: a `FashionAssistant` class in its `pipeline.py` (same interface,
  different internals per assistant) plus a zero-arg `build()` in its `__init__.py` returning
  it ready to answer `recommend_dataframe(query, user_id, top_k)` with `name`/`version`.
- **`models/`** — the models and everything related to training them: model code, `train.py`,
  trained weights. Weights are gitignored (`*.pkl`); training runs where the RAM is (the
  AutoDL instance).
- **`data/`** — the data pool. `data/processed/hm/train` is what assistants and training may
  use; `data/processed/hm/test` is the answer key, read only by the eval.

An assistant can grab any model from `models/` and any data from `data/`. The eval
(`scripts/eval/`) is fixed and touches none of this — it imports one `build()` and scores.

```
fashion-style-ai-assistant/
│
├── config.yaml
├── pyproject.toml
│
├── src/fashion_style_ai_assistant/
│   ├── config.py  data.py  schemas.py       # shared library (eval + assistants + training import these)
│   ├── intent.py                            # shared: one query language → ParsedIntent; each assistant
│   │                                        #   decides how to USE the parsed intent
│   ├── rag.py                               # shared response layer: every assistant returns ranked rows +
│   │                                        #   evidence in the schemas.py format; rag turns it into the
│   │                                        #   human-readable grounded answer (one renderer for all)
│   ├── business.py                          # shared: stock/margin/return decoration computed from the
│   │                                        #   ranked rows — assistant-agnostic, like rag
│   ├── cli.py  __main__.py                  # shared terminal front door: --assistant <name> picks who
│   │                                        #   answers (imports that assistant's build()); rag renders
│   │                                        #   e.g. fashion-assistant demo --assistant behavioural --query "..."
│   │
│   ├── baseline/                            # assistant 1 — TF-IDF engine
│   │   ├── __init__.py                      #   build() → user_profiles.csv + data pool → FashionAssistant
│   │   │                                    #     (explicit data_dir → fit live; demo/CLI path)
│   │   ├── pipeline.py                      #   class FashionAssistant — intent → retrieval → rerank → respond
│   │   ├── retrieval.py                     #   TF-IDF index + search over the catalog
│   │   └── ranking.py                       #   weighted-sum rerank (intent match, preference, price, stock)
│   │
│   └── behavioural/                         # assistant 2 — grabs its trained model from models/
│       ├── __init__.py                      #   build() → load models/behavioural/weights.pkl → FashionAssistant
│       ├── pipeline.py                      #   class FashionAssistant — intent → retrieval → rerank → respond
│       ├── retrieval.py                     #   candidate generation: reads the model's learned stores
│       │                                    #     (popularity, repurchase, item-CF, ALS neighbors),
│       │                                    #     filtered/boosted by the parsed intent (query-aware)
│       └── ranking.py                       #   featurize (user, candidate) + call trained LGBM+CatBoost
│
│   # assistants own only how they CHOOSE and RANK items (retrieval, ranking, pipeline).
│   # everything before and after is shared: intent in, rag + business out, schemas in between.
│
├── models/                                  # models + everything related to training them
│   ├── baseline/
│   │   ├── train.py                         #   precompute all users' profiles once (vectorized
│   │   │                                    #     build_user_profile, verified identical)
│   │   └── user_profiles.csv                #   the whole artifact: user_id, kind, key, value
│   │                                        #     (plain CSV, inspectable; gitignored; built on instance)
│   │                                        #   TF-IDF is NOT saved — build() refits it from
│   │                                        #     products.csv in seconds; users/inventory/reviews
│   │                                        #     come from the data pool
│   └── behavioural/
│       ├── train.py                         #   builds the learned stores + fits the GBDTs on train/
│       │                                    #     (imports featurize from src ranking.py — train and
│       │                                    #      serve share ONE featurize, so they cannot drift)
│       └── weights.pkl                      #   THE trained model: learned stores + fitted LGBM+CatBoost
│                                            #     (plain dict of arrays/boosters; gitignored; trained on instance)
│
├── data/
│   ├── sample/                              # tiny fixture for tests/demo
│   └── processed/hm/
│       ├── train/                           # usable by assistants and training
│       └── test/                            # answer key — eval only
│
├── scripts/
│   ├── eval/                                # the fixed eval — untouchable
│   ├── data_ingest/                         # ingest_*.py, split_hm.py
│   └── run_demo.py
│
├── app/                                     # api.py, streamlit_app.py (demo apps over baseline)
└── tests/
```

## The I/O contract (schemas.py)

Every assistant reads and writes the exact same dataclasses — this is what lets the shared
components stay assistant-blind:

```
                 ┌─────────────────────────────────────────────┐
  query ─────────►  shared intent.py  → ParsedIntent           │
  CatalogBundle ─►                                             │
                 │  assistant pipeline (retrieval → rerank)    │
                 │       ↓                                     │
                 │  list[RecommendationResult]                 │  ← same output for ALL assistants
                 │    each: product_id, rank, score, product,  │
                 │          why, evidence[RecommendationEvidence]
                 └───────┬─────────────────────┬───────────────┘
                         ▼                     ▼
                 shared rag.py          shared business.py
                 (grounded text)        (stock/margin/return)
```

- `recommend_dataframe(query, user_id, top_k)` — the eval's view: ranked ids from that list.
- `recommend(query, user_id, top_k)` — the human's view: rag + business rendered over the same list.
- Assistants differ only in HOW they fill the list; never in its shape. A new assistant that
  emits `list[RecommendationResult]` gets the CLI, the apps, rag, business, and the eval for free.

### The full assistant contract

| required by | what | detail |
|---|---|---|
| eval | `build()` | zero-arg factory in the folder's `__init__.py`, returns a ready object |
| eval | `name`, `version` | `str` attributes; leaderboard label — bump version on any change that moves numbers |
| eval | `recommend_dataframe(query, user_id, top_k)` | keyword-called; ranked ids; extra params need defaults |
| system | `recommend(query, user_id, top_k)` | full response dict via shared rag + business; consumed by CLI and apps |
| system | pipeline → `list[RecommendationResult]` | the internal waist; both public methods are views over it |
| convention | `class FashionAssistant` in `pipeline.py` | same class name everywhere; module path disambiguates |
| convention | inputs | settings from `config.yaml`, data from `data/`, trained models from `models/<name>/` |

Two public methods, two label attributes, one internal dataclass boundary — nothing else.

## Submitting

Both assistants follow the same two steps: build the artifact once (on the instance, full
`train/`), then score any number of times.

```bash
# baseline
uv run python models/baseline/train.py
uv run python scripts/eval/evaluate.py --config config.yaml \
    --data-dir data/processed/hm --assistant fashion_style_ai_assistant.baseline:build

# behavioural
uv run python models/behavioural/train.py
uv run python scripts/eval/evaluate.py --config config.yaml \
    --data-dir data/processed/hm --assistant fashion_style_ai_assistant.behavioural:build
```
