# Evaluation spec

The contract for anyone submitting an assistant to the exam and the leaderboard.

Everything in this document is fixed. The evaluation is identical for every submission; if it is
not, the resulting numbers are not comparable and will not be recorded.

---

## 1. The task

Given a **user** and a **query**, return your top-K ranked `product_id`s.

One held-out purchase is the correct answer. You are scored on whether it appears in your list.

---

## 2. The interface you must implement

### The factory

You supply a **zero-argument factory** that returns an object ready to answer questions:

```python
# mypkg/mymodel.py
def build():
    model = MyRecommender(...)
    model.load("artifacts/mymodel.pkl")   # or read train/, or fit, or call an API
    return model
```

Pass it as `--assistant mypkg.mymodel:build`. It is imported and called once, with no arguments.
Everything it does — reading `train/`, fitting, loading a checkpoint — happens on your side and
is never visible to the evaluation.

### The object it returns

Must answer exactly this call:

```python
recommend_dataframe(query: str, user_id: str, top_k: int)
```

One method, no alternatives. `recommend_dataframe` is the ranking without the explanation and
business decoration the eval discards, so it is what gets called. An object without it is
rejected before scoring starts.

All three are passed as **keyword** arguments. Any parameter beyond these three must have a
default, or the call fails and your submission cannot be scored.

Your object must also expose two attributes, used only to label your leaderboard row:

| attribute | type | example |
|---|---|---|
| `name` | `str` | `"tfidf_weighted"` |
| `version` | `str` | `"1.0"` |

### Return value

Best-first ranked `product_id`s, in any of these forms:

| form | how it is read |
|---|---|
| `pandas.DataFrame` | the `product_id` column, in row order |
| `dict` | `[r["product_id"] for r in value["recommendations"]]` |
| anything iterable | taken as the id sequence directly |

Ids are cast to `str` and truncated to the first `top_k`. Returning fewer than `top_k` is legal
and simply reduces your chance of a hit. Duplicate ids waste slots; they are not de-duplicated.

---

## 3. Data

Source: `data/processed/hm/`, split by `scripts/data_ingest/split_hm.py`.

**Cohort** — 10,000 users, drawn from the 1,362,281 in the log. Quotas are proportional to each
`age_band × budget_tier` cell's share of the population, and within a cell the users with the
most distinct purchase dates are taken. So the cohort matches the real customer mix on both
attributes to within 0.05pp, while every user has substantial history. Selection is
deterministic — no seed.

**Split rule** — one purchase from each user's **last date** is their test case. Everything
before that date is `train/`. The user's other purchases on that same date are **discarded** —
not moved to train — so no item bought alongside the answer sits in the history.

| | rows |
|---|---:|
| `full/` (source) | 31,788,324 |
| `train/` | 2,208,989 |
| `test/` | **10,000** — one case per user |
| discarded (same-date siblings) | 19,212 |

### `train/` — everything you may use

`products.csv`, `interactions.csv`, `inventory.csv`, `users.csv`, `reviews.csv`, `outfits.csv`.

### `test/` — the answer key

`interactions.csv` (the cases), plus `products.csv` and `inventory.csv` used for grading.

> **Reading `test/` for any purpose other than being scored is disqualifying.** It contains the
> answers. This includes fitting on it, tuning against it, or using it to select a checkpoint.

### Properties you should design for

- **Every user has history.** Distinct purchase dates per user: minimum 8, median 62, maximum
  427; median 191 purchases. There are no cold-start users in this cohort.
- **One case per user**, so every customer weighs the same in the score. A heavy basket cannot
  outvote a light one.
- `event_timestamp` is **date-only** — no time of day, so a date is a shopping trip, not a
  moment. Test dates span 2019-04-23 to 2020-09-22 and are per-user, not a single global week.
- `train/users.csv` is the cohort (10,000 rows); `products.csv` and `inventory.csv` are the
  **whole** catalog (99,772 items), so the candidate space is not reduced.
- `inventory.csv` is synthetic (`stock`, `margin`, `return_rate` are generated). Business
  metrics computed from it are not statements about a real business.

---

## 4. Protocol

Fixed. Not configurable per submission.

1. Load cases from `test/interactions.csv` — 10,000, one per user.
2. Score all of them. (`sample_users` is `null`; if set, cases beyond it are subsampled with
   `cases.sample(n=sample_users, random_state=sample_seed)`.)
3. For each case:
   - `user_id` — from the case row.
   - `relevant` — `{case.product_id}`, exactly one correct answer.
   - `query` — built from the held-out product's own catalog row:
     ```python
     f"{style_tags} {category} for {occasion_tags}"
     ```
   - Call your object with `query`, `user_id`, `top_k`.
   - Truncate the answer to `top_k` ids.
4. Cases whose `product_id` is absent from `products.csv` are skipped.
5. Report the mean of each metric across all scored cases.

### Configuration

| key | value | meaning |
|---|---|---|
| `top_k` | `5` | length of your list; the K in every metric |
| `sample_users` | `null` | score every case; set an integer to subsample |
| `sample_seed` | `42` | only used when `sample_users` is set |

Set in `config.yaml`. Changing any of them produces a different board (§6).

---

## 5. Metrics

One correct answer per case and one case per user, so `recall@5` reads directly as **the share
of the 10,000 users whose next purchase appeared in your top 5**. It follows that
`precision@5 == recall@5 / 5` and `users_evaluated == 10,000`.

| metric | definition |
|---|---|
| `recall_at_k` | **primary.** `\|top_k ∩ relevant\| / \|relevant\|` |
| `precision_at_k` | `\|top_k ∩ relevant\| / k` |
| `map_at_k` | `Σ(hits_so_far / rank) / min(\|relevant\|, k)` |
| `ndcg_at_k` | `DCG@k / ideal DCG@k`, binary gain |
| `inventory_hit_rate` | share of your top-k with `stock > 0` |
| `category_diversity` | distinct `category` values among your top-k |
| `users_evaluated` | number of cases scored — 10,000, one per user |

Ranking is by `recall_at_k`, descending.

`inventory_hit_rate` and `category_diversity` are reported, not ranked on. When `inventory.csv`
is absent, stock defaults to 1 and `inventory_hit_rate` is 1.0 by construction — meaningless,
not perfect.

### Reading a score honestly

Scoring all 10,000 cases, the 95% confidence interval on a recall of *p* is roughly
`±1.96·√(p(1−p)/10000)` — about **±0.7 points** at `p = 0.15`. Two submissions closer than that
are not meaningfully separated, however many decimals the board prints.

---

## 6. Leaderboard

One board holds runs for exactly one dataset and configuration, identified by:

```
<data-dir-name>|top_k=<k>|cases=all            # or cases=<n>|seed=<s> if subsampled
```

Submitting a run whose key differs from the board's raises `DatasetMismatch`; the evaluation
still succeeds and writes its report, but the row is refused. This is deliberate — a run scored
on 100 easy cases is not comparable to one scored on 1000, and would otherwise sit at the top
forever.

To keep a separate board, pass `--leaderboard <other-path>`.

Each row records: `id`, `name`, `version`, `run_at`, `dataset`, `users_evaluated`, and the six
metrics. Entries are append-only. `leaderboard.html` is regenerated alongside the JSON.

> The key encodes the **directory name**, not the contents of the data. If the underlying data is
> regenerated or re-split, start a new board — the key will not catch it for you.

---

## 7. Submitting

Every submission is scored by one command:

```bash
uv run python scripts/eval/evaluate.py --config config.yaml \
    --data-dir data/processed/hm --assistant mypkg.mymodel:build
```

`--assistant` is your factory (§2). `--data-dir` must contain a `test/` directory; build one
with `scripts/data_ingest/split_hm.py` if it does not.

`top_k`, `sample_users` and `sample_seed` are read from `config.yaml` and cannot be overridden
from the command line — that is what keeps every submission on one board. Editing those values
in `config.yaml` starts a different board (§6).

Submitting requires **no edits inside `scripts/eval/`**. If you find yourself needing one, the
spec is wrong — say so rather than working around it.

This writes:

- `artifacts/eval_report_<name>_<dataset>.json` — this run alone; overwritten on re-run
- `artifacts/leaderboard.json` / `.html` — appended

Add `--leaderboard none` to score without recording. Use it while you are iterating; the board
is append-only and every recorded run stays there.

### Rules

1. Bump `version` on any change that can move your numbers. A stale row otherwise reports the
   score of code that no longer exists.
2. Fix your seeds. Two runs of the same `name`/`version` must produce the same score.
3. Never read `test/` except to be scored.
4. Do not modify anything under `scripts/eval/`, and do not edit the `evaluation` block in
   `config.yaml`. A submission that changes how it is graded is not a submission.

---

## 8. Scope

The eval reads `test/`, calls your factory, asks each question, and returns a score. It does not
open `train/`, does not build or fit anything, and does not care how your assistant produces its
answer.

Getting your object ready to answer is entirely your side of the line.
