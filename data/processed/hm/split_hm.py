"""Select a user cohort from full/ and split it into train/ and test/.

    <data-dir>/full/    complete normalized dataset (input, never modified)
    <data-dir>/train/   cohort users, every date except their last
    <data-dir>/test/    cohort users, exactly one purchase from their last date

The cohort is stratified on age_band x budget_tier, with each cell's quota proportional to its
share of the population, so the cohort matches the real attribute mix. Within a cell the users
with the most distinct purchase dates are taken, so every user has history to personalise from.
Selection is deterministic -- ties break by purchase count, then user_id.

Users are selected first, then split, so both halves describe exactly the same cohort.

Test holds ONE purchase per user, so every user weighs the same in the score. The rest of that
user's last date is discarded rather than moved to train -- otherwise a same-day basket sibling
of the answer would sit in the history.

Usage:
    python scripts/data_ingest/split_hm.py --data-dir data/processed/hm --n-users 10000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

STRATA = ["age_band", "budget_tier"]
# train/ is a self-contained dataset for the cohort; test/ carries only the grading tables.
TRAIN_TABLES = ["products.csv", "inventory.csv", "reviews.csv", "outfits.csv"]
TEST_TABLES = ["products.csv", "inventory.csv"]


def column_index(header: str, name: str) -> int:
    """Position of a column by name -- layouts differ between datasets, so never hardcode."""
    columns = [c.strip() for c in header.rstrip("\r\n").split(",")]
    if name not in columns:
        raise SystemExit(f"interactions.csv has no {name!r} column (found {columns})")
    return columns.index(name)


def scan_users(source: Path) -> tuple[pd.DataFrame, dict[str, str], dict[str, int], int]:
    """One pass: per user, purchase count, distinct purchase dates, and last purchase date."""
    purchases: dict[str, int] = {}
    n_dates: dict[str, int] = {}
    last_date: dict[str, str] = {}
    last_rows: dict[str, int] = {}
    total = 0
    out_of_order = 0
    previous = ""
    with source.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline()
        u_at, t_at = column_index(header, "user_id"), column_index(header, "event_timestamp")
        width = max(u_at, t_at) + 1
        for line in handle:
            if not line.strip():
                continue
            total += 1
            parts = line.split(",", width)
            uid, ts = parts[u_at], parts[t_at]
            if ts < previous:
                out_of_order += 1
            previous = ts
            purchases[uid] = purchases.get(uid, 0) + 1
            if last_date.get(uid) != ts:
                n_dates[uid] = n_dates.get(uid, 0) + 1
                last_date[uid] = ts
                last_rows[uid] = 1
            else:
                last_rows[uid] += 1
            if total % 10_000_000 == 0:
                print(f"  scan: {total:,} rows, {len(purchases):,} users", flush=True)

    # Counting distinct dates by "changed since the last row" is only valid on a date-sorted file.
    if out_of_order:
        raise SystemExit(f"{source} is not sorted by event_timestamp ({out_of_order:,} rows out of order)")

    frame = pd.DataFrame(
        {
            "user_id": list(purchases),
            "purchases": list(purchases.values()),
            "n_dates": [n_dates[u] for u in purchases],
        }
    )
    return frame, last_date, last_rows, total


def _quotas(sizes: pd.Series, n: int) -> dict[str, int]:
    """Largest-remainder allocation of n across cells, proportional to population share."""
    raw = sizes / sizes.sum() * n
    out = np.floor(raw).astype(int)
    for key in (raw - out).sort_values(ascending=False).index:
        if out.sum() >= n:
            break
        out[key] += 1
    return {k: int(v) for k, v in out.items() if v > 0}


def select_cohort(frame: pd.DataFrame, users_csv: Path, n_users: int) -> pd.DataFrame:
    """Proportional quota per age_band x budget_tier cell; the most-dates users within each."""
    attrs = pd.read_csv(users_csv, usecols=["user_id", *STRATA], dtype=str)
    frame = frame.merge(attrs, on="user_id", how="left")
    for col in STRATA:
        frame[col] = frame[col].fillna("(nan)")
    frame["cell"] = frame[STRATA].agg(" | ".join, axis=1)

    ordered = frame.sort_values(
        ["n_dates", "purchases", "user_id"], ascending=[False, False, True], kind="mergesort"
    )
    quotas = _quotas(frame["cell"].value_counts(), n_users)
    picked = [g.head(quotas[cell]) for cell, g in ordered.groupby("cell", sort=False) if cell in quotas]
    return pd.concat(picked).reset_index(drop=True)


def split(data_dir: Path, n_users: int) -> dict:
    full = data_dir / "full"
    source = full / "interactions.csv"
    if not source.exists():
        raise SystemExit(f"Missing {source}")
    train_dir, test_dir = data_dir / "train", data_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    frame, last_date, last_rows, total = scan_users(source)
    print(f"  scan done: {total:,} rows, {len(frame):,} users", flush=True)

    cohort_frame = select_cohort(frame, full / "users.csv", n_users)
    cohort = set(cohort_frame["user_id"])
    print(f"  cohort {len(cohort):,} users, min dates {cohort_frame.n_dates.min()}", flush=True)

    train_rows = test_rows = discarded = 0
    train_users: set[str] = set()
    # Which of the user's last-date rows becomes the case. crc32 is stable across runs and
    # independent of row order within the date, so no positional bias.
    target = {u: zlib.crc32(u.encode()) % last_rows[u] for u in cohort}
    at: dict[str, int] = {}
    # Build under temp names and swap in at the end: an incrementally-written file left open for
    # minutes at the final path came out short and reproducibly so.
    train_tmp = train_dir / ".interactions.csv.tmp"
    test_tmp = test_dir / ".interactions.csv.tmp"
    with (
        source.open("r", encoding="utf-8", newline="") as handle,
        train_tmp.open("w", encoding="utf-8", newline="") as train_out,
        test_tmp.open("w", encoding="utf-8", newline="") as test_out,
    ):
        header = handle.readline()
        u_at, t_at = column_index(header, "user_id"), column_index(header, "event_timestamp")
        width = max(u_at, t_at) + 1
        train_out.write(header)
        test_out.write(header)
        seen = 0
        for line in handle:
            if not line.strip():
                continue
            seen += 1
            parts = line.split(",", width)
            uid, ts = parts[u_at], parts[t_at]
            if uid not in cohort:
                continue
            if ts == last_date[uid]:
                i = at.get(uid, 0)
                at[uid] = i + 1
                if i == target[uid]:
                    test_out.write(line)
                    test_rows += 1
                else:
                    discarded += 1
            else:
                train_out.write(line)
                train_rows += 1
                train_users.add(uid)
            if seen % 10_000_000 == 0:
                print(f"  route: {seen:,} rows scanned", flush=True)
        train_out.flush()
        test_out.flush()
        os.fsync(train_out.fileno())
        os.fsync(test_out.fileno())

    # A short file here means writes were issued but did not land. Fail rather than leave a
    # silently incomplete answer key behind.
    for tmp, final, expected in (
        (train_tmp, train_dir / "interactions.csv", train_rows),
        (test_tmp, test_dir / "interactions.csv", test_rows),
    ):
        with tmp.open("r", encoding="utf-8", newline="") as fh:
            fh.readline()
            written = sum(1 for line in fh if line.strip())
        if written != expected:
            raise SystemExit(f"{tmp}: wrote {expected:,} rows but file holds {written:,}")
        tmp.replace(final)

    for name in TRAIN_TABLES:
        if (full / name).exists():
            shutil.copy2(full / name, train_dir / name)
    for name in TEST_TABLES:
        if (full / name).exists():
            shutil.copy2(full / name, test_dir / name)
    # users.csv is subset to the cohort; the catalog stays whole.
    users = pd.read_csv(full / "users.csv", dtype=str)
    users[users["user_id"].isin(cohort)].to_csv(train_dir / "users.csv", index=False)

    manifest = {
        "rule": "stratified on age_band x budget_tier; most-dates users within each cell; "
                "one purchase from each user's last date is the test case, the rest of that "
                "date is discarded",
        "source": str(source),
        "source_rows": total,
        "source_users": len(frame),
        "n_users_requested": n_users,
        "n_users_selected": len(cohort),
        "strata": STRATA,
        "cohort_dates_min": int(cohort_frame["n_dates"].min()),
        "cohort_dates_median": float(cohort_frame["n_dates"].median()),
        "cohort_dates_max": int(cohort_frame["n_dates"].max()),
        "cohort_purchases_median": float(cohort_frame["purchases"].median()),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "test_cases": test_rows,
        "last_date_rows_discarded": discarded,
        "users_no_train_history": len(cohort) - len(train_users),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (data_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--n-users", type=int, default=10_000)
    args = parser.parse_args()
    print(json.dumps(split(args.data_dir, args.n_users), indent=2))


if __name__ == "__main__":
    main()
