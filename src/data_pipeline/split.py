"""Stage 4: Combine normalized + crisis-flagged data into per-split parquets.

Inputs are already pre-split at the HF level for ``go_emotions`` and
``empathetic_dialogues``. ``suicide_depression`` only has a ``train`` split
in HF, so we deterministically split it 80/10/10 here (stratified by the
``topic`` column when possible).

Outputs:
  - one parquet per split (``unified_{train,val,test}.parquet``)
  - each parquet contains rows from every dataset that has rows for that split
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PROCESSED_DIR, SCHEMA_COLUMNS, SEED, SPLIT_RATIOS, ensure_dirs, processed_path


# Map HF-style split names to our canonical names.
SPLIT_ALIAS = {
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
}

# Datasets whose HF 'train' needs to be deterministically re-split.
RESPLIT_DATASETS = frozenset({"suicide_depression"})


def _canonicalize_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["split"] = df["split"].map(SPLIT_ALIAS).fillna(df["split"])
    return df


def _resplit_single(
    df: pd.DataFrame,
    split_ratios: dict[str, float],
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Deterministically split a single DataFrame into train/val/test.

    Stratifies by the ``topic`` column if it has >=2 distinct non-null
    values, else falls back to a plain shuffle.
    """
    stratify = None
    if "topic" in df.columns:
        non_null = df["topic"].dropna()
        if non_null.nunique() >= 2:
            # sklearn requires the stratify array to have the same length as
            # df. We fill missing topics with a sentinel bucket.
            stratify = df["topic"].fillna("__missing__")

    test_size = split_ratios["test"]
    val_size = split_ratios["val"] / max(1.0 - test_size, 1e-9)

    # First: train vs (val + test).
    train_df, valtest_df = train_test_split(
        df,
        test_size=test_size + split_ratios["val"],
        random_state=seed,
        stratify=stratify,
    )
    # Second: val vs test (stratify from the original topic column).
    val_stratify = stratify.loc[valtest_df.index] if stratify is not None else None
    val_df, test_df = train_test_split(
        valtest_df,
        test_size=test_size / (test_size + split_ratios["val"]),
        random_state=seed,
        stratify=val_stratify,
    )

    return {"train": train_df, "val": val_df, "test": test_df}


def _resplit(datasets: dict[str, pd.DataFrame], split_ratios: dict[str, float], seed: int) -> None:
    """Re-split any dataset that has only a 'train' split, deterministically."""
    for slug in RESPLIT_DATASETS:
        df = datasets.get(slug)
        if df is None or df.empty:
            continue
        # Skip if the dataset already provides multiple splits.
        if "split" in df.columns and df["split"].nunique() > 1:
            continue

        print(f"[split] re-splitting {slug} deterministically (80/10/10, stratified by topic)")
        parts = _resplit_single(df, split_ratios, seed)
        # Tag each part with its new split name IN PLACE on the dict value
        # (re-binding a loop variable doesn't mutate the dict).
        out_parts = []
        for split_name, part in parts.items():
            part = part.copy()
            part["split"] = split_name
            out_parts.append(part)
        datasets[slug] = pd.concat(out_parts, ignore_index=True)


def run(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Combine all datasets into per-split DataFrames and write parquets."""
    ensure_dirs()

    # Re-split any dataset that only has a 'train' split.
    datasets = {slug: df.copy() for slug, df in datasets.items()}
    _resplit(datasets, SPLIT_RATIOS, SEED)

    combined = pd.concat(datasets.values(), ignore_index=True)
    combined = _canonicalize_split(combined)

    out: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        sub = combined.loc[combined["split"] == split].copy()
        # Drop rows with empty text.
        mask = sub["text"].isna() | (sub["text"].astype(str).str.strip() == "")
        if mask.any():
            print(f"[split] {split}: dropped {int(mask.sum())} empty-text rows")
            sub = sub.loc[~mask]

        sub = sub.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        for col in SCHEMA_COLUMNS:
            if col not in sub.columns:
                sub[col] = pd.NA
        sub = sub[SCHEMA_COLUMNS]
        sub["crisis_flag"] = sub["crisis_flag"].fillna(False).astype(bool)

        path = processed_path(split)
        sub.to_parquet(path, index=False, engine="pyarrow")
        print(f"[split] wrote {len(sub):>8} rows -> {path}")
        out[split] = sub

    return out
