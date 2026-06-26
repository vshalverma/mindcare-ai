"""Stage 2: Normalize raw CSVs into the unified schema.

One function per dataset. Each returns a ``pandas.DataFrame`` with the
columns defined in ``config.SCHEMA_COLUMNS``.

Unified schema (one row == one training example):

    id, dataset, split, text, context,
    emotion_label, emotion_labels (comma-separated str),
    sentiment, topic, crisis_flag, crisis_source
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import SCHEMA_COLUMNS, raw_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOE_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]
# Index 27 = neutral in the simplified GoEmotions config.
_GOE_LABEL_BY_INDEX = {i: name for i, name in enumerate(_GOE_LABELS)}

# Lightweight cleanup of the literal "comma_" word that appears in some ED
# parquet conversions (pixelsandpointers/empathetic_dialogues_for_lm), but
# harmless on plain text — kept here for forward-compatibility.
_COMMA_WORD_RE = re.compile(r"\bcomma_\b")


def _read_csv(slug: str, file_name: str) -> pd.DataFrame:
    """Read a raw CSV by dataset slug + file name. Returns a DataFrame."""
    path = raw_dir(slug) / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}. "
            f"Run 'python -m src.data_pipeline.run_pipeline --stage download' first."
        )
    return pd.read_csv(path, encoding="utf-8")


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every schema column exists (NaN-filled if absent)."""
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[SCHEMA_COLUMNS]


def _drop_empty_text(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where ``text`` is missing or blank."""
    mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
    dropped = int(mask.sum())
    if dropped:
        print(f"  [normalize] dropped {dropped} empty-text rows")
    return df.loc[~mask].copy()


# ---------------------------------------------------------------------------
# Per-dataset normalizers
# ---------------------------------------------------------------------------

def normalize_go_emotions() -> pd.DataFrame:
    """GoEmotions (simplified config, multi-label).

    Raw columns: ``text``, ``labels`` (list of int indices), ``id``.
    Mapping: index -> emotion string per ``_GOE_LABEL_BY_INDEX``.
    """
    print("[normalize] go_emotions")
    frames = []
    for hf_split, file_name in [
        ("train", "train.csv"),
        ("validation", "validation.csv"),
        ("test", "test.csv"),
    ]:
        df = _read_csv("go_emotions", file_name)
        # Some parquet->csv conversions collapse the list to a string like
        # "[27]" or "27"; handle both.
        def _parse_labels(value) -> list[int]:
            if isinstance(value, list):
                return [int(v) for v in value]
            s = str(value).strip("[] ")
            if not s:
                return []
            return [int(x) for x in s.split(",") if x.strip().isdigit()]

        df["emotion_label_list"] = df["labels"].apply(_parse_labels)
        df["emotion_labels"] = df["emotion_label_list"].apply(
            lambda idxs: ",".join(
                _GOE_LABEL_BY_INDEX[i] for i in idxs if i in _GOE_LABEL_BY_INDEX
            )
        )
        # Primary label = first listed (GoEmotions rows are usually 1-2 labels).
        df["emotion_label"] = df["emotion_label_list"].apply(
            lambda idxs: _GOE_LABEL_BY_INDEX.get(idxs[0]) if idxs else None
        )

        out = pd.DataFrame({
            "id": "goemotions_" + df["id"].astype(str),
            "dataset": "go_emotions",
            "split": hf_split,
            "text": df["text"],
            "context": pd.NA,
            "emotion_label": df["emotion_label"],
            "emotion_labels": df["emotion_labels"],
            "sentiment": pd.NA,
            "topic": pd.NA,
            "crisis_flag": False,        # filled in by crisis.py
            "crisis_source": pd.NA,
        })
        frames.append(out)

    df = pd.concat(frames, ignore_index=True)
    df = _drop_empty_text(df)
    return _ensure_schema(df)


def normalize_empathetic_dialogues() -> pd.DataFrame:
    """EmpatheticDialogues — pre-split ``(situation, emotion)`` rows.

    Source: bdotloh/empathetic-dialogues-contexts parquet mirror.
    Raw columns: ``situation``, ``emotion`` (32 emotion categories).
    """
    print("[normalize] empathetic_dialogues")
    frames = []
    for hf_split, file_name in [
        ("train", "train.csv"),
        ("validation", "validation.csv"),
        ("test", "test.csv"),
    ]:
        df = _read_csv("empathetic_dialogues", file_name)
        # Normalize "comma_" token (forward-compat with one ED mirror).
        situations = df["situation"].astype(str).str.replace(_COMMA_WORD_RE, ",", regex=True)

        out = pd.DataFrame({
            "id": "ed_" + df.index.astype(str) + "_" + hf_split,
            "dataset": "empathetic_dialogues",
            "split": hf_split,
            "text": situations,
            "context": pd.NA,
            "emotion_label": df["emotion"].astype(str).str.lower(),
            "emotion_labels": df["emotion"].astype(str).str.lower(),
            "sentiment": pd.NA,
            "topic": pd.NA,
            "crisis_flag": False,
            "crisis_source": pd.NA,
        })
        frames.append(out)

    df = pd.concat(frames, ignore_index=True)
    df = _drop_empty_text(df)
    return _ensure_schema(df)


def normalize_suicide_depression() -> pd.DataFrame:
    """Reddit r/SuicideWatch + r/depression (jquiros/suicide).

    Raw columns: ``text``, ``class`` in {``suicide``, ``non-suicide``}.
    We tag every row with ``topic`` (suicide/depression/non-suicide) and
    pre-set ``crisis_flag=True`` for ``class == 'suicide'`` (the
    keyword heuristic pass in ``crisis.py`` may add more).
    """
    print("[normalize] suicide_depression")
    df = _read_csv("suicide_depression", "train.csv")

    # HF split 'train' was the only one returned; keep its name.
    hf_split = "train"

    # Map class -> topic + pre-flag.
    def _topic(value: str) -> str:
        v = str(value).strip().lower()
        return v if v in {"suicide", "non-suicide", "depression"} else v or "unknown"

    out = pd.DataFrame({
        "id": "sd_" + df.index.astype(str),
        "dataset": "suicide_depression",
        "split": hf_split,
        "text": df["text"],
        "context": pd.NA,
        "emotion_label": pd.NA,
        "emotion_labels": pd.NA,
        "sentiment": pd.NA,
        "topic": df["class"].apply(_topic),
        "crisis_flag": df["class"].astype(str).str.lower().eq("suicide"),
        "crisis_source": df["class"].astype(str).str.lower().eq("suicide").apply(
            lambda x: "dataset_label" if x else pd.NA
        ),
    })

    out = _drop_empty_text(out)
    return _ensure_schema(out)


# ---------------------------------------------------------------------------
# Stage entrypoint
# ---------------------------------------------------------------------------

NORMALIZERS = {
    "go_emotions": normalize_go_emotions,
    "empathetic_dialogues": normalize_empathetic_dialogues,
    "suicide_depression": normalize_suicide_depression,
}


def run() -> dict[str, pd.DataFrame]:
    """Normalize every dataset and return a {slug: DataFrame} mapping."""
    return {slug: fn() for slug, fn in NORMALIZERS.items()}
