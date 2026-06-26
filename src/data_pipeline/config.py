"""Configuration for the mindcare-ai data pipeline.

Single source of truth for paths, dataset names, splits, and seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root: <repo>/mindcare-ai
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

SPLIT_RATIOS = {"train": 0.8, "val": 0.10, "test": 0.10}

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# (huggingface_id, local_slug, hf_config_or_none)
DATASETS = {
    "go_emotions": {
        "hf_id": "google-research-datasets/go_emotions",
        "hf_config": "simplified",
        # HF split -> local file name. Preserved exactly.
        "split_file_map": {
            "train": "train.csv",
            "validation": "validation.csv",
            "test": "test.csv",
        },
        "license": "Apache-2.0",
    },
    "empathetic_dialogues": {
        # Parquet-converted version of the original Facebook EmpatheticDialogues release.
        # Provides pre-split (situation, emotion) pairs — used here for the
        # `emotion_label` and `context` columns of the unified schema.
        "hf_id": "bdotloh/empathetic-dialogues-contexts",
        "hf_config": None,
        "split_file_map": {
            "train": "train.csv",
            "validation": "validation.csv",
            "test": "test.csv",
        },
        "license": "Research use (see dataset card)",
    },
    "suicide_depression": {
        # Reddit-derived (r/SuicideWatch + r/depression) binary classification dataset.
        # Used as positive examples for the `crisis_flag` column.
        "hf_id": "jquiros/suicide",
        "hf_config": None,
        "split_file_map": "from_hf",
        "license": "Public Reddit posts; redistribution subject to Reddit ToS",
    },
}

# ---------------------------------------------------------------------------
# Unified schema columns (one row == one training example)
# ---------------------------------------------------------------------------

SCHEMA_COLUMNS = [
    "id",
    "dataset",
    "split",
    "text",
    "context",
    "emotion_label",
    "emotion_labels",
    "sentiment",
    "topic",
    "crisis_flag",
    "crisis_source",
]


def raw_dir(slug: str) -> Path:
    """Return the raw-data directory for a given dataset slug."""
    return RAW_DIR / slug


def processed_path(split: str) -> Path:
    """Return the processed parquet path for a given split."""
    return PROCESSED_DIR / f"unified_{split}.parquet"


def report_path() -> Path:
    """Return the data-quality report path."""
    return REPORTS_DIR / "data_quality.md"


@dataclass(frozen=True)
class PipelinePaths:
    """Bundle of all important paths for passing between stages."""

    project_root: Path
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path

    @classmethod
    def default(cls) -> "PipelinePaths":
        return cls(
            project_root=PROJECT_ROOT,
            raw_dir=RAW_DIR,
            processed_dir=PROCESSED_DIR,
            reports_dir=REPORTS_DIR,
        )


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)