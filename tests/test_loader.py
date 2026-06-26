"""Tests for ``src.inference.loader``.

The loader is the single source of truth for turning a checkpoint
directory into a live model + tokenizer + label_map bundle. These
tests target the parts that are testable without a real torch /
transformers / safetensors install:

  - ``_resolve_encoder_name`` (pure function over a dict)
  - ``LoadedClassifier`` shape and properties
  - error paths in ``load_classifier`` (missing label_map, missing weights)

End-to-end loading against the real checkpoint is covered by
``smoke_test_chat.py`` and the manual ``eval_per_class.py`` run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.inference.loader import LoadedClassifier, _resolve_encoder_name, load_classifier


# ---------------------------------------------------------------------------
# _resolve_encoder_name — pure function, no fixtures
# ---------------------------------------------------------------------------

def test_resolve_encoder_name_prefers_base_encoder_name() -> None:
    label_map = {
        "base_encoder_name": "roberta-base",
        "encoder_name": "bert-base-uncased",  # legacy key — must be ignored
    }
    assert _resolve_encoder_name(label_map) == "roberta-base"


def test_resolve_encoder_name_falls_back_to_legacy_key() -> None:
    assert _resolve_encoder_name({"encoder_name": "bert-base-uncased"}) == "bert-base-uncased"


def test_resolve_encoder_name_defaults_when_missing() -> None:
    assert _resolve_encoder_name({}) == "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# LoadedClassifier — typed bundle; shape & convenience properties
# ---------------------------------------------------------------------------

def _fake_bundle(emotions: list[str]) -> LoadedClassifier:
    """Construct a LoadedClassifier with stand-in objects for model/tokenizer.

    The real loader builds real torch modules; here we just want to check
    that the dataclass exposes what callers actually use.
    """
    return LoadedClassifier(
        model=object(),
        tokenizer=object(),
        label_map={"emotions": emotions},
        model_dir=Path("models/checkpoints/final"),
        device="cpu",
    )


def test_loaded_classifier_emotions_property() -> None:
    bundle = _fake_bundle(["neutral", "joy", "sadness"])
    assert bundle.emotions == ["neutral", "joy", "sadness"]


def test_loaded_classifier_num_emotions_property() -> None:
    bundle = _fake_bundle(["neutral"] * 28)
    assert bundle.num_emotions == 28


def test_loaded_classifier_round_trip() -> None:
    # Sanity: the field set on the dataclass is what callers actually read.
    bundle = _fake_bundle(["a", "b"])
    assert isinstance(bundle.model, object)
    assert isinstance(bundle.tokenizer, object)
    assert bundle.device == "cpu"
    assert bundle.model_dir == Path("models/checkpoints/final")


# ---------------------------------------------------------------------------
# load_classifier — error paths
# ---------------------------------------------------------------------------
# These don't need torch; the function should raise BEFORE importing torch
# when the label_map.json is missing. That keeps early failure cheap and
# gives a clear diagnostic in environments without the heavy deps.

def test_load_classifier_raises_when_label_map_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="label_map.json"):
        load_classifier(tmp_path)


def test_load_classifier_raises_when_weights_missing(tmp_path: Path) -> None:
    # label_map present, but no safetensors and no pytorch_model.bin.
    (tmp_path / "label_map.json").write_text('{"emotions": ["neutral"]}', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="No weights found"):
        load_classifier(tmp_path)


def test_load_classifier_accepts_string_path(tmp_path: Path) -> None:
    # Path-coercion works for both Path and str. We only check the
    # label_map-missing branch (no model load attempted).
    with pytest.raises(FileNotFoundError, match="label_map.json"):
        load_classifier(str(tmp_path))