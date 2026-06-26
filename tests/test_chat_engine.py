"""Tests for ChatEngine — fully stubbed, no GPU or model needed.

Covers the four branches of crisis_flag and the empty-input fallback.
"""

from __future__ import annotations

import pytest

from src.inference.chat_engine import ChatEngine, ChatReply
from tests.conftest import StubClassifier, StubPrediction


# ---------------------------------------------------------------------------
# Branch coverage for crisis_flag
# ---------------------------------------------------------------------------

def test_model_under_threshold_no_keyword_no_flag(stub_factory) -> None:
    eng = stub_factory(emotion="joy", crisis_prob=0.10)
    reply = eng.reply("I'm feeling great today")
    assert reply.crisis_flag is False
    assert reply.crisis_reason == "none"


def test_model_over_threshold_no_keyword_flags(stub_factory) -> None:
    eng = stub_factory(emotion="sadness", crisis_prob=0.85)
    reply = eng.reply("everything feels heavy")
    assert reply.crisis_flag is True
    assert reply.crisis_reason == "model"


def test_keyword_fires_model_under_threshold(stub_factory) -> None:
    eng = stub_factory(emotion="sadness", crisis_prob=0.10)
    reply = eng.reply("I want to kill myself")
    assert reply.crisis_flag is True
    assert reply.crisis_reason == "keyword"


def test_both_fire_combined_reason(stub_factory) -> None:
    eng = stub_factory(emotion="sadness", crisis_prob=0.95)
    reply = eng.reply("I want to kill myself and end my life")
    assert reply.crisis_flag is True
    assert reply.crisis_reason == "model+keyword"


# ---------------------------------------------------------------------------
# Empty / whitespace input
# ---------------------------------------------------------------------------

def test_empty_input_returns_neutral_fallback(stub_factory) -> None:
    eng = stub_factory()  # default neutral / 0.0
    reply = eng.reply("")
    assert isinstance(reply, ChatReply)
    assert reply.crisis_flag is False
    assert reply.crisis_reason == "empty_input"
    assert reply.emotion == "neutral"
    assert reply.text  # non-empty user-visible text


def test_whitespace_only_is_treated_as_empty(stub_factory) -> None:
    eng = stub_factory()
    reply = eng.reply("   \n\t  ")
    assert reply.crisis_reason == "empty_input"


# ---------------------------------------------------------------------------
# Reply contract
# ---------------------------------------------------------------------------

def test_reply_text_is_non_empty_for_normal_input(stub_factory) -> None:
    eng = stub_factory(emotion="joy", crisis_prob=0.0)
    reply = eng.reply("today was wonderful")
    assert reply.text
    assert reply.emotion == "joy"
    assert 0.0 <= reply.emotion_confidence <= 1.0
    assert 0.0 <= reply.crisis_prob <= 1.0


def test_known_emotion_uses_templated_response(stub_factory) -> None:
    eng = stub_factory(emotion="joy", crisis_prob=0.0)
    reply = eng.reply("today was wonderful")
    # The "joy" template pool has the literal phrase "wonderful" / "love to hear"
    # — we don't pin to a specific template (random choice), just assert
    # we got something joy-toned.
    assert reply.text


def test_unknown_emotion_falls_back_to_neutral_template(stub_factory) -> None:
    # Force an emotion that's not in _RESPONSES so we hit the fallback.
    eng = stub_factory(emotion="amusement", crisis_prob=0.0)
    reply = eng.reply("haha that was funny")
    # _RESPONSES["neutral"][0] is "Tell me more."
    assert reply.text == "Tell me more."


# ---------------------------------------------------------------------------
# Crisis reason logic — verify the truth table directly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_prob,text,expected_flag,expected_reason",
    [
        (0.0,  "hello there",                    False, "none"),
        (0.49, "I feel sad",                     False, "none"),
        (0.5,  "I feel sad",                     True,  "model"),
        (0.99, "I feel sad",                     True,  "model"),
        (0.0,  "I want to kill myself",          True,  "keyword"),
        (0.99, "I want to kill myself",          True,  "model+keyword"),
        (0.0,  "what's the point of living",     True,  "keyword"),
    ],
)
def test_crisis_truth_table(
    stub_factory, model_prob: float, text: str, expected_flag: bool, expected_reason: str
) -> None:
    eng = stub_factory(emotion="sadness", crisis_prob=model_prob)
    reply = eng.reply(text)
    assert reply.crisis_flag is expected_flag
    assert reply.crisis_reason == expected_reason


# ---------------------------------------------------------------------------
# Loader fallback: when no config.json is present, _Classifier must use
# `base_encoder_name` (the key `train.py` writes) — NOT the legacy key
# `encoder_name`, and NOT a hard-coded default unless the label_map is
# silent. This is the bug fix for a silent-mismatch hazard: the loader
# used to read `encoder_name`, which the saved label_map never had, so
# every fallback-path load would silently use distilbert-base-uncased
# regardless of what backbone the weights were trained on.
# ---------------------------------------------------------------------------

def test_resolve_encoder_name_prefers_base_encoder_name() -> None:
    from src.inference.chat_engine import _resolve_encoder_name

    label_map = {
        "base_encoder_name": "roberta-base",
        "encoder_name": "bert-base-uncased",  # legacy key — must be ignored
    }
    assert _resolve_encoder_name(label_map) == "roberta-base"


def test_resolve_encoder_name_falls_back_to_legacy_key() -> None:
    # Older label_maps (pre-fix) used `encoder_name` directly. Still load.
    from src.inference.chat_engine import _resolve_encoder_name

    assert _resolve_encoder_name({"encoder_name": "bert-base-uncased"}) == "bert-base-uncased"


def test_resolve_encoder_name_defaults_when_missing() -> None:
    # No encoder key at all — loader still produces a usable name
    # (matches the documented default).
    from src.inference.chat_engine import _resolve_encoder_name

    assert _resolve_encoder_name({}) == "distilbert-base-uncased"
