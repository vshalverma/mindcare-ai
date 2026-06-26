"""Shared fixtures for the mindcare-ai test suite.

Keeps tests GPU-free and model-free: every test that exercises the
ChatEngine uses a stub classifier injected via monkey-patch, so CI on a
laptop / a runner with no model checkpoint still passes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Stub classifier
# ---------------------------------------------------------------------------

@dataclass
class StubPrediction:
    """Mirrors ChatEngine.ClassifierPrediction (duck-typed)."""
    emotion_label: str
    emotion_probs: dict[str, float] = field(default_factory=dict)
    crisis_prob: float = 0.0


class StubClassifier:
    """Drop-in replacement for ChatEngine._Classifier.

    Returns a sequence of canned predictions in order — tests register the
    queue via `engine._classifier = StubClassifier([...])` and the engine
    consumes one prediction per .predict([text]) call.
    """

    def __init__(self, queue: list[StubPrediction] | None = None) -> None:
        self.queue: list[StubPrediction] = list(queue or [])
        self.calls: list[str] = []

    def predict(self, texts):
        # Pop one prediction per call (we always pass single strings in tests).
        if not self.queue:
            raise AssertionError("StubClassifier ran out of canned predictions")
        self.calls.extend(texts)
        return [self.queue.pop(0)]


@pytest.fixture
def stub_factory():
    """Returns a factory (emotion, crisis_prob) -> ChatEngine with stub."""

    def _make(emotion: str = "neutral", crisis_prob: float = 0.0):
        from src.inference.chat_engine import ChatEngine

        # Skip the lazy loader by injecting a stub before any .reply() call.
        eng = ChatEngine(model_dir=Path("models/checkpoints/final"))
        eng._classifier = StubClassifier(
            [StubPrediction(emotion_label=emotion, crisis_prob=crisis_prob)]
        )
        return eng

    return _make


# ---------------------------------------------------------------------------
# Real label_map (read-only fixture, never modified)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def label_map() -> dict:
    path = PROJECT_ROOT / "models" / "checkpoints" / "final" / "label_map.json"
    if not path.exists():
        pytest.skip(f"label_map.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))
