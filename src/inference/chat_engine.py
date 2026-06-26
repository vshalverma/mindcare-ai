"""Inference + chatbot module for mindcare-ai.

`ChatEngine` ties together:
  1. a tiny text classifier (emotion + crisis) trained by `src.models.train`
  2. a fast keyword safety gate (the same heuristics used in the pipeline)
  3. a small retrieval-augmented response picker

The classifier drives the chatbot's empathetic tone (it picks the response
template whose emotion label matches the predicted emotion). The keyword
gate + classifier's crisis head jointly decide whether to surface a
crisis-safety banner — a conservative OR of the two signals so we err on
the side of catching real risk.

The response generator is deliberately template-based in this first
version — no LLM API calls. This keeps the system self-contained,
predictable, and auditable (important for a safety-sensitive domain).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Keyword safety gate
# ---------------------------------------------------------------------------
# A small, fast substring match that mirrors the data-pipeline heuristic.
# It's the second line of defense alongside the model's crisis head.

_CRISIS_PHRASES: tuple[str, ...] = (
    "kill myself",
    "killing myself",
    "end my life",
    "ending my life",
    "end it all",
    "ending it all",
    "take my life",
    "taking my life",
    "want to die",
    "wanna die",
    "wish i was dead",
    "wish i were dead",
    "better off without me",
    "no reason to live",
    "nothing to live for",
    "going to kill myself",
    "going to end it",
    "commit suicide",
    "committing suicide",
    "suicide plan",
    "suicidal thoughts",
    "thoughts of suicide",
    "hurt myself",
    "hurting myself",
    "self harm",
    "self-harm",
    "cut myself",
    "cutting myself",
    "overdose",
    "jump off",
    "hang myself",
    "shoot myself",
    "i can't go on",
    "i cant go on",
    "i can't take it anymore",
    "i cant take it anymore",
    "no way out",
    "tired of living",
    "tired of being alive",
    "everyone would be better off",
    "wouldn't miss me",
    "no point in living",
    "what's the point",
    "whats the point",
    "i give up",
    "done with life",
)

# Pre-compile once. Word-boundary lookarounds (instead of \b) avoid the
# apostrophe quirk in "what's".
_CRISIS_PATTERNS = [
    re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
    for phrase in _CRISIS_PHRASES
]


def keyword_crisis_score(text: str) -> float:
    """Return a [0, 1] heuristic score: 1 if any phrase matches else 0."""
    if not text:
        return 0.0
    for pat in _CRISIS_PATTERNS:
        if pat.search(text):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------
# Curated empathetic responses keyed by emotion label. The model picks a
# label; we pick a template; we lightly personalize with the user's text.

_RESPONSES: dict[str, tuple[str, ...]] = {
    "sadness": (
        "I'm really sorry you're feeling this way. It sounds heavy.",
        "That sounds painful. Would you like to tell me more about what's going on?",
        "I'm here for you. Take your time — what's on your mind?",
    ),
    "fear": (
        "That sounds scary. You're not alone in this.",
        "It's okay to feel afraid. What's making things feel unsafe right now?",
    ),
    "anger": (
        "I hear that you're frustrated. What happened that brought this on?",
        "That sounds infuriating. Tell me more.",
    ),
    "grief": (
        "I'm so sorry for your loss. Grief can be overwhelming.",
        "That's a heavy thing to carry. I'm here to listen.",
    ),
    "disappointment": (
        "That sounds disappointing. What were you hoping would happen?",
    ),
    "nervousness": (
        "Feeling anxious can be exhausting. What's been on your mind lately?",
        "Take a breath — we're in this together. What's worrying you?",
    ),
    "joy": (
        "That's wonderful! I'd love to hear more about it.",
    ),
    "love": (
        "It sounds like you really care. Tell me about them.",
    ),
    "gratitude": (
        "That's lovely. What are you feeling most grateful for?",
    ),
    "neutral": (
        "Tell me more.",
        "I'm listening. What's on your mind?",
        "How are you feeling right now?",
    ),
}


def _pick_response(emotion: str) -> str:
    """Return an empathetic reply for the given emotion label."""
    pool = _RESPONSES.get(emotion)
    if pool is None:
        # Fall back to a neutral prompt if we don't have a tailored template.
        return _RESPONSES["neutral"][0]
    return pool[np.random.randint(0, len(pool))]


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

@dataclass
class ClassifierPrediction:
    emotion_label: str
    emotion_probs: dict[str, float]
    crisis_prob: float


class _Classifier:
    """Lazy wrapper around the trained MultiTaskClassifier.

    Loads weights once on first use. The HF model directory lives at
    ``<output_dir>/final`` per ``src.models.train``.
    """

    def __init__(self, model_dir: Path, device: str | None = None) -> None:
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Local imports keep this module importable without torch installed
        # (handy for unit tests that only exercise the keyword gate).
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from src.models.train import MultiTaskClassifier

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.label_map = json.loads(
            (model_dir / "label_map.json").read_text(encoding="utf-8")
        )
        self.emotions = list(self.label_map["emotions"])

        # Some older checkpoints (saved before train.py was fixed to copy
        # config.json explicitly) don't have one next to the weights. Fall
        # back to the base encoder name stored in the label map, or — last
        # resort — to the distilbert-base-uncased default.
        config_path = model_dir / "config.json"
        cache_dir = None  # not exposed; trust HF's default cache
        if not config_path.exists():
            encoder_name = self.label_map.get("encoder_name", "distilbert-base-uncased")
            self.model = MultiTaskClassifier(
                encoder_name=encoder_name,
                num_emotions=len(self.emotions),
                cache_dir=cache_dir,
            )
        else:
            self.model = MultiTaskClassifier(
                encoder_name=str(model_dir),
                num_emotions=len(self.emotions),
                cache_dir=cache_dir,
            )
        # Load weights saved by `trainer.save_model`. The Trainer dumps
        # the full module (encoder + heads), so load_state_dict works.
        state_dict_file = model_dir / "model.safetensors"
        if state_dict_file.exists():
            from safetensors.torch import load_file
            self.model.load_state_dict(load_file(state_dict_file))
        else:
            bin_file = model_dir / "pytorch_model.bin"
            self.model.load_state_dict(
                torch.load(bin_file, map_location=self.device)
            )
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, texts: Sequence[str]) -> list[ClassifierPrediction]:
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        )
        emo_logits = out["emotion_logits"].float().cpu().numpy()
        crisis_logit = out["crisis_logit"].float().cpu().numpy()
        emo_probs = _softmax(emo_logits, axis=-1)
        crisis_prob = 1.0 / (1.0 + np.exp(-crisis_logit))

        results: list[ClassifierPrediction] = []
        for probs, c_prob in zip(emo_probs, crisis_prob):
            idx = int(np.argmax(probs))
            label = self.emotions[idx]
            top = {self.emotions[i]: float(p) for i, p in enumerate(probs)}
            results.append(
                ClassifierPrediction(
                    emotion_label=label,
                    emotion_probs=top,
                    crisis_prob=float(c_prob),
                )
            )
        return results


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# ChatEngine
# ---------------------------------------------------------------------------

@dataclass
class ChatReply:
    text: str
    emotion: str
    emotion_confidence: float
    crisis_prob: float
    crisis_flag: bool
    crisis_reason: str


class ChatEngine:
    """High-level chatbot API used by the Streamlit UI.

    Single entrypoint: ``engine.reply(user_text)`` -> ``ChatReply``.
    The classifier is loaded lazily on first call so that the import of
    this module is cheap.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        crisis_threshold: float = 0.5,
    ) -> None:
        default_dir = Path("models/checkpoints/final")
        self.model_dir = Path(model_dir) if model_dir else default_dir
        self.crisis_threshold = float(crisis_threshold)
        self._classifier: _Classifier | None = None

    def _get_classifier(self) -> _Classifier:
        if self._classifier is None:
            self._classifier = _Classifier(self.model_dir)
        return self._classifier

    def predict(self, text: str) -> ClassifierPrediction:
        """Run the classifier only (useful for the UI to display scores)."""
        return self._get_classifier().predict([text])[0]

    def reply(self, text: str) -> ChatReply:
        """Produce a reply + safety signal for one user turn."""
        text = (text or "").strip()
        if not text:
            return ChatReply(
                text="I'm here. Tell me how you're feeling.",
                emotion="neutral",
                emotion_confidence=1.0,
                crisis_prob=0.0,
                crisis_flag=False,
                crisis_reason="empty_input",
            )

        pred = self.predict(text)
        kw_score = keyword_crisis_score(text)
        # Conservative: trigger the safety banner if EITHER the model
        # crosses the threshold OR the keyword gate fires.
        model_flag = pred.crisis_prob >= self.crisis_threshold
        crisis_flag = bool(model_flag or kw_score >= 1.0)
        if crisis_flag:
            if model_flag and kw_score >= 1.0:
                reason = "model+keyword"
            elif model_flag:
                reason = "model"
            else:
                reason = "keyword"

        reply_text = _pick_response(pred.emotion_label)

        return ChatReply(
            text=reply_text,
            emotion=pred.emotion_label,
            emotion_confidence=float(pred.emotion_probs.get(pred.emotion_label, 0.0)),
            crisis_prob=float(pred.crisis_prob),
            crisis_flag=crisis_flag,
            crisis_reason=reason if crisis_flag else "none",
        )


# ---------------------------------------------------------------------------
# Crisis resources (constant, easy to update)
# ---------------------------------------------------------------------------

CRISIS_RESOURCES = {
    "US": {
        "name": "988 Suicide & Crisis Lifeline",
        "call": "988",
        "text": "988",
        "site": "https://988lifeline.org",
    },
    "IN": {
        "name": "iCall (India)",
        "call": "+91-9152987821",
        "text": "+91-9152987821",
        "site": "https://icallhelpline.org",
    },
}