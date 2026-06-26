"""Tests for the keyword crisis safety gate.

These run without torch / GPU / model checkpoint.
"""

from __future__ import annotations

import pytest

from src.inference.chat_engine import keyword_crisis_score


# (input_text, expected_score) — covers positives, negatives, edge cases.
CASES = [
    # ---- direct positives (each phrase from the gate's table) ----
    ("I want to kill myself", 1.0),
    ("I'm going to end my life tonight", 1.0),
    ("I've been having suicidal thoughts lately", 1.0),
    ("I want to hurt myself", 1.0),
    ("I cut myself last week", 1.0),
    ("I'm thinking about an overdose", 1.0),
    ("I might jump off the bridge", 1.0),
    ("I have a suicide plan", 1.0),
    ("no reason to live", 1.0),
    ("what's the point of any of this", 1.0),
    ("I'm done with life", 1.0),
    # ---- case insensitivity ----
    ("I WANT TO KILL MYSELF", 1.0),
    ("Suicide Plan", 1.0),
    # ---- embedded in longer text ----
    ("Honestly, I want to kill myself and I'm not sure what to do", 1.0),
    # ---- negatives (no trigger) ----
    ("I am so happy today!", 0.0),
    ("I'm sad but I'll be okay", 0.0),
    ("My dog died last year", 0.0),
    ("", 0.0),
    # ---- near-misses that must NOT fire (word boundaries) ----
    ("I killed the spider", 0.0),                 # "kill" alone, not "kill myself"
    ("My life is great", 0.0),                    # "life" alone, not "end my life"
    ("I hurt my ankle", 0.0),                     # "hurt" alone, not "hurt myself"
    ("This is a self-harm free zone (read a paper on self-harm)", 1.0),  # "self-harm" present
    # ---- adjacent punctuation must not break the match ----
    ("I want to kill myself.", 1.0),
    ("kill myself!", 1.0),
    ("end my life?", 1.0),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_keyword_crisis_score(text: str, expected: float) -> None:
    assert keyword_crisis_score(text) == expected


def test_empty_string_is_zero() -> None:
    assert keyword_crisis_score("") == 0.0


def test_returns_float_in_unit_interval() -> None:
    # Defence-in-depth: the function is documented to return [0, 1].
    for sample in ("kill myself", "hello world", "", "no way out"):
        score = keyword_crisis_score(sample)
        assert 0.0 <= score <= 1.0
