"""Crisis-keyword phrase list used by `crisis.py`.

Curated, lower-case substrings. Matching is done case-insensitively against
the `text` column. All matches are stored in `crisis_source` for audit.

This is a first-pass heuristic and WILL produce false positives on
negative constructions like "I don't want to die". The keyword gate
is paired with the model's crisis head in `ChatEngine`, so the model
should be the primary signal and the keyword list is a backstop.

Sources cross-referenced:
  - Reddit r/SuicideWatch common phrasing
  - Public crisis-text-detection literature keywords
"""

from __future__ import annotations

# All patterns combined. Direct (self-harm / suicidal ideation) and
# indirect (high-signal but less specific) phrases are interleaved by
# category for readability; matching is the same either way.
ALL_PATTERNS: tuple[str, ...] = (
    # Direct expressions of self-harm / suicidal ideation.
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
    # Indirect but high-signal phrases.
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