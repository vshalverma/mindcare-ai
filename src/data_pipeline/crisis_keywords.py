"""Crisis-keyword phrase list used by `crisis.py`.

Curated, lower-case substrings. Matching is done case-insensitively against
the `text` column. All matches are stored in `crisis_source` for audit.

IMPORTANT: This is a first-pass heuristic. It WILL produce false positives
on negative constructions like "I don't want to die". A negation-aware
refinement is planned for Phase 2.

Sources cross-referenced:
  - Reddit r/SuicideWatch common phrasing
  - Public crisis-text-detection literature keywords
"""

from __future__ import annotations

# Direct expressions of self-harm / suicidal ideation.
DIRECT_CRISIS_PATTERNS: tuple[str, ...] = (
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
)

# Indirect but high-signal phrases.
INDIRECT_CRISIS_PATTERNS: tuple[str, ...] = (
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

# All patterns combined. Order is preserved for reporting the first match.
ALL_PATTERNS: tuple[str, ...] = DIRECT_CRISIS_PATTERNS + INDIRECT_CRISIS_PATTERNS