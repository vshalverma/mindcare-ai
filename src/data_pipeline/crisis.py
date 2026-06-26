"""Stage 3: Apply ``crisis_flag`` heuristic to every normalized row.

Two-tier approach:
  Tier 1 — dataset labels  : suicide_depression rows already carry
                              ``crisis_flag=True`` set by ``normalize.py``.
  Tier 2 — keyword pass    : scan every text for any phrase in
                              ``crisis_keywords.ALL_PATTERNS``.

Rows already flagged by Tier 1 are preserved; the Tier-2 pass only adds
flags where there were none. ``crisis_source`` records who tripped the
flag for downstream auditing.
"""

from __future__ import annotations

import re

import pandas as pd

from .crisis_keywords import ALL_PATTERNS


def _compile_patterns() -> list[re.Pattern[str]]:
    """Compile keyword patterns once. Word-boundary anchored to reduce noise."""
    compiled: list[re.Pattern[str]] = []
    for phrase in ALL_PATTERNS:
        # Escape regex metachars in the phrase; \b won't always work cleanly
        # around apostrophes, so we use lookarounds on word characters.
        escaped = re.escape(phrase)
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        compiled.append(pattern)
    return compiled


_COMPILED_PATTERNS = _compile_patterns()


def _scan_text(text: str) -> str | None:
    """Return the first matching phrase, or None."""
    if not text:
        return None
    for phrase, pattern in zip(ALL_PATTERNS, _COMPILED_PATTERNS):
        if pattern.search(text):
            return phrase
    return None


def apply_crisis_flags(
    df: pd.DataFrame,
    text_col: str = "text",
    flag_col: str = "crisis_flag",
    source_col: str = "crisis_source",
) -> pd.DataFrame:
    """Add/update crisis flags in-place-safe copy.

    - If ``flag_col`` is True but ``source_col`` is empty/NA, set source to
      ``dataset_label`` (covers the suicide_depression case).
    - For rows not already flagged, run the keyword heuristic. The first
      match wins; the matched phrase is recorded as ``heuristic:<phrase>``.
    """
    df = df.copy()

    # Tier 1: ensure source is populated for rows already flagged.
    already_flagged = df[flag_col].fillna(False).astype(bool)
    missing_source = already_flagged & (df[source_col].isna() | (df[source_col].astype(str) == ""))
    df.loc[missing_source, source_col] = "dataset_label"

    # Tier 2: keyword heuristic on the rest.
    not_flagged = ~already_flagged
    if not_flagged.any():
        sample = df.loc[not_flagged, text_col].astype(str)
        first_match = sample.apply(_scan_text)
        newly_flagged = first_match.notna()

        idx_to_flag = sample.index[newly_flagged]
        df.loc[idx_to_flag, flag_col] = True
        df.loc[idx_to_flag, source_col] = first_match.loc[idx_to_flag].apply(
            lambda phrase: f"heuristic:{phrase}"
        )

    return df


def run(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply crisis flags to every dataset's DataFrame."""
    out: dict[str, pd.DataFrame] = {}
    for slug, df in datasets.items():
        out[slug] = apply_crisis_flags(df)
        n_flagged = int(out[slug]["crisis_flag"].fillna(False).sum())
        print(f"[crisis] {slug}: {n_flagged} / {len(out[slug])} rows flagged")
    return out
