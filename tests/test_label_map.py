"""Tests for the saved label_map.json.

Catches regressions like: a typo in an alias, an emotion missing from the
id map, or two emotions sharing the same id.
"""

from __future__ import annotations


def test_label_map_has_three_required_keys(label_map: dict) -> None:
    assert "emotions" in label_map
    assert "emotion_to_id" in label_map
    assert "emotion_aliases" in label_map


def test_emotions_list_matches_id_keys(label_map: dict) -> None:
    emotions = set(label_map["emotions"])
    id_keys = set(label_map["emotion_to_id"].keys())
    assert emotions == id_keys, (
        f"emotions list and emotion_to_id keys differ: "
        f"only-in-list={emotions - id_keys}, only-in-keys={id_keys - emotions}"
    )


def test_emotion_to_id_ids_are_unique(label_map: dict) -> None:
    ids = list(label_map["emotion_to_id"].values())
    assert len(ids) == len(set(ids)), f"duplicate ids in emotion_to_id: {ids}"


def test_emotion_to_id_ids_form_dense_range(label_map: dict) -> None:
    # The classifier head expects num_emotions = max_id + 1; a gap would
    # waste parameters and silently break the head shape.
    ids = sorted(label_map["emotion_to_id"].values())
    assert ids == list(range(len(ids))), f"ids are not a dense 0..N-1 range: {ids}"


def test_every_alias_points_to_known_emotion(label_map: dict) -> None:
    emotions = set(label_map["emotions"])
    bad = {
        alias: target
        for alias, target in label_map["emotion_aliases"].items()
        if target not in emotions
    }
    assert not bad, f"aliases point to unknown emotions: {bad}"


def test_every_alias_source_is_lowercase(label_map: dict) -> None:
    # Aliases come from raw dataset tokens and should stay normalised.
    bad = [a for a in label_map["emotion_aliases"] if a != a.lower()]
    assert not bad, f"aliases are not lowercase: {bad}"


def test_neutral_is_in_taxonomy(label_map: dict) -> None:
    # The chat engine falls back to "neutral" if the predicted emotion
    # isn't in the response table; removing it would break that fallback.
    assert "neutral" in label_map["emotions"]
