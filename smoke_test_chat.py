"""Smoke test for ChatEngine.

Exercises the four behaviours we care about for the demo:
  1. normal happy input     -> no crisis banner, normal reply
  2. sadness input          -> no crisis banner, normal reply
  3. keyword-gate input     -> crisis banner (keyword trigger)
  4. empty input            -> graceful neutral fallback

Exits non-zero on any unexpected behaviour so this can be wired into CI later.
Run from the project root:
    venv/Scripts/python smoke_test_chat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.chat_engine import ChatEngine  # noqa: E402


CASES = [
    # (label, input_text, expect_crisis_flag, expect_reason_in)
    ("happy",      "I just got promoted at work, I am so happy!",       False, {"none"}),
    ("sadness",    "I feel really down and I have been crying all day", False, {"none"}),
    ("keyword",    "I want to kill myself, there is no way out",       True,  {"model", "keyword", "model+keyword"}),
    ("keyword_2",  "I am going to end my life tonight",                True,  {"model", "keyword", "model+keyword"}),
    ("empty",      "",                                                 False, {"empty_input"}),
]


def main() -> int:
    print("[smoke] loading ChatEngine (this hits the trained checkpoint)...")
    engine = ChatEngine()
    # Trigger lazy load.
    engine.predict("warmup")
    print("[smoke] engine loaded.\n")

    failures: list[str] = []

    for label, text, expect_flag, expect_reasons in CASES:
        reply = engine.reply(text)
        ok_flag = reply.crisis_flag is expect_flag
        ok_reason = reply.crisis_reason in expect_reasons
        status = "OK" if (ok_flag and ok_reason) else "FAIL"
        print(
            f"[{status}] case={label:9s} "
            f"emotion={reply.emotion:14s} "
            f"emo_conf={reply.emotion_confidence:.2f} "
            f"crisis_prob={reply.crisis_prob:.2f} "
            f"flag={reply.crisis_flag!s:5s} "
            f"reason={reply.crisis_reason}"
        )
        if status == "FAIL":
            failures.append(
                f"{label}: expected flag={expect_flag}/reason in {expect_reasons}, "
                f"got flag={reply.crisis_flag}/reason={reply.crisis_reason}"
            )
        # Sanity: every reply must produce some non-empty text.
        if not reply.text:
            failures.append(f"{label}: empty reply text")
            print(f"[FAIL] case={label} produced empty text")

    print()
    if failures:
        print(f"[smoke] FAILED: {len(failures)} issue(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[smoke] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
