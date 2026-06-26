"""CLI entrypoint for the mindcare-ai data pipeline.

Usage:
    python -m src.data_pipeline.run_pipeline --stage all
    python -m src.data_pipeline.run_pipeline --stage download
    python -m src.data_pipeline.run_pipeline --stage normalize
    python -m src.data_pipeline.run_pipeline --stage split
    python -m src.data_pipeline.run_pipeline --stage validate
    python -m src.data_pipeline.run_pipeline --stage download --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Make sure we can run as a script too (python src/data_pipeline/run_pipeline.py).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline import download, normalize, crisis, split, validate  # noqa: E402


STAGES = ("download", "normalize", "crisis", "split", "validate", "all")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mindcare-ai data pipeline",
        description="Download -> Normalize -> Crisis-flag -> Split -> Validate",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Which stage(s) to run. 'all' runs every stage in order.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if cached files exist.",
    )
    args = parser.parse_args(argv)

    stages = STAGES[:-1] if args.stage == "all" else [args.stage]

    # Stage 1: download
    if "download" in stages:
        print("\n=== STAGE 1: download ===")
        download.run(force=args.force)

    # Stage 2: normalize
    if "normalize" in stages:
        print("\n=== STAGE 2: normalize ===")
        norm_out = normalize.run()

    # Stage 3: crisis-flag
    if "crisis" in stages:
        print("\n=== STAGE 3: crisis flag ===")
        if "normalize" not in stages:
            # Re-normalize from disk if user skipped normalize stage.
            norm_out = normalize.run()
        crisis_out = crisis.run(norm_out)
    elif "normalize" in stages and {"split", "validate"} & set(stages):
        # Crisis flagging is required before split/validate to populate the column.
        print("\n=== STAGE 3 (auto): crisis flag ===")
        crisis_out = crisis.run(norm_out)
    else:
        crisis_out = {}

    # Stage 4: split
    if "split" in stages:
        print("\n=== STAGE 4: split ===")
        if not crisis_out:
            # Load fresh from disk.
            crisis_out = {slug: df for slug, df in normalize.run().items()}
            crisis_out = crisis.run(crisis_out)
        split.run(crisis_out)

    # Stage 5: validate
    if "validate" in stages:
        print("\n=== STAGE 5: validate ===")
        validate.run()

    print("\n[run_pipeline] DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
