"""Stage 1: Download datasets from HuggingFace into ``data/raw/<slug>/``.

Idempotent. Re-running skips files that already exist unless ``--force`` is
passed. Each dataset is converted to CSV format and a small JSON manifest is
written alongside it with provenance information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from .config import DATASETS, ensure_dirs, raw_dir, PROJECT_ROOT


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class DatasetManifest:
    slug: str
    hf_id: str
    hf_config: str | None
    license: str
    splits: dict[str, int]            # split_name -> row_count
    downloaded_at: str                # ISO 8601
    file_sha256: dict[str, str]       # relative file name -> sha256
    output_dir: str                   # relative to PROJECT_ROOT

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _sha256_of_file(path: Path) -> str:
    """SHA-256 of the first 64 KiB of a file (fast reproducibility tag)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(64 * 1024))
    return h.hexdigest()


def _row_count(path: Path) -> int:
    """Count CSV rows minus header."""
    with path.open(encoding="utf-8") as fh:
        return max(0, sum(1 for _ in fh) - 1)


# ---------------------------------------------------------------------------
# Generic downloader (works for any pre-split HF dataset)
# ---------------------------------------------------------------------------

def _download_pre_split(
    slug: str,
    force: bool = False,
) -> DatasetManifest:
    """Download a dataset that already has train/val/test splits in HF.

    The splits/files mapping comes from ``DATASETS[slug]['split_file_map']``:
    - a dict ``{hf_split_name: file_name_on_disk}`` for fixed mapping
    - the string ``"from_hf"`` to use the HF split names verbatim
    """
    cfg = DATASETS[slug]
    out_dir = raw_dir(slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide the mapping.
    sfm = cfg["split_file_map"]
    if sfm == "from_hf":
        print(f"[{slug}] Downloading {cfg['hf_id']} (split names preserved)...")
        ds = load_dataset(cfg["hf_id"])
        split_file_map = {split_name: f"{split_name}.csv" for split_name in ds.keys()}
    else:
        # Check whether we already have all files.
        expected_files = list(sfm.values())
        all_present = all((out_dir / f).exists() for f in expected_files)
        if all_present and not force:
            print(f"[{slug}] Already downloaded at {out_dir}. Use --force to re-download.")
            return _build_manifest_from_existing(slug, out_dir, sfm)
        print(f"[{slug}] Downloading {cfg['hf_id']}...")
        ds = load_dataset(cfg["hf_id"], cfg["hf_config"])
        split_file_map = dict(sfm)

    # Write each split.
    rows: dict[str, int] = {}
    csv_paths: dict[str, Path] = {}
    for hf_split, file_name in split_file_map.items():
        if hf_split not in ds:
            print(f"  [warn] HF split '{hf_split}' missing in {slug}; skipping.")
            continue
        out_path = out_dir / file_name
        df = ds[hf_split].to_pandas()
        df.to_csv(out_path, index=False, encoding="utf-8")
        rows[hf_split] = len(df)
        csv_paths[hf_split] = out_path
        print(f"  -> {file_name}: {len(df)} rows")

    manifest = DatasetManifest(
        slug=slug,
        hf_id=cfg["hf_id"],
        hf_config=cfg["hf_config"],
        license=cfg["license"],
        splits=rows,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        file_sha256={p.name: _sha256_of_file(p) for p in csv_paths.values()},
        output_dir=str(out_dir.relative_to(PROJECT_ROOT)),
    )
    (out_dir / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def _build_manifest_from_existing(
    slug: str,
    out_dir: Path,
    split_file_map: dict[str, str],
) -> DatasetManifest:
    cfg = DATASETS[slug]
    csv_paths = [out_dir / file_name for file_name in split_file_map.values()]
    rows = {hf_split: _row_count(out_dir / file_name)
            for hf_split, file_name in split_file_map.items()}
    manifest = DatasetManifest(
        slug=slug,
        hf_id=cfg["hf_id"],
        hf_config=cfg["hf_config"],
        license=cfg["license"],
        splits=rows,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        file_sha256={p.name: _sha256_of_file(p) for p in csv_paths},
        output_dir=str(out_dir.relative_to(PROJECT_ROOT)),
    )
    (out_dir / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DOWNLOADERS = {
    "go_emotions": _download_pre_split,
    "empathetic_dialogues": _download_pre_split,
    "suicide_depression": _download_pre_split,
}


def run(force: bool = False) -> list[DatasetManifest]:
    ensure_dirs()
    manifests: list[DatasetManifest] = []
    for slug, fn in DOWNLOADERS.items():
        try:
            manifests.append(fn(slug, force=force))
        except Exception as exc:
            print(f"[{slug}] DOWNLOAD FAILED: {exc!r}", file=sys.stderr)
            raise
    return manifests


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download mindcare-ai datasets from HuggingFace.")
    p.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(force=args.force)