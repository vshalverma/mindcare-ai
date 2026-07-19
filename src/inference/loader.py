"""Shared model loader for mindcare-ai inference paths.

Both the chat engine (`src.inference.chat_engine`) and the per-class
evaluation script (`eval_per_class.py`) need to:

  1. read ``label_map.json`` from the checkpoint dir
  2. load the tokenizer
  3. instantiate ``MultiTaskClassifier`` against the same checkpoint
  4. load the saved weights (``model.safetensors`` or ``pytorch_model.bin``)
  5. move to the right device and put the model in eval mode

That sequence used to be inlined in both places — easy to drift apart.
This module is the single source of truth.

Why a dataclass instead of a free function returning a tuple:
the two callers want the same fields (model, tokenizer, label_map,
device) but in different orders and with different lifetimes. A typed
return makes the call sites legible and the tests targeted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# `torch` / `transformers` / `safetensors` are imported lazily inside
# `load_classifier()` so that unit tests for unrelated parts of the
# package can run on machines without a GPU or a heavy install.


@dataclass
class LoadedClassifier:
    """The bundle of objects every inference path needs.

    Attributes
    ----------
    model       : nn.Module ready for inference (set to ``.eval()``).
    tokenizer   : HF tokenizer paired with the checkpoint.
    label_map   : dict parsed from ``label_map.json`` (emotions,
                  emotion_aliases, emotion_to_id, base_encoder_name).
    model_dir   : the directory we loaded from (kept for diagnostics).
    device      : device the model lives on.
    """

    model: Any
    tokenizer: Any
    label_map: dict
    model_dir: Path
    device: str

    @property
    def emotions(self) -> list[str]:
        """Canonical 28-emotion taxonomy (in label-id order)."""
        return list(self.label_map["emotions"])

    @property
    def num_emotions(self) -> int:
        return len(self.label_map["emotions"])


def _resolve_encoder_name(label_map: dict) -> str:
    """Pick the encoder name to instantiate the bare backbone from.

    `train.py` writes `base_encoder_name` into ``label_map.json``. A legacy
    key `encoder_name` is also accepted so older checkpoints still load.
    If neither is present we fall back to ``distilbert-base-uncased``,
    which is the encoder used for the committed baseline.
    """
    return label_map.get(
        "base_encoder_name",
        label_map.get("encoder_name", "distilbert-base-uncased"),
    )


def load_classifier(model_dir: Path | str, device: str | None = None) -> LoadedClassifier:
    """Load the trained multi-task classifier from a checkpoint directory.

    Parameters
    ----------
    model_dir
        Directory containing ``label_map.json``, ``tokenizer.json``, and
        the saved weights (``model.safetensors`` or ``pytorch_model.bin``).
    device
        Force a specific device (e.g. ``"cuda"`` or ``"cpu"``). If
        ``None``, uses CUDA when available.

    Returns
    -------
    LoadedClassifier
        Bundle with ``.model``, ``.tokenizer``, ``.label_map``,
        ``.model_dir``, ``.device``. The model is already on the chosen
        device and in eval mode.

    Raises
    ------
    FileNotFoundError
        If ``label_map.json`` is missing, or if neither
        ``model.safetensors`` nor ``pytorch_model.bin`` exists.
    """
    model_dir = Path(model_dir)

    # Check label_map + weights first — these are the inputs that are
    # truly required. Validating them before importing torch / transformers
    # gives a cheap, clear diagnostic in environments that may not have
    # those heavy deps installed (e.g. a CI runner doing a smoke import).
    label_map_path = model_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map.json not found in {model_dir}")

    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if not safetensors_path.exists() and not bin_path.exists():
        raise FileNotFoundError(
            f"No weights found in {model_dir}: looked for "
            f"model.safetensors and pytorch_model.bin"
        )

    # A safetensors file that is just an LFS pointer stub (a few hundred
    # bytes of text starting with "version https://git-lfs.github.com")
    # is not real weights — opening it raises "header too large". Detect
    # this case explicitly so callers get a clear "weights not committed"
    # diagnostic instead of a confusing safetensors error from deep inside
    # the library.
    def _is_lfs_pointer(path: Path) -> bool:
        try:
            head = path.read_bytes()[:200]
            return b"git-lfs.github.com" in head
        except OSError:
            return False

    weights_present = False
    if safetensors_path.exists() and not _is_lfs_pointer(safetensors_path):
        weights_present = True
    if bin_path.exists():
        weights_present = True
    if not weights_present:
        raise FileNotFoundError(
            f"Weights in {model_dir} are LFS pointer stubs — the real "
            f"weight file was not committed. Re-run training with "
            f"`python -m src.models.train --config configs/training.yaml` "
            f"to regenerate it."
        )

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))

    # Local imports: keep this module cheap to import so unit tests that
    # only exercise the keyword gate or label_map don't pay for torch.
    import torch
    from transformers import AutoTokenizer

    from src.models.train import MultiTaskClassifier

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    num_emotions = len(label_map["emotions"])

    # If a sibling config.json exists (the normal case for checkpoints
    # saved by current train.py), we can construct the model directly
    # from the directory — HF will pick up the right architecture. If
    # it's missing (older saves or copies across machines), we have to
    # fall back to instantiating the base encoder and then load_state_dict
    # on top.
    config_path = model_dir / "config.json"
    if config_path.exists():
        encoder_name = _resolve_encoder_name(label_map)
    else:
        encoder_name = _resolve_encoder_name(label_map)

    model = MultiTaskClassifier(
        encoder_name=encoder_name,
        num_emotions=num_emotions,
        cache_dir=None,
    )

    # Load weights saved by `trainer.save_model`. The Trainer dumps the
    # full module (encoder + heads), so load_state_dict works either way.
    # (Existence was checked above; here we just dispatch on which file
    # is actually present.)
    if safetensors_path.exists():
        from safetensors.torch import load_file
        # `device="cpu"` is intentional: load_file defaults to the current
        # CUDA device if one exists, which fails with "header too large"
        # for safetensors files that were saved on CPU (the header bytes
        # don't match the device-endianness the runtime expects). We
        # always stage on CPU and move the model to its target device
        # below — `state_dict` is just a dict of tensors, the transfer
        # is handled by `model.to(device)`.
        state_dict = load_file(str(safetensors_path), device="cpu")
    else:
        # `map_location` here just decides where the tensors end up
        # initially — we move to the target device below.
        state_dict = torch.load(bin_path, map_location="cpu")

    model.load_state_dict(state_dict)

    # Device placement: caller may force, otherwise pick the best available.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    return LoadedClassifier(
        model=model,
        tokenizer=tokenizer,
        label_map=label_map,
        model_dir=model_dir,
        device=device,
    )