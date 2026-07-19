"""Multi-task classifier training for mindcare-ai.

Two heads sharing a small encoder:
  - emotion head  : 28-class softmax over the canonical emotion taxonomy
  - crisis head   : binary sigmoid (positive = crisis-flagged)

Why multi-task? The emotion head uses the large GoEmotions corpus, the
crisis head uses the suicide_depression corpus + heuristic flags. Sharing
the encoder lets both benefit from the same general emotional-language
representations; for a tiny encoder on a 4 GB GPU this is a much better
trade-off than training two separate models.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    encoder_name: str
    max_length: int
    cache_dir: str
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    fp16: bool
    eval_every_steps: int
    seed: int
    max_train_samples: int
    max_eval_samples: int
    gradient_accumulation_steps: int
    dataloader_num_workers: int
    train_file: Path
    val_file: Path
    test_file: Path
    emotions: list[str]
    emotion_aliases: dict[str, str]

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        data_dir = Path(raw["data"]["processed_dir"])
        return cls(
            encoder_name=raw["model"]["name"],
            max_length=int(raw["model"]["max_length"]),
            cache_dir=raw["model"]["cache_dir"],
            output_dir=raw["training"]["output_dir"],
            num_train_epochs=int(raw["training"]["num_train_epochs"]),
            per_device_train_batch_size=int(raw["training"]["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(raw["training"]["per_device_eval_batch_size"]),
            learning_rate=float(raw["training"]["learning_rate"]),
            weight_decay=float(raw["training"]["weight_decay"]),
            warmup_ratio=float(raw["training"]["warmup_ratio"]),
            fp16=bool(raw["training"]["fp16"]),
            eval_every_steps=int(raw["training"]["eval_every_steps"]),
            seed=int(raw["training"]["seed"]),
            max_train_samples=int(raw["training"].get("max_train_samples", 0)),
            max_eval_samples=int(raw["training"].get("max_eval_samples", 5000)),
            gradient_accumulation_steps=int(
                raw["training"].get("gradient_accumulation_steps", 1)
            ),
            dataloader_num_workers=int(raw["training"].get("dataloader_num_workers", 0)),
            train_file=data_dir / raw["data"]["train_file"],
            val_file=data_dir / raw["data"]["val_file"],
            test_file=data_dir / raw["data"]["test_file"],
            emotions=list(raw["emotions"]),
            emotion_aliases=dict(raw.get("emotion_aliases", {})),
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class MindcareDataset(Dataset):
    """Pre-tokenized dataset for the multi-task classifier.

    Each row produces:
      - input_ids, attention_mask
      - emotion_label_id (int, -100 = no label / not used)
      - crisis_label      (float, -1.0 = not used)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        emotion_to_id: dict[str, int],
        emotion_aliases: dict[str, str],
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.emotion_to_id = emotion_to_id
        self.emotion_aliases = emotion_aliases

        # Pre-tokenize once (much faster than tokenizing in __getitem__).
        texts = df["text"].astype(str).tolist()
        enc = tokenizer(
            texts,
            truncation=True,
            padding=False,           # collator will pad dynamically
            max_length=max_length,
            return_tensors=None,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

        # Build label tensors.
        emo_labels: list[int] = []
        crisis_labels: list[float] = []
        for raw_emo, raw_crisis in zip(
            df["emotion_label"].tolist(), df["crisis_flag"].tolist()
        ):
            # Emotion label
            e = self._canonicalize_emotion(raw_emo)
            if e is None:
                emo_labels.append(-100)
            else:
                emo_labels.append(emotion_to_id.get(e, -100))

            # Crisis label — boolean from parquet (stored as bool)
            try:
                crisis_labels.append(1.0 if bool(raw_crisis) else 0.0)
            except Exception:
                crisis_labels.append(0.0)

        self.emotion_labels = emo_labels
        self.crisis_labels = crisis_labels

    def _canonicalize_emotion(self, raw: object) -> str | None:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return None
        s = str(raw).strip().lower()
        if not s:
            return None
        # Apply alias table, then check canonical set is enforced in the caller.
        s = self.emotion_aliases.get(s, s)
        return s

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "emotion_labels": self.emotion_labels[idx],
            "crisis_labels": self.crisis_labels[idx],
        }


@dataclass
class MultiTaskCollator:
    """Pads input_ids/attention_mask and stacks scalar labels into tensors."""

    tokenizer: object
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict]) -> dict:
        # Filter out non-tensor keys (labels) before handing to the tokenizer
        # pad, then re-attach them. Some tokenizers (e.g. distilbert) emit
        # token_type_ids which our model doesn't accept, so we drop extras.
        label_keys = ("emotion_labels", "crisis_labels")
        label_values = {k: [f[k] for f in features] for k in label_keys}
        seq_features = [
            {k: v for k, v in f.items() if k not in label_keys} for f in features
        ]
        batch = self.tokenizer.pad(
            seq_features,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        # Drop token_type_ids; our model doesn't use it and distilbert emits it.
        batch.pop("token_type_ids", None)
        batch["emotion_labels"] = torch.tensor(
            label_values["emotion_labels"], dtype=torch.long
        )
        batch["crisis_labels"] = torch.tensor(
            label_values["crisis_labels"], dtype=torch.float32
        )
        return batch


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiTaskClassifier(nn.Module):
    """Tiny encoder + emotion (multiclass) + crisis (binary) heads."""

    def __init__(self, encoder_name: str, num_emotions: int, cache_dir: str | None = None) -> None:
        super().__init__()
        
        # Fallback mechanism: if loading from a checkpoint directory that is missing config.json,
        # try to look up the base model name from label_map if present, or rescue with the argument.
        # `low_cpu_mem_usage=False` matters when `encoder_name` is a local
        # checkpoint directory: newer transformers defaults to True, which
        # instantiates the model on the `meta` device and then tries to copy
        # state-dict tensors into it (which fails with "Cannot copy out of
        # meta tensor"). Disabling it forces a normal CPU materialization,
        # which is safe for our small encoder.
        try:
            self.encoder = AutoModel.from_pretrained(
                encoder_name, cache_dir=cache_dir, low_cpu_mem_usage=False
            )
        except Exception:
            # Check if encoder_name looks like a directory path containing a label_map
            potential_map_path = Path(encoder_name) / "label_map.json"
            if potential_map_path.exists():
                try:
                    labels = json.loads(potential_map_path.read_text(encoding="utf-8"))
                    # If label_map tracks the base encoder name, extract it; otherwise fallback safely
                    fallback_name = labels.get("base_encoder_name", "distilbert-base-uncased")
                    print(f"[Warning] Failed to instantiate checkpoint from '{encoder_name}'. "
                          f"Falling back to base encoder model: '{fallback_name}'")
                    self.encoder = AutoModel.from_pretrained(
                        fallback_name, cache_dir=cache_dir, low_cpu_mem_usage=False
                    )
                except Exception:
                    # Reraise original fallback if parsing completely fails
                    raise
            else:
                raise

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.emotion_head = nn.Linear(hidden, num_emotions)
        self.crisis_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, emotion_labels=None, crisis_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] equivalent — first token's last_hidden_state.
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        emo_logits = self.emotion_head(pooled)
        crisis_logit = self.crisis_head(pooled).squeeze(-1)

        loss = None
        if emotion_labels is not None and crisis_labels is not None:
            bce = nn.BCEWithLogitsLoss()
            crisis_loss = bce(crisis_logit, crisis_labels)

            # CE is NaN if every label in the batch is -100 (e.g. a batch
            # of suicide_depression rows that have no emotion). Only
            # compute it when at least one row has a real label.
            valid = (emotion_labels != -100).any().item()
            if valid:
                ce = nn.CrossEntropyLoss(ignore_index=-100)
                emo_loss = ce(emo_logits, emotion_labels)
                loss = emo_loss + crisis_loss
            else:
                loss = crisis_loss

        return {
            "loss": loss,
            "emotion_logits": emo_logits,
            "crisis_logit": crisis_logit,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict[str, float]:
    """HF Trainer will call this with (predictions, label_ids).

    `predictions` is the tuple of arrays our model returned
    (emotion_logits, crisis_logit). `label_ids` is a tuple of arrays
    in the same order they were added in the collator: emotion_labels,
    crisis_labels.
    """
    emo_logits, crisis_logit = eval_pred.predictions
    emo_labels_arr, crisis_labels_arr = eval_pred.label_ids
    emo_labels = np.asarray(emo_labels_arr)
    crisis_labels = np.asarray(crisis_labels_arr)

    emo_preds = np.argmax(emo_logits, axis=-1)
    mask = emo_labels != -100
    emo_acc = float((emo_preds[mask] == emo_labels[mask]).mean()) if mask.any() else 0.0

    crisis_prob = 1.0 / (1.0 + np.exp(-crisis_logit))
    crisis_pred = (crisis_prob > 0.5).astype(np.int32)
    crisis_acc = float((crisis_pred == crisis_labels.astype(np.int32)).mean())

    # Naive balanced accuracy on the rare crisis class is more useful than
    # raw accuracy given the class imbalance.
    pos = (crisis_labels == 1).sum()
    neg = (crisis_labels == 0).sum()
    if pos > 0 and neg > 0:
        tpr = float(((crisis_pred == 1) & (crisis_labels == 1)).sum() / pos)
        tnr = float(((crisis_pred == 0) & (crisis_labels == 0)).sum() / neg)
        bal_acc = 0.5 * (tpr + tnr)
    else:
        bal_acc = 0.0

    return {
        "emotion_accuracy": emo_acc,
        "crisis_accuracy": crisis_acc,
        "crisis_balanced_accuracy": bal_acc,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df
    # Stratify by `dataset` so we keep the same proportion of
    # go_emotions / empathetic_dialogues / suicide_depression even
    # when we cap training size.
    counts = df["dataset"].value_counts()
    weights = (counts / counts.sum()).to_dict()
    n_per = {ds: max(1, int(round(n * w))) for ds, w in weights.items()}
    # If rounding dropped us short, top up from the largest slice.
    diff = n - sum(n_per.values())
    if diff != 0:
        biggest = max(n_per, key=n_per.get)
        n_per[biggest] += diff
    parts = []
    for ds, k in n_per.items():
        sub = df.loc[df["dataset"] == ds]
        parts.append(sub.sample(n=min(k, len(sub)), random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _to_parquet_in_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes so HF Trainer + parquet types don't trip us up."""
    df = df.copy()
    # Make sure crisis_flag is bool; some parquet readers yield object.
    df["crisis_flag"] = df["crisis_flag"].fillna(False).astype(bool)
    return df


def train(config_path: Path) -> Path:
    cfg = TrainConfig.from_yaml(config_path)
    _set_all_seeds(cfg.seed)

    print(f"[train] config: {config_path}")
    print(f"[train] encoder: {cfg.encoder_name}, max_length: {cfg.max_length}")
    print(f"[train] device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.encoder_name, cache_dir=cfg.cache_dir
    )
    emotion_to_id = {name: i for i, name in enumerate(cfg.emotions)}
    model = MultiTaskClassifier(
        encoder_name=cfg.encoder_name,
        num_emotions=len(cfg.emotions),
        cache_dir=cfg.cache_dir,
    )

    # Data
    train_df = _to_parquet_in_memory(pd.read_parquet(cfg.train_file))
    val_df = _to_parquet_in_memory(pd.read_parquet(cfg.val_file))
    train_df = _subsample(train_df, cfg.max_train_samples, cfg.seed)
    val_df = _subsample(val_df, cfg.max_eval_samples, cfg.seed)

    print(f"[train] train rows: {len(train_df)}  val rows: {len(val_df)}")
    print(f"[train] emotion label space: {len(cfg.emotions)} classes")

    train_ds = MindcareDataset(
        train_df, tokenizer, cfg.max_length, emotion_to_id, cfg.emotion_aliases
    )
    val_ds = MindcareDataset(
        val_df, tokenizer, cfg.max_length, emotion_to_id, cfg.emotion_aliases
    )

    collator = MultiTaskCollator(tokenizer=tokenizer)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=cfg.eval_every_steps,
        save_strategy="no",                 # we save explicitly below
        logging_steps=100,
        report_to="none",
        seed=cfg.seed,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
        # Don't auto-load best model — we save once at the end.
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"[train] final eval: {metrics}")

    # Save model + label maps in a single directory so inference is one path.
    save_dir = output_dir / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    # Save base config and label map definitions explicitly.
    from transformers import AutoConfig
    base_config = AutoConfig.from_pretrained(cfg.encoder_name, cache_dir=cfg.cache_dir)
    base_config.save_pretrained(str(save_dir))
    
    import shutil
    from transformers.utils import cached_file
    for fname in ("vocab.txt", "special_tokens_map.json"):
        try:
            cached_path = cached_file(
                cfg.encoder_name, fname, cache_dir=cfg.cache_dir
            )
            shutil.copy2(cached_path, save_dir / fname)
        except Exception:
            pass

    label_map = {
        "base_encoder_name": cfg.encoder_name,  # Added this field explicitly to support future fallbacks
        "emotions": cfg.emotions,
        "emotion_aliases": cfg.emotion_aliases,
        "emotion_to_id": emotion_to_id,
    }
    (save_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2, sort_keys=True), encoding="utf-8"
    )
    (save_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[train] saved model + label map -> {save_dir}")
    return save_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the mindcare-ai multi-task classifier.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to the YAML training config.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(0 if train(args.config) else 1)