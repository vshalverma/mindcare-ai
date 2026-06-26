"""Per-emotion + per-crisis classification report.

Loads the trained checkpoint, runs inference on the val parquet, and
writes `reports/eval_per_class.md` with precision/recall/F1/support per
class and a confusion matrix summary.

Run from the project root:
    venv/Scripts/python eval_per_class.py

Why a separate script? The training report (`training_metrics.json`)
collapses everything to overall accuracy. For a 28-class classifier
that's almost useless — we need per-class breakdown to see *which*
emotions the model is weak on, so we know where to invest next (more
data, different head, or simply documenting it as a known limitation).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import MultiTaskClassifier  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"
VAL_PARQUET = PROJECT_ROOT / "data" / "processed" / "unified_val.parquet"
CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "final"
OUT_PATH = PROJECT_ROOT / "reports" / "eval_per_class.md"


# ---------------------------------------------------------------------------
# Reproduce the trainer's label-construction logic
# ---------------------------------------------------------------------------

def build_labels(df: pd.DataFrame, emotion_to_id: dict, emotion_aliases: dict):
    """Return (emotion_labels, crisis_labels) the same way the trainer does.

    Rows with no canonical emotion get -100 (the trainer's sentinel for
    "ignore in loss"). We use the same mask to ignore them in metrics.
    """
    emotion_labels = []
    crisis_labels = []
    for raw_emo, raw_crisis in zip(df["emotion_label"].tolist(), df["crisis_flag"].tolist()):
        if raw_emo is None or (isinstance(raw_emo, float) and pd.isna(raw_emo)):
            e = None
        else:
            s = str(raw_emo).strip().lower()
            e = emotion_aliases.get(s, s) if s else None
        emotion_labels.append(emotion_to_id.get(e, -100) if e is not None else -100)
        crisis_labels.append(1.0 if bool(raw_crisis) else 0.0)
    return np.array(emotion_labels, dtype=np.int64), np.array(crisis_labels, dtype=np.float32)


# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_inference(model, tokenizer, texts: list[str], device: str, batch_size: int = 64):
    emo_logits_all = []
    crisis_logit_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True, max_length=128, return_tensors="pt"
        ).to(device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        emo_logits_all.append(out["emotion_logits"].float().cpu().numpy())
        crisis_logit_all.append(out["crisis_logit"].float().cpu().numpy())
    emo_logits = np.concatenate(emo_logits_all, axis=0)
    crisis_logit = np.concatenate(crisis_logit_all, axis=0)
    emo_preds = np.argmax(emo_logits, axis=-1)
    crisis_prob = 1.0 / (1.0 + np.exp(-crisis_logit))
    crisis_pred = (crisis_prob >= 0.5).astype(np.int64)
    return emo_preds, emo_logits, crisis_pred, crisis_prob


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

def per_class_prf(y_true, y_pred, num_classes: int) -> list[dict]:
    """Return one dict per class with precision/recall/F1/support.

    Macro and weighted averages are computed at the call site.
    """
    rows = []
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "class": c,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
    return rows


# ---------------------------------------------------------------------------
# Confusion matrix (top confused pairs)
# ---------------------------------------------------------------------------

def top_confusions(y_true, y_pred, id_to_name: dict, k: int = 10) -> list[tuple[str, str, int]]:
    pairs: dict[tuple[int, int], int] = {}
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t != p and t != -100:
            pairs[(t, p)] = pairs.get((t, p), 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1])[:k]
    return [(id_to_name.get(t, str(t)), id_to_name.get(p, str(p)), c) for (t, p), c in ranked]


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------

def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_report(
    emo_rows, crisis: dict, confusions, id_to_name, total_scored, total_rows, out_path: Path
) -> None:
    lines: list[str] = []
    lines.append("# Per-class evaluation report")
    lines.append("")
    lines.append("Generated by `eval_per_class.py` on the unified validation set.")
    lines.append("Scores are from a single checkpoint (`models/checkpoints/final`), 1-epoch, full ~248k-row training run.")
    lines.append("")
    lines.append(f"- Val rows: **{total_rows:,}**")
    lines.append(f"- Rows with a canonical emotion label: **{total_scored:,}** "
                 f"({fmt_pct(total_scored / total_rows)})")
    lines.append("")

    # ---- Crisis head ----
    lines.append("## Crisis head (binary)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| accuracy | {fmt_pct(crisis['accuracy'])} |")
    lines.append(f"| precision (positive = crisis) | {fmt_pct(crisis['precision'])} |")
    lines.append(f"| recall (positive = crisis) | {fmt_pct(crisis['recall'])} |")
    lines.append(f"| F1 (positive = crisis) | {fmt_pct(crisis['f1'])} |")
    lines.append(f"| support (crisis positive) | {crisis['support_pos']:,} |")
    lines.append(f"| support (crisis negative) | {crisis['support_neg']:,} |")
    lines.append("")
    lines.append("> The crisis head is the safety-critical head: low recall here would")
    lines.append("> mean missed suicides. The keyword safety gate in the chat engine is")
    lines.append("> a second line of defence, but the model itself should still score well.")
    lines.append("")

    # ---- Emotion head ----
    lines.append("## Emotion head (28-class)")
    lines.append("")
    supports = np.array([r["support"] for r in emo_rows])
    f1s = np.array([r["f1"] for r in emo_rows])
    precs = np.array([r["precision"] for r in emo_rows])
    recs = np.array([r["recall"] for r in emo_rows])
    macro_f1 = float(f1s.mean())
    weighted_f1 = float((f1s * supports).sum() / max(supports.sum(), 1))
    macro_p = float(precs.mean())
    macro_r = float(recs.mean())
    overall_acc = float(
        sum(r["tp"] for r in emo_rows) / max(sum(r["support"] for r in emo_rows), 1)
    )

    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| overall accuracy (on scored rows) | {fmt_pct(overall_acc)} |")
    lines.append(f"| macro precision | {fmt_pct(macro_p)} |")
    lines.append(f"| macro recall | {fmt_pct(macro_r)} |")
    lines.append(f"| macro F1 | {fmt_pct(macro_f1)} |")
    lines.append(f"| weighted F1 (by support) | {fmt_pct(weighted_f1)} |")
    lines.append("")

    lines.append("### Per-emotion scores")
    lines.append("")
    lines.append("Rows sorted by support descending. F1 = 0.0% with `*` means the model never predicted that class on scored rows.")
    lines.append("")
    lines.append("| emotion | precision | recall | F1 | support |")
    lines.append("|---|---:|---:|---:|---:|")
    # Sort by support desc
    for r in sorted(emo_rows, key=lambda r: -r["support"]):
        name = id_to_name.get(r["class"], str(r["class"]))
        f1_str = "0.0%*" if r["f1"] == 0.0 and r["support"] > 0 else fmt_pct(r["f1"])
        lines.append(
            f"| {name} | {fmt_pct(r['precision'])} | {fmt_pct(r['recall'])} | {f1_str} | {r['support']:,} |"
        )
    lines.append("")

    # ---- Top confusions ----
    lines.append("## Top 10 emotion confusions")
    lines.append("")
    lines.append("Format: `true → predicted` (count).")
    lines.append("")
    if confusions:
        lines.append("| true | predicted | count |")
        lines.append("|---|---|---:|")
        for t, p, c in confusions:
            lines.append(f"| {t} | {p} | {c} |")
    else:
        lines.append("_No confusions on scored rows._")
    lines.append("")

    # ---- Footer ----
    lines.append("---")
    lines.append("")
    lines.append("Reproduce: `venv/Scripts/python eval_per_class.py`")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"[eval] loading config from {CONFIG_PATH}")
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    emotion_to_id = {name: i for i, name in enumerate(cfg["emotions"])}
    emotion_aliases = cfg.get("emotion_aliases", {})
    id_to_name = {i: name for name, i in emotion_to_id.items()}

    print(f"[eval] loading val parquet from {VAL_PARQUET}")
    df = pd.read_parquet(VAL_PARQUET)
    print(f"[eval]   {len(df):,} rows")

    y_emo, y_crisis = build_labels(df, emotion_to_id, emotion_aliases)
    scored_mask = y_emo != -100
    print(f"[eval]   {int(scored_mask.sum()):,} rows have a canonical emotion label")
    print(f"[eval]   {int(y_crisis.sum()):,} rows have crisis_flag=True")

    print(f"[eval] loading model from {CKPT_DIR}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT_DIR))
    model = MultiTaskClassifier(encoder_name=str(CKPT_DIR), num_emotions=len(emotion_to_id))
    from safetensors.torch import load_file
    state_dict_file = CKPT_DIR / "model.safetensors"
    model.load_state_dict(load_file(str(state_dict_file)))
    model.to(device).eval()
    print(f"[eval]   on device: {device}")

    print(f"[eval] running inference on {len(df):,} rows")
    texts = df["text"].astype(str).tolist()
    emo_preds, _, crisis_pred, _ = run_inference(model, tokenizer, texts, device)

    # ---- Crisis metrics (binary) ----
    tp = int(((crisis_pred == 1) & (y_crisis == 1)).sum())
    fp = int(((crisis_pred == 1) & (y_crisis == 0)).sum())
    fn = int(((crisis_pred == 0) & (y_crisis == 1)).sum())
    tn = int(((crisis_pred == 0) & (y_crisis == 0)).sum())
    crisis = {
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": tp / max(tp + 0.5 * (fp + fn), 1),
        "support_pos": int(y_crisis.sum()),
        "support_neg": int((y_crisis == 0).sum()),
    }

    # ---- Emotion metrics (28-class, restricted to scored rows) ----
    emo_rows = per_class_prf(y_emo[scored_mask], emo_preds[scored_mask], num_classes=len(emotion_to_id))
    confusions = top_confusions(y_emo[scored_mask], emo_preds[scored_mask], id_to_name, k=10)

    write_report(
        emo_rows=emo_rows,
        crisis=crisis,
        confusions=confusions,
        id_to_name=id_to_name,
        total_scored=int(scored_mask.sum()),
        total_rows=len(df),
        out_path=OUT_PATH,
    )

    # Quick stdout summary
    print()
    print("[eval] crisis head  acc={:.3f}  prec={:.3f}  rec={:.3f}  f1={:.3f}".format(
        crisis["accuracy"], crisis["precision"], crisis["recall"], crisis["f1"]
    ))
    macro_f1 = float(np.mean([r["f1"] for r in emo_rows]))
    overall_acc = sum(r["tp"] for r in emo_rows) / max(sum(r["support"] for r in emo_rows), 1)
    print(f"[eval] emotion head acc={overall_acc:.3f}  macro_f1={macro_f1:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
