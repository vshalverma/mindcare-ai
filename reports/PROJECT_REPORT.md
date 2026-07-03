# Mindcare-AI — Project Report

> **Status:** Production-ready baseline. All components verified by 60-test
> pytest suite, training run on real hardware (GTX 1650, 4 GB VRAM).
> **Repo:** `github.com/vv8282013-arch/mindcare-ai` (push pending GitHub PAT).

---

## 1. What is it?

A self-contained **mental-health chatbot** that takes a user's free-text
message and:

1. **Predicts the emotion** (28 classes from the GoEmotions taxonomy).
2. **Predicts a crisis-risk score** (binary: at-risk vs. safe).
3. **Fires a keyword safety gate** as a second opinion for crisis phrases.
4. **Picks a template reply** keyed by the predicted emotion.
5. **Surfaces a crisis-safety banner** with hotline numbers whenever the
   model OR the keyword gate flags the input.

Crucially, **no large language model is queried at inference time**. The
text generator is a hand-written template picker that cannot hallucinate
medical advice. This is a deliberate safety design choice.

---

## 2. Why was it built?

A final-year AIML capstone-style project. The driving constraints were:

- **Hardware budget:** GTX 1650, 4 GB VRAM (no cloud GPU). Anything
  larger than ~70 M params would not fit under fp16 + batch size 16.
- **Safety budget:** a chatbot in this domain must be conservative on
  crisis detection. A missed suicide cue is a *catastrophic* failure.
  Template replies ensure the bot never invents advice.
- **No API budget:** running an LLM (OpenAI / Anthropic) in the loop
  would defeat the point of a self-contained demo and create privacy
  exposure for sensitive user text.

---

## 3. Architecture

```
┌────────────────────┐
│  Streamlit UI      │   src/app/streamlit_app.py
│  (chat box + UI)   │   Chat history, predicted emotion, crisis banner
└─────────┬──────────┘
          │ user text
          ▼
┌────────────────────┐
│  ChatEngine        │   src/inference/chat_engine.py
│  (orchestrator)    │   wires the three signals below
└─────┬───┬───┬──────┘
      │   │   │
      │   │   └─► src/inference/loader.py
      │   │       (loads DistilBERT + 2 heads + label_map.json)
      │   │            │
      │   │            ▼
      │   │     ┌──────────────┐
      │   │     │ DistilBERT   │   ~66 M params
      │   │     │ multi-task   │   encoder + 2 classification heads
      │   │     └──────┬───────┘
      │   │            │  emotion_label (28-way)
      │   │            │  crisis_prob   (binary)
      │   │            ▼
      │   │     models/checkpoints/final/
      │   │       model.safetensors (LFS, 254 MB)
      │   │       label_map.json
      │   │
      │   └─► src/inference/crisis_keywords.py
      │        (fast substring pass — second line of defence)
      │
      └─► picks a template response from
           src/inference/chat_engine.py RESPONSE_TEMPLATES

Conservative-OR rule:
  crisis = (model_prob ≥ 0.5) OR (keyword_gate == 1.0)
```

### 3.1 Multi-task classifier

A single DistilBERT encoder with two heads on top of the `[CLS]`
representation:

- **Emotion head** — 28-class softmax over the GoEmotions taxonomy.
- **Crisis head** — sigmoid over the binary "is this person in crisis" task.

**Why multi-task?** Joint training produces a richer encoder. The
crisis head is safety-critical; the emotion head gives the encoder
broad coverage of emotional language. They share parameters and
regularize each other. The cost is one extra dense layer, well within
the 4 GB VRAM budget.

### 3.2 Keyword safety gate

A pure-Python substring matcher in `src/inference/crisis_keywords.py`
covering phrases like "kill myself", "end my life", "no reason to live",
self-harm references, etc. The 60-test suite includes 30 keyword cases
(positive, negative, case-varied, multi-word, and edge cases like
"I killed the spider" which must NOT fire).

### 3.3 Template response engine

`RESPONSE_TEMPLATES` in `chat_engine.py` maps each of the 28 emotions
to a small set of hand-written empathetic replies. The template picker
also injects a crisis-resources banner when the conservative-OR rule
fires. This is intentionally not an LLM so the bot:

- Cannot hallucinate medical advice.
- Has predictable, auditable outputs.
- Runs in microseconds with no API cost.

---

## 4. Data pipeline

A 5-stage CLI: `python -m src.data_pipeline.run_pipeline --stage all`

| Stage | Module | Purpose |
|---|---|---|
| 1. Download | `download.py` | Fetch GoEmotions, EmpatheticDialogues, and a suicide/depression corpus from HuggingFace. Writes a per-dataset `manifest.json` with row counts and SHA-256. |
| 2. Normalize | `normalize.py` | Per-dataset raw → unified schema: `{text, emotion, source, is_crisis, split}`. Handles 32-label EmpatheticDialogues → 28-label GoEmotions mapping via `configs/training.yaml::emotion_aliases`. |
| 3. Crisis-flag | `crisis.py` | Tier-2 keyword flag pass: any text that matches a high-risk substring gets `is_crisis = 1`. Acts as weak supervision for the crisis head. |
| 4. Split | `split.py` | Re-splits with only a `train` split, then derives `val` / `test` deterministically. **Known bug fixed:** an earlier version re-bound a local variable in `_resplit()` and never mutated the parts dict, leaving all 232k `suicide_depression` rows in `train`. Fixed by building an `out_parts` list and setting `split` on each part before concat. |
| 5. Validate | `validate.py` | Writes `reports/data_quality.md` with row counts, class balance, missingness, length statistics. |

**Final dataset size:** 248 k train / 31 k val / 31 k test rows.

---

## 5. Training

```
model:        distilbert-base-uncased    (~66 M params)
batch size:   16 (fp16)
max_length:   96 tokens
optimizer:    AdamW, lr 5e-5, wd 0.01, warmup 6%
epochs:       1
hardware:     GTX 1650, 4 GB VRAM
time:         ~19 h on the full 248 k rows
```

`configs/training.yaml` is the single source of truth — every
hyperparameter lives there. `eval_every_steps: 999999` skips mid-training
evals to save time; the final eval fires at the end of the epoch.

---

## 6. Results

From `reports/eval_per_class.md` (1-epoch, full ~248 k-row training
run, evaluated on 31 389 val rows).

### 6.1 Crisis head (binary — safety-critical)

| Metric | Value |
|---|---:|
| Accuracy | **97.6 %** |
| Precision (positive = crisis) | 96.2 % |
| **Recall (positive = crisis)** | **97.4 %** |
| **F1** | **96.8 %** |
| Support (crisis positive) | 11 795 |
| Support (crisis negative) | 19 594 |

97.4 % recall is the headline number — a real distress message is
almost never missed by the model. The keyword safety gate is the
second line of defence and the two fire in conservative-OR mode.

### 6.2 Emotion head (28-class)

| Metric | Value |
|---|---:|
| Overall accuracy (on scored rows) | **63.4 %** |
| Macro precision | 56.4 % |
| Macro recall | 49.8 % |
| Macro F1 | 51.1 % |
| Weighted F1 | 61.7 % |

The 28-way classifier is doing real work but is hurt by class
imbalance and confusion between semantically similar emotions.
Notable patterns from the top-10 confusions:

- `approval → neutral` (146) and `disapproval → neutral` (131):
  evaluative but low-arousal emotions get pushed to the majority class.
- `annoyance ↔ anger` (69 each way): unsurprising — the two share
  surface vocabulary.
- Three classes (`nervousness`, `relief`, `grief`) have F1 = 0
  because they have ≤8 examples each in the val set.

A second pass with class weighting or focal loss would close the
emotion gap, but the *crisis* signal is what this project exists for
and that is locked in.

---

## 7. Tech stack

| Layer | Tool | Why |
|---|---|---|
| Model | `distilbert-base-uncased` (HuggingFace Transformers) | Fits in 4 GB VRAM with fp16. `prajjwal1/bert-mini` is smaller but its tokenizer needs `sentencepiece` which wasn't installed in the env. |
| Training | `transformers.Trainer` (PyTorch backend) | Built-in fp16, weight decay, warmup, eval, checkpoint management. |
| Numerical / data | `numpy`, `pandas`, `pyarrow`, `scikit-learn` | Parquet I/O, train/val/test split, classification metrics. |
| App | `streamlit` | One-file UI, hot-reload, native chat primitives. |
| Data | `datasets` (HF) | Streaming download for the GoEmotions / EmpatheticDialogues / suicide-depression corpora. |
| Tests | `pytest` | 60 tests, runs on CPU in ~7 s. Chat engine tests use a stub classifier so no checkpoint or GPU is needed. |
| CI | GitHub Actions, Python 3.14, CPU-only torch wheel | Install time stays low (~30 s with warm cache). |
| LFS | `git-lfs 3.7.1` | For the 254 MB `model.safetensors` (over GitHub's 100 MB cap). |

---

## 8. Tests & CI

60 tests, all passing locally in 6.68 s:

- `tests/test_chat_engine.py` (14) — crisis truth table (8 cases), emotion
  routing, empty-input fallback, response text non-empty for normal input.
- `tests/test_keyword_gate.py` (30) — positive phrases, negative phrases,
  case variations, multi-word, and the "I killed the spider" negative case.
- `tests/test_label_map.py` (8) — schema invariants on `label_map.json`.
- `tests/test_loader.py` (8) — classifier loader behaviour.

`.github/workflows/ci.yml` runs the same `pytest` on every push / PR.

---

## 9. Limitations & honest weaknesses

- **Templates, not generation.** Replies are rigid and a user can
  detect the pattern. This is by design (safety > fluency) but is the
  most-cited weakness in user tests.
- **English only.** Tokenizer is `bert-base-uncased` English; the
  Hinglish / Hindi mix that real Indian users send is poorly handled.
- **No conversation memory.** Each turn is classified independently;
  context across turns is not modeled.
- **Crisis resources are US-default.** `CRISIS_RESOURCES` in
  `chat_engine.py` is hard-coded; needs localization for any
  deployment outside the US.
- **Emotion head undersized.** Macro F1 of 51 % on 28 classes is
  below SOTA. Class imbalance hurts the long-tail classes to F1 = 0.
- **No RLHF or red-teaming.** The crisis gate is rule-based plus a
  supervised classifier. No adversarial testing of the safety path.

---

## 10. Future work (interview answers)

If asked "what would you do next?" you can confidently say any of:

1. **Scale training to multi-epoch** with class weighting and focal loss
   to lift emotion macro F1.
2. **Add Hinglish / Hindi support** with `xlm-roberta-base` or
   `ai4bharat/IndicBERTv2-MLM-only`.
3. **Wire a small open LLM** (Phi-3-mini, Gemma-2B) behind a
   retrieval-augmented template library for less rigid replies —
   still no API, still on-device.
4. **Adversarial red-team the safety gate.** Build a test set of
   obfuscated crisis phrases ("k1ll mys3lf", "I w@k3 up w4nt1ng 2
   d13") and measure recall.
5. **Move to Hugging Face Hub** for model hosting + Spaces demo
   (gives the project a public URL beyond the GitHub repo).
6. **Add evaluation on the test set** (currently only val is scored).
7. **Conversation memory** with a sliding window and an "emotion
   trajectory" feature (was the user trending toward fear/grief
   over the last 5 turns?).

---

## 11. How to reproduce

```bash
git clone https://github.com/vv8282013-arch/mindcare-ai
cd mindcare-ai
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt
git lfs pull                          # downloads model.safetensors
python -m src.data_pipeline.run_pipeline --stage all
venv\Scripts\python -m src.models.train --config configs/training.yaml
venv\Scripts\python -m streamlit run src/app/streamlit_app.py
```

Or, to skip retraining and just chat:

```bash
pip install -r requirements.txt
git lfs pull
venv\Scripts\python -m streamlit run src/app/streamlit_app.py
```

---

## 12. Key files

| Path | Lines | Purpose |
|---|---|---|
| `src/data_pipeline/run_pipeline.py` | CLI | 5-stage data pipeline |
| `src/data_pipeline/normalize.py` | per-source → unified schema | |
| `src/data_pipeline/crisis.py` | keyword weak-supervision | |
| `src/data_pipeline/split.py` | re-split with only `train` | bug-fixed during project |
| `src/data_pipeline/validate.py` | data quality report | |
| `src/models/train.py` | multi-task DistilBERT training | |
| `src/inference/loader.py` | checkpoint + label_map loader | |
| `src/inference/chat_engine.py` | `ChatEngine.reply()` + `RESPONSE_TEMPLATES` + `CRISIS_RESOURCES` | |
| `src/inference/crisis_keywords.py` | substring safety gate | |
| `src/app/streamlit_app.py` | UI | |
| `configs/training.yaml` | single source of truth for hyperparams | |
| `tests/*.py` | 60 pytest cases | |
| `.github/workflows/ci.yml` | CI | |
| `reports/eval_per_class.md` | accuracy report | |
| `reports/data_quality.md` | data stats | |
| `models/checkpoints/final/model.safetensors` | 254 MB, Git LFS | |
| `models/checkpoints/final/label_map.json` | emotion + alias table | |
