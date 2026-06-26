# mindcare-ai

A small, **self-contained** mental-health chatbot demo. Built to run on
consumer hardware (trained and tested on a GTX 1650, 4 GB VRAM).

## What it does

Given a user message, `mindcare-ai`:

1. **Classifies** the text with a DistilBERT multi-task model that outputs
   - `emotion_label` (28-class, GoEmotions taxonomy)
   - `crisis_prob` (binary, suicide/depression classifier)
2. **Runs a keyword safety gate** on top — a fast substring pass that
   catches high-risk phrases the model might miss.
3. **Picks a template response** keyed by the predicted emotion.
4. **Surfaces a crisis-safety banner** with hotline numbers whenever the
   model OR the keyword gate flags the input (conservative OR).

The classifier is **multi-task** because the encoder learns better
emotional-language representations when it sees both emotion labels
(GoEmotions + EmpatheticDialogues) and crisis labels
(suicide_depression corpus).

## Pipeline

```
data/raw/                                (HuggingFace CSVs, gitignored)
  ├── go_emotions/
  ├── empathetic_dialogues/
  └── suicide_depression/
        │
        ▼  src/data_pipeline/download.py
data/raw/<slug>/manifest.json
        │
        ▼  src/data_pipeline/normalize.py
        ▼  src/data_pipeline/crisis.py
data/processed/unified_{train,val,test}.parquet
        │
        ▼  src/models/train.py
models/checkpoints/final/                (encoder + heads + label map)
        │
        ▼  src/inference/chat_engine.py
        ▼  src/app/streamlit_app.py
```

Run the whole data pipeline with:

```bash
python -m src.data_pipeline.run_pipeline --stage all
```

Re-generate the data-quality report only:

```bash
python -m src.data_pipeline.run_pipeline --stage validate
```

## Training

```bash
venv/Scripts/python -m src.models.train --config configs/training.yaml
```

The YAML config controls everything: encoder, batch size, max length,
learning rate, mixed precision, train/eval sample caps.

First-run config trains for 1 epoch on 20,000 stratified samples
(~25 min on a GTX 1650). To use the full ~248k rows, set
`max_train_samples: 0`.

## Chat UI

```bash
venv/Scripts/python -m streamlit run src/app/streamlit_app.py
```

Opens at `http://localhost:8501`.

## Safety notes

- This is a **demo**, not a clinical product. The response generator is
  template-based on purpose so it cannot hallucinate medical advice.
- Crisis handling is **conservative** — a banner appears if EITHER the
  model or the keyword gate fires. False positives are acceptable; false
  negatives are not.
- Hotlines hard-coded in `src/inference/chat_engine.py` (`CRISIS_RESOURCES`).
  Update them as needed for your deployment region.

## Files of interest

| Path                                            | Purpose                                    |
|-------------------------------------------------|--------------------------------------------|
| `configs/training.yaml`                         | Single source of truth for training config |
| `src/data_pipeline/run_pipeline.py`             | CLI entrypoint for data pipeline           |
| `src/data_pipeline/normalize.py`                | Per-dataset raw → unified schema           |
| `src/data_pipeline/crisis.py`                   | Tier-2 keyword flag pass                   |
| `src/data_pipeline/split.py`                    | Re-split datasets with only a train split  |
| `src/data_pipeline/validate.py`                 | Writes `reports/data_quality.md`           |
| `src/models/train.py`                           | Multi-task classifier training             |
| `src/inference/chat_engine.py`                  | ChatEngine + safety gate                   |
| `src/app/streamlit_app.py`                      | Streamlit UI                               |
| `reports/data_quality.md`                       | Auto-generated data stats                  |
| `models/checkpoints/final/training_metrics.json`| Final eval metrics from training           |