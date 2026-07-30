# Vertex AI batch explainability (HELOC)

Runnable companion to the [Medium tutorial](https://github.com/04pallav/gcp-ml-tutorials/blob/main/tutorials/vertex-ai-batch-explainability.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp config.example.json config.local.json   # set project + bucket
python train_model.py                      # optional local check
python prepare_bq_input.py                 # load instances.json → BigQuery
python run_batch_explain.py                # train, upload, batch explain
```

## Files

| File | Purpose |
|---|---|
| `config.example.json` | Project, bucket, BQ tables |
| `train_model.py` | Train sklearn model locally |
| `prepare_bq_input.py` | Load `instances.json` into BigQuery |
| `run_batch_explain.py` | Upload model + `batch_predict(generate_explanation=True)` |
| `instances.json` | Two sample applicants |
| `feature_names.json` | Feature list for explanation metadata |

`config.local.json` is gitignored — never commit your project IDs or bucket names.
