# ☁️Vertex AI Batch Explainability: Explain Credit-Risk Predictions with BigQuery and scikit-learn 🔍

**By Pallav Anand**

Find the code on GitHub: **[github.com/04pallav/gcp-ml-staging](https://github.com/04pallav/gcp-ml-staging)** — tutorial code lives in [`code/vertex-batch-explainability/`](https://github.com/04pallav/gcp-ml-staging/tree/main/code/vertex-batch-explainability).

---

This tutorial walks through **Vertex AI batch explainability** on a real credit-risk use case. You train a scikit-learn classifier on the **FICO HELOC** dataset, register it on Vertex with explanation metadata, and run **`Model.batch_predict(..., generate_explanation=True)`** so every row gets a prediction **and** per-feature attributions — in one batch job, reading from **BigQuery** and writing results back to **BigQuery**.

That is the pattern you want when explanations must scale: compliance reviews, model monitoring, or offline analysis across thousands of applicants — without calling `endpoint.explain()` row by row.

**Stack:**

- 🗃️ **Cloud Storage** — `model.joblib` and `feature_names.json`
- 📊 **BigQuery** — batch input table and explanation output table
- 🤖 **Vertex AI Model** — sklearn classifier with `ExplanationMetadata` at upload
- 🔍 **Vertex Explainable AI** — explanations attached to batch prediction output
- 📈 **Looker Studio** (optional) — dashboards on the output table

> This tutorial uses **Vertex AI batch prediction with explanations**. It is not the open-source `shap` package, not Beam/Dataflow custom explainers, and not online-only `endpoint.explain()`.

Clone the code repo, copy `config.example.json` to `config.local.json`, and follow the steps below.

---

## 🎯 What is batch explainability?

**Online explain:** deploy a model to an endpoint, call `explain()` per request. Fine for a few rows; expensive at scale.

**Batch explain:** register explanation metadata once, then run `batch_predict` with `generate_explanation=True`. Vertex scores every row and returns **feature attributions** in the same job output.

Explanations are computed **at batch prediction time**, not during training. You declare which inputs and outputs are explainable when uploading the model; Vertex handles the rest when the batch job runs.

---

## 📋 The HELOC dataset

We use the **Home Equity Line of Credit (HELOC)** dataset from OpenML — a classic binary credit-risk benchmark:

- **22 numeric features** (payment history, utilization, inquiries, etc.)
- **Label:** good/bad credit risk
- **Sample batch rows:** `instances.json` ships with two applicants for a cheap smoke test

In production, your BigQuery input table would hold the same feature columns for the population you need to explain.

---

## 🏗️ Architecture

```
instances.json
      │
      ├─ train_heloc_model.py ──► model.joblib + feature_names.json ──► GCS models/
      │
      └─ prepare_bq_input.py ──► BigQuery input table (heloc_batch_input)
                                      │
              Model.upload(explanation_metadata=…)
                                      │
              Model.batch_predict(generate_explanation=True)
                                      │
                              BigQuery output table (heloc_batch_explanations)
                                      │
                         BigQuery Console / Looker Studio / visualize_attributions.py
```

**GCS** stores only model artifacts:

```
gs://YOUR_BUCKET/gcp-ml-staging/vertex-batch-explain/
└── models/
    ├── model.joblib
    └── feature_names.json
```

**BigQuery** holds batch I/O (configurable in `config.local.json`):

| Table | Purpose |
|---|---|
| `YOUR_PROJECT.ml_explainability.heloc_batch_input` | One FLOAT column per HELOC feature |
| `YOUR_PROJECT.ml_explainability.heloc_batch_explanations` | Predictions + explanations (written by Vertex) |

---

## 📦 Repo layout

| File | Purpose |
|---|---|
| `config.example.json` | Template — project, bucket, BQ tables, machine type |
| `config.local.json` | Your values (gitignored) |
| `instances.json` | Two sample applicants |
| `feature_names.json` | 22 feature names for explanation metadata |
| `train_heloc_model.py` | Train sklearn pipeline → `model.joblib` |
| `prepare_bq_input.py` | `instances.json` → BigQuery input table |
| `run_batch_explainability.py` | Main entry — local smoke test or full Vertex run |
| `vertex_batch_explain.py` | Upload model + `batch_predict` helpers |
| `visualize_attributions.py` | Feature attribution bar chart (demo or JSONL) |
| `prepare_batch_input.py` | Optional GCS JSONL path |
| `run_explainability.py` | Optional online `endpoint.explain()` demo |

Everything is **config-driven**: one JSON file (plus optional env vars) controls project, bucket, BQ tables, machine type, and batch settings. No project IDs hardcoded in the scripts.

---

## 👩‍💻 Step 0: Smoke test (no Vertex calls)

Before enabling APIs or spending quota, verify training and instance validation locally:

```bash
git clone https://github.com/04pallav/gcp-ml-staging.git
cd gcp-ml-staging/code/vertex-batch-explainability
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt

python run_batch_explainability.py --local-only
python -m unittest tests.test_local -v
python visualize_attributions.py --demo
```

What this does:

1. Trains a small HELOC model into `artifacts/models/`
2. Validates that `instances.json` has all 22 features
3. Writes `artifacts/batch-input/instances.jsonl` (local only — not used in the BigQuery path)
4. Saves a demo attribution chart to `artifacts/attributions-demo.png`

If all tests pass, your feature schema and training pipeline are ready for Vertex.

---

## 🗃️ Step 1: Configure project, bucket, and auth

```bash
cp config.example.json config.local.json
```

Edit `config.local.json`:

```json
{
  "project_id": "YOUR_GCP_PROJECT_ID",
  "region": "us-central1",
  "staging_bucket_uri": "gs://YOUR_BUCKET/gcp-ml-staging/vertex-batch-explain",
  "display_name": "heloc-batch-explain",
  "job_name": "heloc-batch-explain-job",
  "input_format": "bigquery",
  "bq_dataset": "ml_explainability",
  "bq_input_table": "heloc_batch_input",
  "bq_output_table": "heloc_batch_explanations",
  "serving_container": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest",
  "machine_type": "n2-standard-4",
  "starting_replica_count": 1,
  "max_replica_count": 1,
  "batch_size": 16
}
```

Environment variable overrides (optional):

| Variable | Config field |
|---|---|
| `VERTEX_PROJECT_ID` | `project_id` |
| `VERTEX_REGION` | `region` |
| `GCS_STAGING_BUCKET_URI` | `staging_bucket_uri` |
| `BQ_DATASET` | `bq_dataset` |

Enable APIs and install Vertex dependencies:

```bash
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud services enable aiplatform.googleapis.com storage.googleapis.com bigquery.googleapis.com

pip install -r requirements.txt
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account.json
```

Your service account needs at least:

- `roles/aiplatform.user` — upload models, run batch prediction
- `roles/storage.objectAdmin` on the staging bucket — upload model artifacts
- `roles/bigquery.dataEditor` — load input table; Vertex writes output table

❗ **scikit-learn version:** train with sklearn **1.6.x** (pinned in `requirements-local.txt`) and serve with `sklearn-cpu.1-6`. Training on sklearn 1.9 and serving on an older container causes deserialization errors.

❗ **Same region:** keep bucket, Vertex jobs, and BigQuery in the same region (e.g. `us-central1` / `US`).

---

## 🐝 Step 2: Train the model

Train locally — Vertex serves this exact `model.joblib`:

```bash
python train_heloc_model.py --output-dir artifacts/models
```

The pipeline is: **median imputer → standard scaler → logistic regression** (balanced class weights).

Outputs:

- `artifacts/models/model.joblib`
- `artifacts/models/feature_names.json` — feature list + `explanation_output_key: "probability"`

For a faster dev run:

```bash
python train_heloc_model.py --output-dir artifacts/models --row-limit 2000
```

`run_batch_explainability.py` can also train on the fly when you do not pass `--model-id`; training locally first lets you inspect the artifact before upload.

---

## 📊 Step 3: Load batch input into BigQuery

`instances.json` has two sample rows. Load them into the input table:

```bash
python prepare_bq_input.py --config config.local.json
```

The script:

1. Creates dataset `ml_explainability` if it does not exist
2. Writes table `heloc_batch_input` with **one FLOAT column per feature**
3. Truncates and reloads on each run (`WRITE_TRUNCATE`)

Confirm in BigQuery Console:

```sql
SELECT *
FROM `YOUR_GCP_PROJECT_ID.ml_explainability.heloc_batch_input`
LIMIT 10;
```

❗ Column names must match `feature_names.json` exactly — they become the `inputs` map in Vertex `ExplanationMetadata`.

In production, you would skip `instances.json` and point the batch job at a table your ETL pipeline already maintains.

---

## 🔍 Step 4: Register explainability metadata

When uploading the model, `vertex_batch_explain.py` attaches Vertex explanation config:

```python
explanation_parameters = aiplatform.explain.ExplanationParameters({
    "sampled_shapley_attribution": {"path_count": 10},
})
explanation_metadata = aiplatform.explain.ExplanationMetadata(
    inputs={name: {} for name in feature_names},
    outputs={"probability": {}},
)
```

| Field | Meaning |
|---|---|
| `inputs` | One entry per HELOC feature column |
| `outputs.probability` | Positive-class probability from the sklearn classifier |
| `path_count` | Vertex explanation tuning — higher values are slower but often more stable |

You do not run this snippet yourself — `run_batch_explainability.py` calls it inside `upload_explainable_model()`.

---

## 🚀 Step 5: Run the batch explain job

```bash
python run_batch_explainability.py --config config.local.json
```

Pipeline steps:

1. **Upload** `model.joblib` + `feature_names.json` to `gs://YOUR_BUCKET/.../models/`
2. **Register** Vertex model with explanation metadata
3. **Start** `batch_predict` with `instances_format="bigquery"`, `generate_explanation=True`
4. **Write** predictions + explanations to `heloc_batch_explanations`

Reuse an existing model (skip re-upload):

```bash
python run_batch_explainability.py --config config.local.json \
  --model-id projects/YOUR_PROJECT/locations/us-central1/models/MODEL_ID
```

Submit without waiting on the terminal:

```bash
python run_batch_explainability.py --config config.local.json --async-submit
```

Open **Vertex AI → Batch predictions** in the console. A two-row smoke job typically takes a few minutes (model deploy + explanation compute). Wait for **`JOB_STATE_SUCCEEDED`**.

### Machine type tips

If the job fails with **"Machine type temporarily unavailable"** (gRPC code 14), switch `machine_type` in config — e.g. `n2-standard-4`, `c2-standard-4`, or `e2-standard-4` — and retry. That is regional capacity, not an application bug.

---

## 📈 Step 6: Read and visualize results

### BigQuery output

After the job succeeds:

```sql
SELECT *
FROM `YOUR_GCP_PROJECT_ID.ml_explainability.heloc_batch_explanations`
LIMIT 10;
```

Vertex writes prediction fields and explanation payloads (including **`featureAttributions`**) into the output table. Each attribution tells you how much a feature pushed the score up or down for that row.

- **Positive attribution** → feature increased the prediction
- **Negative attribution** → feature decreased the prediction

### Attribution chart (JSONL path or demo)

For a quick chart before your first Vertex run:

```bash
python visualize_attributions.py --demo
```

If you use the optional GCS JSONL path (see below), download one output file and run:

```bash
python visualize_attributions.py --input /path/to/predictions_00000.jsonl
```

### Looker Studio (optional)

Connect Looker Studio to `heloc_batch_explanations`. Build a bar chart or table of top attributions per applicant — useful when stakeholders want a dashboard instead of SQL.

---

## 🌠 Optional: GCS JSONL instead of BigQuery

BigQuery is the default. For a GCS-only path, set in `config.local.json`:

```json
"input_format": "jsonl"
```

Then:

```bash
python prepare_batch_input.py
python run_batch_explainability.py --config config.local.json
```

- **Input:** `gs://YOUR_BUCKET/.../batch-input/instances.jsonl`
- **Output:** `gs://YOUR_BUCKET/.../batch-output/`

❗ Vertex sklearn containers expect **numeric feature lists** in JSONL, not dicts keyed by feature name. `prepare_batch_input.py` writes one line per row: `{"instances": [[57, 179, 8, ...]]}`.

---

## ❗ Common issues

| Issue | Fix |
|---|---|
| Feature name mismatch | Align `instances.json` / BQ columns with `feature_names.json` |
| `float() argument must be a ... 'dict'` on predict | Use numeric lists in JSONL, or BQ FLOAT columns |
| sklearn version mismatch at serve time | Train with sklearn 1.6.x; use `sklearn-cpu.1-6` container |
| Machine type unavailable (code 14) | Retry with `n2-standard-4` / `c2-standard-4` or another region |
| Missing `project_id` or bucket | Fill `config.local.json` or export env vars |
| Batch vs online explain | This tutorial uses **batch** `Model.batch_predict`; see `run_explainability.py` for online demo only |
| `batch_predict()` unexpected keyword | Pass `batch_size=` directly (not `manual_batch_tuning_parameters`) on recent `google-cloud-aiplatform` SDK versions |

---

## ✅ Checklist

- [ ] Local smoke test passes (`--local-only`, unit tests)
- [ ] `config.local.json` has project, bucket, BQ tables
- [ ] Model trained with sklearn 1.6.x
- [ ] `prepare_bq_input.py` loaded input table
- [ ] Batch job `JOB_STATE_SUCCEEDED`
- [ ] Output table has predictions + `featureAttributions`
- [ ] (Optional) Looker dashboard on explanation table

---

## Wrapping up

You now have an end-to-end **Vertex AI batch explainability** flow: train a sklearn credit-risk model, stage it on Vertex with explanation metadata, score and explain rows from BigQuery in one job, and inspect attributions in BigQuery or Looker.

The same pattern generalizes to any tabular sklearn model — swap the dataset, feature list, and BQ tables in `config.local.json`.

**Tags:** Vertex AI · Vertex Explainable AI · MLOps · BigQuery · scikit-learn · Google Cloud
