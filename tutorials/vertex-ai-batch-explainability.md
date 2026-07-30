# ☁️Vertex AI Batch Explainability: Explain Credit-Risk Predictions with BigQuery and scikit-learn 🔍

**By Pallav Anand**

Code: [`code/vertex-batch-explainability/`](https://github.com/04pallav/gcp-ml-tutorials/tree/main/code/vertex-batch-explainability)

---

This tutorial walks through **Vertex AI batch explainability** on a credit-risk use case. You train a scikit-learn classifier on the **FICO HELOC** dataset, register it on Vertex with explanation metadata, and run **`Model.batch_predict(..., generate_explanation=True)`** so every row gets a prediction **and** per-feature attributions — reading from **BigQuery** and writing results back to **BigQuery**.

That is the pattern you want when explanations must scale: compliance reviews, model monitoring, or offline analysis across thousands of applicants — without calling `endpoint.explain()` row by row.

**Stack:** Cloud Storage (model artifacts) · BigQuery (batch I/O) · Vertex AI Model + Explainable AI

> This tutorial uses **Vertex AI batch prediction with explanations**. It is not the open-source `shap` package, not Beam/Dataflow custom explainers, and not online-only `endpoint.explain()`.

---

## 🎯 What is batch explainability?

**Online explain:** deploy a model to an endpoint, call `explain()` per request. Fine for a few rows; expensive at scale.

**Batch explain:** register explanation metadata once, then run `batch_predict` with `generate_explanation=True`. Vertex scores every row and returns **feature attributions** in the same job output.

---

## 📋 The HELOC dataset

We use the **Home Equity Line of Credit (HELOC)** dataset from OpenML — 22 numeric features and a good/bad credit-risk label. `instances.json` ships with two sample applicants for a cheap smoke test.

---

## 🏗️ Architecture

```
instances.json
      │
      ├─ train_model.py ──► model.joblib ──► GCS
      │
      └─ prepare_bq_input.py ──► BigQuery input table
                                      │
              Model.upload(explanation_metadata=…)
                                      │
              Model.batch_predict(generate_explanation=True)
                                      │
                              BigQuery output table
```

---

## 📦 Repo layout

| File | Purpose |
|---|---|
| `config.example.json` | Project, bucket, BQ tables |
| `train_model.py` | Train sklearn pipeline locally |
| `prepare_bq_input.py` | Load `instances.json` → BigQuery |
| `run_batch_explain.py` | Upload model + batch explain job |
| `instances.json` | Two sample applicants |

---

## 👩‍💻 Step 1: Clone and configure

```bash
git clone https://github.com/04pallav/gcp-ml-tutorials.git
cd gcp-ml-tutorials/code/vertex-batch-explainability
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp config.example.json config.local.json
```

Edit `config.local.json` — set `project_id` and `staging_bucket_uri`:

```json
{
  "project_id": "YOUR_GCP_PROJECT_ID",
  "region": "us-central1",
  "staging_bucket_uri": "gs://YOUR_BUCKET/vertex-batch-explain",
  "display_name": "heloc-batch-explain",
  "job_name": "heloc-batch-explain-job",
  "bq_dataset": "ml_explainability",
  "bq_input_table": "heloc_batch_input",
  "bq_output_table": "heloc_batch_explanations",
  "serving_container": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest",
  "machine_type": "n2-standard-4",
  "batch_size": 16
}
```

Enable APIs and auth:

```bash
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud services enable aiplatform.googleapis.com storage.googleapis.com bigquery.googleapis.com
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account.json
```

Your service account needs `roles/aiplatform.user`, `roles/storage.objectAdmin` on the bucket, and `roles/bigquery.dataEditor`.

❗ Train with sklearn **1.6.x** and serve with `sklearn-cpu.1-6`. Keep bucket, Vertex, and BigQuery in the same region.

---

## 🐝 Step 2: Train and load BigQuery input

Train locally (optional sanity check):

```bash
python train_model.py
```

Load sample rows into BigQuery:

```bash
python prepare_bq_input.py
```

Confirm in BigQuery Console:

```sql
SELECT * FROM `YOUR_GCP_PROJECT_ID.ml_explainability.heloc_batch_input` LIMIT 10;
```

Column names must match `feature_names.json` exactly — they become the `inputs` map in Vertex `ExplanationMetadata`.

---

## 🚀 Step 3: Run batch explain

One command trains, uploads the model with explanation metadata, and starts the batch job:

```bash
python run_batch_explain.py
```

`run_batch_explain.py` registers the model with sampled Shapley attribution config, then calls:

```python
model.batch_predict(
    instances_format="bigquery",
    bigquery_source="bq://PROJECT.DATASET.INPUT_TABLE",
    predictions_format="bigquery",
    bigquery_destination_prefix="bq://PROJECT.DATASET.OUTPUT_TABLE",
    generate_explanation=True,
    ...
)
```

Open **Vertex AI → Batch predictions** and wait for **`JOB_STATE_SUCCEEDED`**. A two-row smoke job typically takes a few minutes.

If you see **"Machine type temporarily unavailable"**, switch `machine_type` in config (e.g. `c2-standard-4`) and retry.

---

## 📈 Step 4: Read results

```sql
SELECT *
FROM `YOUR_GCP_PROJECT_ID.ml_explainability.heloc_batch_explanations`
LIMIT 10;
```

Vertex writes predictions and **`featureAttributions`** per row. Positive attribution → feature pushed the score up; negative → pushed it down.

Connect **Looker Studio** to the output table for stakeholder dashboards.

---

## ❗ Common issues

| Issue | Fix |
|---|---|
| Feature name mismatch | Align BQ columns with `feature_names.json` |
| sklearn version mismatch | Train with 1.6.x; serve with `sklearn-cpu.1-6` |
| Machine type unavailable | Retry with `n2-standard-4` / `c2-standard-4` |
| Missing project or bucket | Fill `config.local.json` |

---

## ✅ Checklist

- [ ] `config.local.json` has project + bucket
- [ ] `prepare_bq_input.py` loaded input table
- [ ] Batch job `JOB_STATE_SUCCEEDED`
- [ ] Output table has `featureAttributions`

---

## Wrapping up

You now have an end-to-end **Vertex AI batch explainability** flow: train a sklearn credit-risk model, stage it on Vertex with explanation metadata, score and explain rows from BigQuery in one job, and inspect attributions in BigQuery or Looker.

**Tags:** Vertex AI · Vertex Explainable AI · MLOps · BigQuery · scikit-learn · Google Cloud
