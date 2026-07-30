# ☁️Vertex AI Batch Explainability: Explain Credit-Risk Predictions with BigQuery and scikit-learn 🔍

**By Pallav Anand**

This GCP ML project builds **batch explainability for credit-risk scoring** on Vertex AI. You train a scikit-learn classifier on the FICO HELOC dataset, register it with explanation metadata, and explain applicants in bulk — each row gets a prediction and per-feature attributions written back to BigQuery.

🗃️ **Cloud Storage** — `model.joblib` and `feature_names.json`

📊 **BigQuery** — batch input table and explanation output table

🤖 **Vertex AI** — model upload + batch prediction with explanations

📈 **Looker Studio** (optional) — dashboards on attribution results

These services work together to score and explain credit-risk applications at scale — without calling `endpoint.explain()` row by row.

Find the code on GitHub:

[github.com/04pallav/gcp-ml-tutorials](https://github.com/04pallav/gcp-ml-tutorials/tree/main/code/vertex-batch-explainability)

---

## 📋 The HELOC dataset

We use the **Home Equity Line of Credit (HELOC)** dataset from OpenML — a binary credit-risk benchmark with **22 numeric features** (payment history, utilization, inquiries, etc.) and a good/bad label.

`instances.json` in the repo has **two sample applicants** for a cheap smoke test. In production, your BigQuery input table holds the same feature columns for the population you need to explain.

---

## 👩‍💻 Step 1: Clone the repo and set your project

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

Set the project in Cloud Shell (or locally):

```bash
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud services enable aiplatform.googleapis.com storage.googleapis.com bigquery.googleapis.com
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account.json
```

Your service account needs `roles/aiplatform.user`, `roles/storage.objectAdmin` on the bucket, and `roles/bigquery.dataEditor`.

❗ Train with sklearn **1.6.x** and serve with `sklearn-cpu.1-6`. Keep bucket, Vertex, and BigQuery in the same region.

---

## 🐝 Step 2: Train the model

Train locally — Vertex serves this exact `model.joblib`:

```bash
python train_model.py
```

The pipeline is median imputer → standard scaler → logistic regression. Outputs land in `models/model.joblib` and `models/feature_names.json`.

---

## 📊 Step 3: Load batch input into BigQuery

Load the two sample rows into your input table:

```bash
python prepare_bq_input.py
```

Confirm in BigQuery Console:

```sql
SELECT * FROM `YOUR_GCP_PROJECT_ID.ml_explainability.heloc_batch_input` LIMIT 10;
```

❗ Column names must match `feature_names.json` exactly — they become the `inputs` map in Vertex `ExplanationMetadata`.

---

## 🚀 Step 4: Run the batch explain job

One command trains, uploads the model with explanation metadata, and starts the batch job:

```bash
python run_batch_explain.py
```

Unlike online `endpoint.explain()` (one row at a time), batch explain registers metadata once and runs `batch_predict` with `generate_explanation=True` — Vertex returns **feature attributions** alongside predictions in the output table.

Open **Vertex AI → Batch predictions** and wait for **`JOB_STATE_SUCCEEDED`**. A two-row smoke job typically takes a few minutes.

❗ If you see **"Machine type temporarily unavailable"**, switch `machine_type` in config (e.g. `c2-standard-4`) and retry.

---

## 📈 Step 5: Read results in BigQuery

```sql
SELECT *
FROM `YOUR_GCP_PROJECT_ID.ml_explainability.heloc_batch_explanations`
LIMIT 10;
```

Each row includes predictions and **`featureAttributions`**. Positive attribution → feature pushed the score up; negative → pushed it down.

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

You now have an end-to-end Vertex AI batch explainability flow: train a sklearn credit-risk model, stage it on Vertex, score and explain rows from BigQuery in one job, and inspect attributions in BigQuery or Looker.

**Tags:** Vertex AI · Vertex Explainable AI · MLOps · BigQuery · scikit-learn · Google Cloud
