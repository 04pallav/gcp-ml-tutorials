# ☁️Vertex AI Batch Explainability: Explain Credit-Risk Predictions with BigQuery and scikit-learn 🔍

**By Pallav Anand**

This GCP ML project builds **batch explainability for credit-risk scoring** on Vertex AI. You train a scikit-learn classifier on a public lending dataset, register it with explanation metadata, and explain applicants in bulk — each row gets a prediction and per-feature attributions written back to BigQuery.

🗃️ **Cloud Storage** — `model.joblib` and `feature_names.json`

📊 **BigQuery** — batch input table and explanation output table

🤖 **Vertex AI** — model upload + batch prediction with explanations

📈 **Looker Studio** (optional) — dashboards on attribution results

These services work together to score and explain credit-risk applications at scale — without calling `endpoint.explain()` row by row.

Find the code on GitHub:

[github.com/04pallav/gcp-ml-tutorials](https://github.com/04pallav/gcp-ml-tutorials/tree/main/code/vertex-batch-explainability)

---

## 📋 The dataset

**What is a HELOC?** A **Home Equity Line of Credit** is a loan homeowners can draw on using their home equity — similar to a credit card, but secured by the house. Banks use credit scores and payment history to decide approvals and to monitor risk over time.

**What we're using:** The [FICO HELOC dataset on OpenML](https://www.openml.org/d/45023) — a public benchmark FICO released for teaching credit-risk models. Each row is one borrower. The label is **good** (paid as agreed) vs **bad** (90+ days past due within 24 months).

The data has **22 numeric features** lenders care about, for example:

- `ExternalRiskEstimate` — outside risk score
- `NumInqLast6M` — how many times the borrower applied for credit recently
- `NetFractionRevolvingBurden` — how much of their revolving credit limit is in use
- `MSinceMostRecentDelq` — months since the most recent delinquency
- `PercentTradesNeverDelq` — share of accounts never delinquent

`instances.json` in the repo has **two sample borrowers** for a cheap smoke test. In production, your BigQuery input table would hold the same kind of columns for the population you need to explain.

---

## 🗃️ Step 1: Create a staging bucket

Create a Cloud Storage bucket in the same region you will use for Vertex and BigQuery (e.g. `us-central1`). The scripts upload `model.joblib` here before registering the model on Vertex.

Example path: `gs://your-bucket/vertex-batch-explain`

---

## 👩‍💻 Step 2: Cloud Shell setup

Open [Cloud Shell](https://shell.cloud.google.com/) and run:

```bash
gcloud config set project your-project-id
gcloud services enable aiplatform.googleapis.com storage.googleapis.com bigquery.googleapis.com

git clone https://github.com/04pallav/gcp-ml-tutorials.git
cd gcp-ml-tutorials/code/vertex-batch-explainability
pip install -r requirements.txt

cp config.example.json config.local.json
```

Edit `config.local.json` and replace **`your-project-id`** and **`gs://your-bucket/vertex-batch-explain`** with your values. The rest of the defaults are fine for the smoke test — see [`config.example.json`](https://github.com/04pallav/gcp-ml-tutorials/blob/main/code/vertex-batch-explainability/config.example.json) in the repo.

❗ Train with sklearn **1.6.x** and serve with `sklearn-cpu.1-6`. Keep bucket, Vertex, and BigQuery in the same region.

---

## 🐝 Step 3: Train the model

Train locally — Vertex serves this exact `model.joblib`:

```bash
python train_model.py
```

The pipeline is median imputer → standard scaler → logistic regression. Outputs land in `models/model.joblib` and `models/feature_names.json`.

---

## 📊 Step 4: Load batch input into BigQuery

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

## 🚀 Step 5: Run the batch explain job

One command trains, uploads the model with explanation metadata, and starts the batch job:

```bash
python run_batch_explain.py
```

Unlike online `endpoint.explain()` (one row at a time), batch explain registers metadata once and runs `batch_predict` with `generate_explanation=True` — Vertex returns **feature attributions** alongside predictions in the output table.

Open **Vertex AI → Batch predictions** and wait for **`JOB_STATE_SUCCEEDED`**. A two-row smoke job typically takes a few minutes.

❗ If you see **"Machine type temporarily unavailable"**, switch `machine_type` in config (e.g. `c2-standard-4`) and retry.

---

## 📈 Step 6: Read results in BigQuery

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
