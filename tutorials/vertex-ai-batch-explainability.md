# ☁️Vertex AI Batch Explainability: Explain Credit-Risk Predictions with BigQuery and scikit-learn 🔍

**By Pallav Anand**

This GCP ML project builds **batch explainability for credit-risk scoring** on Vertex AI. You train a scikit-learn classifier on a public lending dataset, register it with explanation metadata, and explain applicants in bulk — each row gets a prediction and per-feature attributions written back to BigQuery.

🗃️ **Cloud Storage** is used to store the trained `model.joblib` before Vertex registers it

📊 **BigQuery** holds the batch input rows and the explanation output table

🤖 **Vertex AI** uploads the model and runs batch prediction with explanations enabled

📈 **Looker Studio** (optional) connects to the output table for dashboards

These services work together to score and explain credit-risk applications at scale — without calling `endpoint.explain()` row by row.

Find the code on my GitHub account:

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

`instances.json` in the repo has **two sample borrowers** for a cheap smoke test.

---

## 🗃️ GCS

Create a Cloud Storage bucket in the same region you will use for Vertex and BigQuery (e.g. `us-central1`). The scripts upload `model.joblib` and `feature_names.json` here before registering the model on Vertex.

**[Screenshot: Cloud Console → Cloud Storage → Create bucket]**

Example path: `gs://your-bucket/vertex-batch-explain`

---

## 🐝 `train_model.py`

📖

[`train_model.py` on GitHub](https://github.com/04pallav/gcp-ml-tutorials/blob/main/code/vertex-batch-explainability/train_model.py)

`train_model.py` downloads the HELOC data from OpenML, trains a scikit-learn pipeline, and saves two files locally:

1. `models/model.joblib` — median imputer → standard scaler → logistic regression
2. `models/feature_names.json` — the 22 feature column names Vertex needs for explanations

👩‍💻

Set the project in Cloud Shell: `gcloud config set project your-project-id`

Enable APIs: `gcloud services enable aiplatform.googleapis.com storage.googleapis.com bigquery.googleapis.com`

Clone the repo and install dependencies:

```bash
git clone https://github.com/04pallav/gcp-ml-tutorials.git
cd gcp-ml-tutorials/code/vertex-batch-explainability
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

❗ Train with sklearn **1.6.x** (pinned in `requirements.txt`) and serve with the `sklearn-cpu.1-6` container. Training on a newer sklearn and serving on an older container causes deserialization errors.

---

## 📊 BigQuery

📖

[`prepare_bq_input.py` on GitHub](https://github.com/04pallav/gcp-ml-tutorials/blob/main/code/vertex-batch-explainability/prepare_bq_input.py)

`prepare_bq_input.py` reads `instances.json` (two sample borrowers) and loads them into a BigQuery table — one FLOAT column per HELOC feature. The script creates the `ml_explainability` dataset if it does not exist.

👩‍💻

Load the sample rows:

```bash
python prepare_bq_input.py --project-id your-project-id
```

Open BigQuery and confirm the table:

```sql
SELECT * FROM `your-project-id.ml_explainability.heloc_batch_input` LIMIT 10;
```

**[Screenshot: BigQuery console showing `heloc_batch_input` with 2 rows]**

❗ Column names must match `feature_names.json` exactly. Vertex uses them when registering which inputs are explainable.

❗ Keep bucket, Vertex jobs, and BigQuery in the same region (e.g. `us-central1` / `US`).

---

## 🤖 Vertex AI

📖

[`run_batch_explain.py` on GitHub](https://github.com/04pallav/gcp-ml-tutorials/blob/main/code/vertex-batch-explainability/run_batch_explain.py)

`run_batch_explain.py` does the Vertex work in one run:

1. Trains the HELOC model (same pipeline as `train_model.py`)
2. Uploads `model.joblib` to your GCS bucket
3. Registers the model on Vertex with explanation metadata
4. Starts a batch prediction job with `generate_explanation=True`
5. Writes predictions and per-feature attributions to `heloc_batch_explanations` in BigQuery

Unlike online `endpoint.explain()` (one row at a time), batch explain registers metadata once and returns **feature attributions** alongside predictions for every row in the input table.

👩‍💻

Run the batch explain job:

```bash
python run_batch_explain.py \
  --project-id your-project-id \
  --bucket-uri gs://your-bucket/vertex-batch-explain
```

Open **Vertex AI → Batch predictions** in the console. A two-row smoke job typically takes a few minutes — wait for **JOB_STATE_SUCCEEDED**.

**[Screenshot: Vertex AI → Batch predictions → job status SUCCEEDED]**

❗ If you see **"Machine type temporarily unavailable"**, retry with `--machine-type c2-standard-4` or another type in the same region.

---

## 📈 Looker Studio

After the job succeeds, query the output table:

```sql
SELECT *
FROM `your-project-id.ml_explainability.heloc_batch_explanations`
LIMIT 10;
```

Each row includes predictions and **`featureAttributions`**. Positive attribution → feature pushed the score up; negative → pushed it down.

**[Screenshot: BigQuery output row showing `featureAttributions`]**

Connect **Looker Studio** to `heloc_batch_explanations` and build a bar chart of top attributions per borrower — useful when stakeholders want a dashboard instead of SQL.

**Tags:** Vertex AI · Vertex Explainable AI · BigQuery · scikit-learn · Google Cloud
