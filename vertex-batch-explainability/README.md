# ☁️ Vertex AI Batch Explainability: Explain Credit-Risk Predictions with BigQuery and scikit-learn 🔍

This GCP ML project focuses on **batch explainability for credit-risk scoring** on Vertex AI. You train a scikit-learn classifier, register it on Vertex, and run one batch job that returns a prediction and per-feature attributions for each row in BigQuery.

- 🗃️ GCS is used to store the sample input data
- 📊 BigQuery holds the batch input rows and the explanation output table
- 🤖 Vertex AI uploads the model and runs batch prediction with explanations enabled
- 📈 Looker Studio connects to the output table for dashboards

These services work together to train the model, run batch explanations, and write results to BigQuery.

# 🗃️ GCS

Upload the provided `instances.json` file to your designated Google Cloud Storage (GCS) bucket. This sample data comes from the [FICO HELOC dataset on OpenML](https://www.openml.org/d/45023) — two borrowers a lender might score and explain. Each row includes information such as `ExternalRiskEstimate`, `NumInqLast6M`, `NetFractionRevolvingBurden`, `MSinceMostRecentDelq`, and `PercentTradesNeverDelq` — outside risk score, recent credit applications, revolving utilization, months since last delinquency, and share of accounts never delinquent. The label is **good** (paid as agreed) vs **bad** (90+ days past due), the kind of payment-history and utilization picture banks review for home-equity credit lines.

👩‍💻

Upload from Cloud Shell: `gcloud storage cp instances.json gs://your-bucket/instances.json`

❗ Keep the bucket in the same region as Vertex and BigQuery (e.g. `us-central1`).

# 🐝 `batch_explain.py`

📖

[`batch_explain.py`](batch_explain.py) is a Python script that runs the full Vertex batch explainability flow.

The pipeline consists of the following steps:

1. Command-line arguments are parsed to specify your GCP project, GCS bucket, and input file.
2. `instances.json` is read from your GCS bucket and loaded into the BigQuery table `heloc_batch_input` — one FLOAT column per feature.
3. The HELOC training data is downloaded from OpenML and a scikit-learn pipeline is trained (median imputer → standard scaler → logistic regression).
4. `model.joblib` and `feature_names.json` are uploaded to your GCS bucket.
5. The model is registered on Vertex AI with explanation settings for all 22 input features.
6. A batch prediction job reads from `heloc_batch_input`, scores each row, and writes predictions plus feature attributions to `heloc_batch_explanations`.

👩‍💻

Set the project in Cloud Shell: `gcloud config set project your-project-id`

Enable APIs: `gcloud services enable aiplatform.googleapis.com storage.googleapis.com bigquery.googleapis.com`

Install dependencies: `pip install -r requirements.txt`

Run the pipeline: `python batch_explain.py --project-id your-project-id --bucket-uri gs://your-bucket/vertex-batch-explain --instances gs://your-bucket/instances.json`

❗ Train with sklearn **1.6.x** (pinned in `requirements.txt`) and serve with the `sklearn-cpu.1-6` container.

❗ Keep bucket, Vertex jobs, and BigQuery in the same region (e.g. `us-central1` / `US`).

# 📊 BigQuery

Open BigQuery and check the input table:

```sql
SELECT * FROM `your-project-id.ml_explainability.heloc_batch_input` LIMIT 10;
```

You should see 2 rows — one per sample borrower from `instances.json`.

After the batch job finishes, open the output table:

```sql
SELECT * FROM `your-project-id.ml_explainability.heloc_batch_explanations` LIMIT 10;
```

Each row has a prediction and **feature attributions** — how much each input feature pushed the score up or down for that borrower.

# 🤖 Vertex AI

Open **Vertex AI → Batch predictions** in the console. The job typically takes a few minutes on a two-row smoke test. Wait for **JOB_STATE_SUCCEEDED**.

❗ If you see **"Machine type temporarily unavailable"**, retry with `--machine-type c2-standard-4`.

# 📈 Looker Studio

Connect Looker Studio to `heloc_batch_explanations` and build a bar chart of top attributions per borrower.

## About

Vertex AI batch explainability on HELOC credit-risk data — BigQuery in, predictions + feature attributions out.
