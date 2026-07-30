# <img width="40" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcp.png"> GCP ML Explainability: Batch Credit-Risk Scoring with Vertex AI, BigQuery and scikit-learn 🔍

This GCP ML project focuses on **batch explainability for credit-risk scoring** on Vertex AI. You train a scikit-learn classifier, register it on Vertex, and run one batch job that returns a prediction and per-feature attributions for each row in BigQuery.

- <img width="18" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcs.png"> GCS is used to store the sample input data
- <img width="18" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/bigquery.png"> BigQuery holds the batch input rows and the explanation output table
- <img width="18" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/vertex-ai.png"> Vertex AI uploads the model and runs batch prediction with explanations enabled
- <img width="18" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/looker.png"> Looker Studio connects to the output table for dashboards

These services work together to train the model, run batch explanations, and write results to BigQuery.

Find the code and CSV file on my github account.

[github.com/04pallav/gcp-ml-tutorials/vertex-batch-explainability](https://github.com/04pallav/gcp-ml-tutorials/tree/main/vertex-batch-explainability)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcs.png"> GCS

Upload the provided CSV file to your designated Google Cloud Storage (GCS) bucket. This sample data comes from the [FICO HELOC dataset on OpenML](https://www.openml.org/d/45023) — 10,000 borrowers a lender might score and explain. It includes information such as `ExternalRiskEstimate`, `NumInqLast6M`, `NetFractionRevolvingBurden`, `MSinceMostRecentDelq`, and `PercentTradesNeverDelq` — outside risk score, recent credit applications, revolving utilization, months since last delinquency, and share of accounts never delinquent. The data showcases various credit-risk scenarios, providing valuable insights into payment history and utilization patterns banks review for home-equity credit lines.

![image](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcs-upload-instances.png)

# <img width="40" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/vertex-ai.png"> `batch_explain.py`

📖

`batch_explain.py` code is a batch explainability pipeline implemented using scikit-learn and Vertex AI. It reads data from an input file, trains a credit-risk classifier on HELOC data, registers the model on Vertex AI with explanation settings, and writes predictions plus feature attributions to a BigQuery table.

The pipeline consists of the following steps:

1. Command-line arguments are parsed to specify your GCP project, GCS bucket, and input file.
2. The data is read from the input file and loaded into the BigQuery table `heloc_batch_input` — one FLOAT column per feature.
3. The HELOC training data is read from `heloc.csv` and a scikit-learn pipeline is trained (median imputer → standard scaler → logistic regression).
4. `model.joblib` and `feature_names.json` are uploaded to your GCS bucket.
5. The model is registered on Vertex AI with explanation settings for all 22 input features.
6. A batch prediction job reads from `heloc_batch_input`, scores each row, and writes predictions plus feature attributions to `heloc_batch_explanations`.
7. The batch job is started, and the model and job resource names are printed to the console.

👩‍💻

Set the project in the cloud shell: `gcloud config set project your-project-id`

Install dependencies in the cloud shell: `pip install -r requirements.txt`

Give the batch_explain.py code a test run in the shell and then check the results in BigQuery: `python batch_explain.py --project-id your-project-id --bucket-uri gs://your-bucket/vertex-batch-explain --instances gs://your-bucket/instances.csv`

❗ Make sure that all your files and services are in the same location. E.g. both buckets should be in the same location or you will get a similar error message: ‘Cannot read and write in different locations: source: US, destination: EU’

# 📊 BigQuery

Open BigQuery and check the input table:

```sql
SELECT * FROM `your-project-id.ml_explainability.heloc_batch_input` LIMIT 10;
```

You should see 10,000 rows from `instances.csv`.

After the batch job finishes, open the output table:

```sql
SELECT * FROM `your-project-id.ml_explainability.heloc_batch_explanations` LIMIT 10;
```

Each row has a prediction and **feature attributions** — how much each input feature pushed the score up or down for that borrower.

# 🤖 Vertex AI

Open **Vertex AI → Batch predictions** in the console. Wait for **JOB_STATE_SUCCEEDED**.

# 📈 Looker Studio

Connect Looker Studio to `heloc_batch_explanations` and build a bar chart of top attributions per borrower.

## About

GCP ML explainability on HELOC credit-risk data — BigQuery in, predictions + feature attributions out.
