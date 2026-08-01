# <img width="40" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcp.png"> Predicting and Explaining Loan Default Risk with Vertex AI, BigQuery and scikit-learn 🔍

This GCP ML project focuses on **predicting and explaining loan default risk** on Vertex AI. You train a scikit-learn classifier on FICO consumer credit data, register it on Vertex, and run one batch job that returns each borrower's predicted default probability plus per-feature attributions in BigQuery.

- <img width="18" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcs.png"> GCS is used to store the sample input data
- <img width="18" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/bigquery.png"> BigQuery holds the batch input rows and the explanation output table
- <img width="18" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/vertex-ai.png"> Vertex AI uploads the model and runs batch prediction with explanations enabled
- <img width="18" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/looker.png"> Looker Studio connects to the output table for dashboards

These services work together to train the model, run batch explanations, and write results to BigQuery.

![Architecture diagram](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/architecture-diagram.png)

Find the code and CSV file on my github account.

[github.com/04pallav/gcp-ml-tutorials/vertex-batch-explainability](https://github.com/04pallav/gcp-ml-tutorials/tree/main/vertex-batch-explainability)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcs.png"> Step 1 — Upload the data to GCS

Upload the provided CSV file to your designated Google Cloud Storage (GCS) bucket. This sample data comes from the [FICO consumer credit dataset on OpenML](https://www.openml.org/d/45023) — 10,000 borrowers a lender might score and explain. Each row includes bureau features such as `ExternalRiskEstimate`, `NumInqLast6M`, `NetFractionRevolvingBurden`, `MSinceMostRecentDelq`, and `PercentTradesNeverDelq` — outside risk score, recent credit applications, revolving utilization, months since last delinquency, and share of accounts never delinquent.

👨‍💻 `gsutil cp instances.csv gs://your-bucket/instances.csv`

![Upload instances.csv to GCS](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcs-upload-instances.png)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcp.png"> Step 2 — Set up the cloud shell

Set the project and install scikit-learn and the Vertex AI SDK.

👨‍💻 `gcloud config set project your-project-id`

👨‍💻 `pip install google-cloud-aiplatform 'scikit-learn==1.6.*'`

![Cloud shell install](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/step2-install.png)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/bigquery.png"> Step 3 — Load the batch input into BigQuery

The `load` step reads `instances.csv` and writes it to the BigQuery table `heloc_batch_input` — one FLOAT column per feature.

👨‍💻 `python batch_explain.py load --instances gs://your-bucket/instances.csv`

Then inspect the rows in BigQuery:

```sql
SELECT * FROM `your-project-id.ml_explainability.heloc_batch_input` LIMIT 10;
```

You should see 10,000 rows from `instances.csv`.

![heloc_batch_input in BigQuery](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/step3-bq-input.png)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/vertex-ai.png"> Step 4 — Train the model

**The model** — `batch_explain.py` trains a scikit-learn **logistic regression** on `heloc.csv` (10,000 labeled borrowers from the [FICO consumer credit dataset](https://www.openml.org/d/45023)). It takes 22 credit-bureau features per borrower and predicts `probability` — the chance the borrower ends up 90+ days past due (`RiskPerformance` = Bad).

The `train` step fits the pipeline, prints train/test ROC AUC, and uploads `model.joblib` + `feature_names.json` to your bucket.

👨‍💻 `python batch_explain.py train --bucket-uri gs://your-bucket/vertex-batch-explain`

Check the printed ROC AUC and confirm `model.joblib` landed in `gs://your-bucket/vertex-batch-explain/models/`.

![Training output and model artifact](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/step4-train.png)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/vertex-ai.png"> Step 5 — Register the model and run the batch explanation job

The `explain` step registers the model on Vertex AI with explanations enabled for all 22 features and starts a batch job that writes scores + attributions to `heloc_batch_explanations`. Re-running reuses the existing `heloc-batch-explain` model.

👨‍💻 `python batch_explain.py explain --bucket-uri gs://your-bucket/vertex-batch-explain`

❗ Make sure that all your files and services are in the same location. E.g. both buckets should be in the same location or you will get a similar error message: ‘Cannot read and write in different locations: source: US, destination: EU’

Open **Vertex AI → Batch predictions** in the console and wait for **JOB_STATE_SUCCEEDED**.

![Vertex AI batch prediction job](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/step5-vertex-job.png)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/bigquery.png"> Step 6 — Inspect the explanations in BigQuery

After the batch job finishes, open the output table:

```sql
SELECT * FROM `your-project-id.ml_explainability.heloc_batch_explanations` LIMIT 10;
```

Each row has a prediction and **feature attributions** — how much each input feature pushed the score up or down for that borrower.

![heloc_batch_explanations in BigQuery](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/step6-bq-output.png)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/looker.png"> Step 7 — Visualize in Looker Studio

Connect Looker Studio to `heloc_batch_explanations` and build a bar chart of top attributions per borrower.

![Looker Studio dashboard](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/step7-looker.png)

## About

GCP ML explainability on consumer credit data — BigQuery in, default probabilities + feature attributions out.
