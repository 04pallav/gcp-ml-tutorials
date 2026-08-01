# <img width="40" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcp.png"> GCP ML Project: Explaining Loan Default Risk Predictions with Vertex AI, BigQuery and scikit-learn 🔍

This GCP ML project focuses on **explaining loan default risk predictions** on Vertex AI. You train a scikit-learn classifier on FICO consumer credit data, register it on Vertex, and run one batch job that returns each borrower's predicted default probability plus per-feature attributions in BigQuery.

- <img width="18" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcs.png"> GCS is used to store the sample input data
- <img width="18" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/bigquery.png"> BigQuery holds the batch input rows and the explanation output table
- <img width="18" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/vertex-ai.png"> Vertex AI uploads the model and runs batch prediction with explanations enabled
- <img width="18" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/looker.png"> Looker Studio connects to the output table for dashboards

These services work together to train the model, run batch explanations, and write results to BigQuery.

![Architecture diagram](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/architecture-diagram.png)

Find the code and CSV file on my github account.

[github.com/04pallav/gcp-ml-tutorials/vertex-batch-explainability](https://github.com/04pallav/gcp-ml-tutorials/tree/main/vertex-batch-explainability)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcs.png"> Step 1 — Upload data to GCS

Upload the provided CSV file to your designated Google Cloud Storage (GCS) bucket. This sample data comes from the [FICO consumer credit dataset on OpenML](https://www.openml.org/d/45023) — 10,000 borrowers a lender might score and explain. Feature definitions and missing-value codes (`-7`, `-8`, `-9`) are documented on that [OpenML page](https://www.openml.org/d/45023). Each row includes bureau features such as `ExternalRiskEstimate`, `NumInqLast6M`, `NetFractionRevolvingBurden`, `MSinceMostRecentDelq`, and `PercentTradesNeverDelq` — outside risk score, recent credit applications, revolving utilization, months since last delinquency, and share of accounts never delinquent.

![Upload instances.csv to GCS](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcs-upload-instances.png)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/gcp.png"> Step 2 — Set up cloud shell

Set the project and install scikit-learn and the Vertex AI SDK.

👨‍💻 `gcloud config set project your-project-id`

👨‍💻 `pip install google-cloud-aiplatform 'scikit-learn==1.6.*' matplotlib`

![Cloud shell install](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/step2-install.png)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/bigquery.png"> Step 3 — Load data to BigQuery

The `load` step reads `instances.csv` and writes the 22 feature columns to BigQuery table `heloc_batch_input` (the `RiskPerformance` label stays in the CSV but is not loaded into BigQuery).

👨‍💻 `python batch_explain.py load --instances gs://your-bucket/instances.csv`

Then inspect the rows in BigQuery:

```sql
SELECT * FROM `your-project-id.ml_explainability.heloc_batch_input` ORDER BY ExternalRiskEstimate DESC LIMIT 10;
```

You should see 10,000 rows from `instances.csv`.

![heloc_batch_input in BigQuery](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/step3-bq-input-v2.png)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/vertex-ai.png"> Step 4 — Train model

The `train()` method in `batch_explain.py` fits a scikit-learn **logistic regression** on `instances.csv`. We predict `probability` — the chance the borrower ends up 90+ days past due (`RiskPerformance` = Bad).

The `train` step fits the pipeline and uploads `model.joblib` and `train_metrics.png` to your bucket.

👨‍💻 `python batch_explain.py train --bucket-uri gs://your-bucket/vertex-batch-explain`

Confirm `model.joblib` landed in `gs://your-bucket/vertex-batch-explain/models/`.

![model.joblib in GCS](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/step4-train.png)

Check the printed train/test ROC AUC in the terminal. The script also saves `train_metrics.png` to the same folder — logistic regression **coefficients** on the left (which features push risk up or down) and the test-set **ROC curve** on the right.

![Logistic regression coefficients and ROC curve](https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/step4-roc.png)

# <img width="30" alt="image" src="https://github.com/04pallav/gcp-ml-tutorials/releases/download/readme-assets/vertex-ai.png"> Step 5 — Run batch explanation job

The `explain` step registers your model on Vertex AI, then starts a batch job that scores every row and writes feature attributions to BigQuery.

**`Model.upload()` — register the model**

Vertex loads `model.joblib` from GCS into Google's prebuilt sklearn container ([`sklearn-cpu.1-6`](https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers)) and attaches explanation settings. See [Configure feature-based explanations](https://cloud.google.com/vertex-ai/docs/explainable-ai/configuring-explanations-feature-based#explanation-metadatajson) and [`ExplanationMetadata` `Encoding`](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/ExplanationSpec#Encoding).

```python
model = aiplatform.Model.upload(
    display_name="heloc-batch-explain",  # name shown in the Vertex AI console
    artifact_uri=uri,  # GCS path to model.joblib from Step 4
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest",
    explanation_parameters=aiplatform.explain.ExplanationParameters(
        {"sampled_shapley_attribution": {"path_count": 10}},  # Sampled Shapley: 10 permutations per row
    ),
    explanation_metadata=aiplatform.explain.ExplanationMetadata(
        inputs={"input": {"index_feature_mapping": FEATURES, "encoding": "BAG_OF_FEATURES"}},
        outputs={"probability": {}},
    ),
)
```

**Why `BAG_OF_FEATURES` and `index_feature_mapping`?**

BigQuery stores named columns (`ExternalRiskEstimate`, `NumInqLast6M`, …), but Vertex batch prediction sends each row to the model as an **ordered list of numbers** — like `[55, 144, 58, …]` with no labels attached. Google's docs call this a *tensor*; that just means a list of values in a fixed order, not TensorFlow.

- **`encoding: "BAG_OF_FEATURES"`** — tells Vertex this input is a list of separate features packed together (see the [Encoding docs](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/ExplanationSpec#Encoding): *"each index maps to a feature"*).
- **`index_feature_mapping: FEATURES`** — the cheat sheet: position 0 → `ExternalRiskEstimate`, position 1 → `MSinceOldestTradeOpen`, and so on for all 22 features. Required for `BAG_OF_FEATURES` — same idea as Google's example `input = [27, 6.0, 150]` with `indexFeatureMapping = ["age", "height", "weight"]`.

Without the mapping, Vertex can score the row but attributions come back as "position 14" instead of `NumInqLast6M`.

**`batch_predict()` — score + explain every row**

```python
job = model.batch_predict(
    job_display_name="heloc-batch-explain-job",  # label in Vertex AI → Batch predictions
    instances_format="bigquery",
    bigquery_source="bq://your-project-id.ml_explainability.heloc_batch_input",
    predictions_format="bigquery",
    bigquery_destination_prefix="bq://your-project-id.ml_explainability.heloc_batch_explanations",
    generate_explanation=True,  # per-feature attributions, not just scores
    machine_type="n2-standard-4",  # VM size for each prediction replica
    batch_size=16,  # rows per request to the model server
    starting_replica_count=1,
    max_replica_count=1,  # 1 replica — tutorial scale
)
```

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
