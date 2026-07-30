#!/usr/bin/env python3
"""Vertex batch explainability — BQ input, train, upload, batch explain."""

import argparse
import csv
import io
import json
import os
import tempfile

import joblib
from google.cloud import aiplatform, bigquery, storage
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heloc_data import FEATURE_NAMES, load_heloc

SERVING_CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest"
DATASET = "ml_explainability"
INPUT_TABLE = "heloc_batch_input"
OUTPUT_TABLE = "heloc_batch_explanations"

# Command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--project-id", required=True)
parser.add_argument("--bucket-uri", required=True, help="e.g. gs://your-bucket/vertex-batch-explain")
parser.add_argument("--region", default="us-central1")
parser.add_argument("--instances", default="instances.csv", help="local path or gs:// URI")
args = parser.parse_args()

aiplatform.init(project=args.project_id, location=args.region)

# Read instances CSV
if args.instances.startswith("gs://"):
    bucket_name, _, blob_name = args.instances.removeprefix("gs://").partition("/")
    raw = storage.Client().bucket(bucket_name).blob(blob_name).download_as_text()
    reader = csv.DictReader(io.StringIO(raw))
else:
    with open(args.instances, newline="") as f:
        reader = csv.DictReader(f)
rows = [{name: float(row[name]) for name in FEATURE_NAMES} for row in reader]

# Load input to BigQuery
table_id = f"{args.project_id}.{DATASET}.{INPUT_TABLE}"
bq = bigquery.Client(project=args.project_id)
bq.create_dataset(f"{args.project_id}.{DATASET}", exists_ok=True)
bq.load_table_from_json(
    rows,
    table_id,
    job_config=bigquery.LoadJobConfig(
        schema=[bigquery.SchemaField(name, "FLOAT") for name in FEATURE_NAMES],
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    ),
).result()
print(f"Loaded {len(rows)} rows to {table_id}")

# Train HELOC model
X, y = load_heloc()
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

with tempfile.TemporaryDirectory() as tmp:
    model_dir = os.path.join(tmp, "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "feature_names.json"), "w") as f:
        json.dump({"feature_names": FEATURE_NAMES, "explanation_output_key": "probability"}, f)

    # Upload model artifacts to GCS
    bucket_name, _, prefix = args.bucket_uri.removeprefix("gs://").partition("/")
    prefix = f"{prefix.rstrip('/')}/models" if prefix else "models"
    bucket = storage.Client().bucket(bucket_name)
    for name in os.listdir(model_dir):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            blob = bucket.blob(f"{prefix}/{name}")
            blob.upload_from_filename(path)
            print(f"Uploaded gs://{bucket_name}/{prefix}/{name}")

artifact_uri = f"gs://{bucket_name}/{prefix}/"

# Register explainable model on Vertex AI
vertex_model = aiplatform.Model.upload(
    display_name="heloc-batch-explain",
    artifact_uri=artifact_uri,
    serving_container_image_uri=SERVING_CONTAINER,
    explanation_parameters=aiplatform.explain.ExplanationParameters(
        {"sampled_shapley_attribution": {"path_count": 10}},
    ),
    explanation_metadata=aiplatform.explain.ExplanationMetadata(
        inputs={name: {} for name in FEATURE_NAMES},
        outputs={"probability": {}},
    ),
)
print(f"Model registered: {vertex_model.resource_name}")

# Batch prediction with explanations
bq_input = f"bq://{args.project_id}.{DATASET}.{INPUT_TABLE}"
bq_output = f"bq://{args.project_id}.{DATASET}.{OUTPUT_TABLE}"
job = vertex_model.batch_predict(
    job_display_name="heloc-batch-explain-job",
    instances_format="bigquery",
    bigquery_source=bq_input,
    predictions_format="bigquery",
    bigquery_destination_prefix=bq_output,
    generate_explanation=True,
    machine_type="n2-standard-4",
    batch_size=16,
    starting_replica_count=1,
    max_replica_count=1,
)
print(f"Batch job started: {job.resource_name}")
