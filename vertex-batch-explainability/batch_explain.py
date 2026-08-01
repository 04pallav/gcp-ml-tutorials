#!/usr/bin/env python3
"""Vertex batch explainability — BQ input, train, upload, batch explain."""

import argparse
import csv
import io
import json
import os
import tempfile
from math import isnan

import joblib
from google.cloud import aiplatform, bigquery, storage
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heloc_data import FEATURE_NAMES, feature_value, load_heloc

SERVING_CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest"
MODEL_DISPLAY_NAME = "heloc-batch-explain"
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
print(f"[1] project={args.project_id} bucket={args.bucket_uri} instances={args.instances}", flush=True)

# Read instances CSV
print(f"[2] Reading instances from {args.instances}", flush=True)
if args.instances.startswith("gs://"):
    bucket_name, _, blob_name = args.instances.removeprefix("gs://").partition("/")
    raw = storage.Client().bucket(bucket_name).blob(blob_name).download_as_text()
    reader = csv.DictReader(io.StringIO(raw))
else:
    with open(args.instances, newline="") as f:
        reader = csv.DictReader(f)
rows = [
    {name: (None if isnan(v := feature_value(row[name])) else v) for name in FEATURE_NAMES}
    for row in reader
]

# Load input to BigQuery
table_id = f"{args.project_id}.{DATASET}.{INPUT_TABLE}"
bq = bigquery.Client(project=args.project_id)
bq.create_dataset(f"{args.project_id}.{DATASET}", exists_ok=True)
print(f"[2] Loading {len(rows)} rows to BigQuery table {INPUT_TABLE}", flush=True)
bq.load_table_from_json(
    rows,
    table_id,
    job_config=bigquery.LoadJobConfig(
        schema=[bigquery.SchemaField(name, "FLOAT") for name in FEATURE_NAMES],
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    ),
).result()
print(f"[2] Done — {table_id}", flush=True)

existing_models = list(aiplatform.Model.list(
    filter=f'display_name="{MODEL_DISPLAY_NAME}"',
    order_by="create_time desc",
))
if existing_models:
    vertex_model = existing_models[0]
    print(f"[5] Reusing existing model — {vertex_model.resource_name}", flush=True)
else:
    # Train HELOC model
    print("[3] Training scikit-learn pipeline on heloc.csv", flush=True)
    X, y = load_heloc()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    print("[3] Done — model trained", flush=True)

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
        print(f"[4] Uploading model artifacts to gs://{bucket_name}/{prefix}/", flush=True)
        for name in os.listdir(model_dir):
            path = os.path.join(model_dir, name)
            if os.path.isfile(path):
                blob = bucket.blob(f"{prefix}/{name}")
                blob.upload_from_filename(path)
                print(f"[4] Uploaded gs://{bucket_name}/{prefix}/{name}", flush=True)

        artifact_uri = f"gs://{bucket_name}/{prefix}/"

    # Register explainable model on Vertex AI
    print("[5] Registering model on Vertex AI with explanation settings", flush=True)
    vertex_model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
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
    print(f"[5] Done — {vertex_model.resource_name}", flush=True)

# Batch prediction with explanations
bq_input = f"bq://{args.project_id}.{DATASET}.{INPUT_TABLE}"
bq_output = f"bq://{args.project_id}.{DATASET}.{OUTPUT_TABLE}"
print(f"[6] Starting batch prediction: {INPUT_TABLE} → {OUTPUT_TABLE}", flush=True)
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
print(f"[7] Batch job started: {job.resource_name}", flush=True)
print(f"[7] Model: {vertex_model.resource_name}", flush=True)
