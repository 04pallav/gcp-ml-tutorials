#!/usr/bin/env python3
"""Vertex batch explainability — run one step at a time so you can inspect between.

Steps:
  load     read instances.csv → BigQuery table heloc_batch_input
  train    fit the scikit-learn pipeline on heloc.csv → upload model to GCS
  explain  register the model on Vertex AI and start a batch prediction job

Run `all` to do everything in sequence.
"""

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heloc_data import FEATURE_NAMES, feature_value, load_heloc

SERVING_CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest"
MODEL_DISPLAY_NAME = "heloc-batch-explain"
DATASET = "ml_explainability"
INPUT_TABLE = "heloc_batch_input"
OUTPUT_TABLE = "heloc_batch_explanations"
MODEL_PREFIX = "models"


def models_uri(bucket_uri: str) -> tuple[str, str]:
    """Return (bucket_name, gs://.../models/) for the model artifacts."""
    bucket_name, _, prefix = bucket_uri.removeprefix("gs://").partition("/")
    prefix = f"{prefix.rstrip('/')}/{MODEL_PREFIX}" if prefix else MODEL_PREFIX
    return bucket_name, f"gs://{bucket_name}/{prefix}/"


def load(args) -> None:
    """Step 1 — load instances.csv into BigQuery."""
    print(f"[load] reading instances from {args.instances}", flush=True)
    if args.instances.startswith("gs://"):
        bucket_name, _, blob_name = args.instances.removeprefix("gs://").partition("/")
        raw = storage.Client().bucket(bucket_name).blob(blob_name).download_as_text()
        reader = csv.DictReader(io.StringIO(raw))
    else:
        with open(args.instances, newline="") as f:
            reader = list(csv.DictReader(f))
    rows = [
        {name: (None if isnan(v := feature_value(row[name])) else v) for name in FEATURE_NAMES}
        for row in reader
    ]

    table_id = f"{args.project_id}.{DATASET}.{INPUT_TABLE}"
    bq = bigquery.Client(project=args.project_id)
    bq.create_dataset(f"{args.project_id}.{DATASET}", exists_ok=True)
    print(f"[load] writing {len(rows)} rows to {table_id}", flush=True)
    bq.load_table_from_json(
        rows,
        table_id,
        job_config=bigquery.LoadJobConfig(
            schema=[bigquery.SchemaField(name, "FLOAT") for name in FEATURE_NAMES],
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ),
    ).result()
    print(f"[load] done — inspect it with: SELECT * FROM `{table_id}` LIMIT 10", flush=True)


def train(args) -> None:
    """Step 2 — train the pipeline and upload the model to GCS."""
    print("[train] loading heloc.csv and fitting pipeline", flush=True)
    X, y = load_heloc()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, [p[1] for p in model.predict_proba(X_train)])
    test_auc = roc_auc_score(y_test, [p[1] for p in model.predict_proba(X_test)])
    print(f"[train] ROC AUC — train={train_auc:.3f} test={test_auc:.3f}", flush=True)

    bucket_name, artifact_uri = models_uri(args.bucket_uri)
    with tempfile.TemporaryDirectory() as tmp:
        joblib.dump(model, os.path.join(tmp, "model.joblib"))
        with open(os.path.join(tmp, "feature_names.json"), "w") as f:
            json.dump({"feature_names": FEATURE_NAMES, "explanation_output_key": "probability"}, f)
        bucket = storage.Client().bucket(bucket_name)
        for name in ("model.joblib", "feature_names.json"):
            bucket.blob(f"{MODEL_PREFIX}/{name}").upload_from_filename(os.path.join(tmp, name))
            print(f"[train] uploaded {artifact_uri}{name}", flush=True)
    print(f"[train] done — model artifacts at {artifact_uri}", flush=True)


def explain(args) -> None:
    """Step 3 — register the model on Vertex AI and start a batch job."""
    _, artifact_uri = models_uri(args.bucket_uri)

    existing = list(aiplatform.Model.list(
        filter=f'display_name="{MODEL_DISPLAY_NAME}"',
        order_by="create_time desc",
    ))
    if existing:
        vertex_model = existing[0]
        print(f"[explain] reusing model {vertex_model.resource_name}", flush=True)
    else:
        print(f"[explain] registering model from {artifact_uri}", flush=True)
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
        print(f"[explain] registered {vertex_model.resource_name}", flush=True)

    print(f"[explain] starting batch prediction: {INPUT_TABLE} → {OUTPUT_TABLE}", flush=True)
    job = vertex_model.batch_predict(
        job_display_name="heloc-batch-explain-job",
        instances_format="bigquery",
        bigquery_source=f"bq://{args.project_id}.{DATASET}.{INPUT_TABLE}",
        predictions_format="bigquery",
        bigquery_destination_prefix=f"bq://{args.project_id}.{DATASET}.{OUTPUT_TABLE}",
        generate_explanation=True,
        machine_type="n2-standard-4",
        batch_size=16,
        starting_replica_count=1,
        max_replica_count=1,
    )
    print(f"[explain] batch job started: {job.resource_name}", flush=True)
    print("[explain] watch it in Vertex AI → Batch predictions (wait for JOB_STATE_SUCCEEDED)", flush=True)


STEPS = {"load": load, "train": train, "explain": explain}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("step", choices=[*STEPS, "all"], help="which step to run")
parser.add_argument("--project-id", required=True)
parser.add_argument("--bucket-uri", required=True, help="e.g. gs://your-bucket/vertex-batch-explain")
parser.add_argument("--region", default="us-central1")
parser.add_argument("--instances", default="instances.csv", help="local path or gs:// URI")
args = parser.parse_args()

aiplatform.init(project=args.project_id, location=args.region)

if args.step == "all":
    for fn in STEPS.values():
        fn(args)
else:
    STEPS[args.step](args)
