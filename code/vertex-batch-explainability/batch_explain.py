#!/usr/bin/env python3
"""Vertex batch explainability — one script: BQ input, train, upload, batch explain."""

from __future__ import annotations

import argparse
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


def load_instances_to_bq(project_id: str, dataset: str, table: str, instances_path: str) -> str:
    with open(instances_path) as f:
        instances = json.load(f)["instances"]
    table_id = f"{project_id}.{dataset}.{table}"
    client = bigquery.Client(project=project_id)
    client.create_dataset(f"{project_id}.{dataset}", exists_ok=True)
    rows = [{name: float(row[name]) for name in FEATURE_NAMES} for row in instances]
    schema = [bigquery.SchemaField(name, "FLOAT") for name in FEATURE_NAMES]
    job = client.load_table_from_json(
        rows,
        table_id,
        job_config=bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ),
    )
    job.result()
    print(f"[bq] loaded {len(rows)} rows to {table_id}", flush=True)
    return table_id


def train_model(output_dir: str) -> None:
    X, y = load_heloc()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model.joblib"))
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump({"feature_names": FEATURE_NAMES, "explanation_output_key": "probability"}, f, indent=2)
    print(f"[train] saved model to {output_dir}", flush=True)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    bucket, _, prefix = uri.removeprefix("gs://").partition("/")
    return bucket, prefix.rstrip("/")


def upload_dir_to_gcs(local_dir: str, bucket_uri: str, subdir: str) -> str:
    bucket_name, prefix = parse_gcs_uri(bucket_uri)
    dest_prefix = f"{prefix}/{subdir}" if prefix else subdir
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for name in os.listdir(local_dir):
        path = os.path.join(local_dir, name)
        if os.path.isfile(path):
            blob = bucket.blob(f"{dest_prefix}/{name}")
            blob.upload_from_filename(path)
            print(f"[gcs] gs://{bucket_name}/{dest_prefix}/{name}", flush=True)
    return f"gs://{bucket_name}/{dest_prefix}/"


def upload_explainable_model(project_id: str, region: str, artifact_uri: str, display_name: str):
    explanation_parameters = aiplatform.explain.ExplanationParameters({
        "sampled_shapley_attribution": {"path_count": 10},
    })
    explanation_metadata = aiplatform.explain.ExplanationMetadata(
        inputs={name: {} for name in FEATURE_NAMES},
        outputs={"probability": {}},
    )
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=SERVING_CONTAINER,
        explanation_parameters=explanation_parameters,
        explanation_metadata=explanation_metadata,
    )
    print(f"[vertex] model: {model.resource_name}", flush=True)
    return model


def run_batch_job(model, *, project_id: str, dataset: str, input_table: str, output_table: str,
                  job_name: str, machine_type: str, batch_size: int) -> None:
    bq_input = f"{project_id}.{dataset}.{input_table}"
    bq_output = f"{project_id}.{dataset}.{output_table}"
    print(f"[vertex] batch input:  {bq_input}", flush=True)
    print(f"[vertex] batch output: {bq_output}", flush=True)
    job = model.batch_predict(
        job_display_name=job_name,
        instances_format="bigquery",
        bigquery_source=f"bq://{bq_input}",
        predictions_format="bigquery",
        bigquery_destination_prefix=f"bq://{bq_output}",
        generate_explanation=True,
        machine_type=machine_type,
        batch_size=batch_size,
        starting_replica_count=1,
        max_replica_count=1,
    )
    print(f"[vertex] batch job: {job.resource_name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--bucket-uri", required=True, help="e.g. gs://your-bucket/vertex-batch-explain")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--instances", default="instances.json")
    parser.add_argument("--dataset", default="ml_explainability")
    parser.add_argument("--input-table", default="heloc_batch_input")
    parser.add_argument("--output-table", default="heloc_batch_explanations")
    parser.add_argument("--machine-type", default="n2-standard-4")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    load_instances_to_bq(args.project_id, args.dataset, args.input_table, args.instances)

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "models")
        train_model(model_dir)
        artifact_uri = upload_dir_to_gcs(model_dir, args.bucket_uri, "models")

    model = upload_explainable_model(args.project_id, args.region, artifact_uri, "heloc-batch-explain")
    run_batch_job(
        model,
        project_id=args.project_id,
        dataset=args.dataset,
        input_table=args.input_table,
        output_table=args.output_table,
        job_name="heloc-batch-explain-job",
        machine_type=args.machine_type,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
