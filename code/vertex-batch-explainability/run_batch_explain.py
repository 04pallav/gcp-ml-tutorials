#!/usr/bin/env python3
"""Train, upload explainable model, and run Vertex batch predict with explanations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import joblib
from google.cloud import aiplatform, storage
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heloc_data import FEATURE_NAMES, load_heloc


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


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
    print(f"[train] saved artifacts to {output_dir}", flush=True)


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


def upload_explainable_model(cfg: dict, artifact_uri: str):
    explanation_parameters = aiplatform.explain.ExplanationParameters({
        "sampled_shapley_attribution": {"path_count": 10},
    })
    explanation_metadata = aiplatform.explain.ExplanationMetadata(
        inputs={name: {} for name in FEATURE_NAMES},
        outputs={"probability": {}},
    )
    model = aiplatform.Model.upload(
        display_name=cfg["display_name"],
        artifact_uri=artifact_uri,
        serving_container_image_uri=cfg["serving_container"],
        explanation_parameters=explanation_parameters,
        explanation_metadata=explanation_metadata,
    )
    print(f"[vertex] model: {model.resource_name}", flush=True)
    return model


def run_batch_job(cfg: dict, model) -> None:
    input_table = f"{cfg['project_id']}.{cfg['bq_dataset']}.{cfg['bq_input_table']}"
    output_table = f"{cfg['project_id']}.{cfg['bq_dataset']}.{cfg['bq_output_table']}"
    print(f"[vertex] batch input:  {input_table}", flush=True)
    print(f"[vertex] batch output: {output_table}", flush=True)
    job = model.batch_predict(
        job_display_name=cfg["job_name"],
        instances_format="bigquery",
        bigquery_source=f"bq://{input_table}",
        predictions_format="bigquery",
        bigquery_destination_prefix=f"bq://{output_table}",
        generate_explanation=True,
        machine_type=cfg.get("machine_type", "n2-standard-4"),
        batch_size=cfg.get("batch_size", 16),
        starting_replica_count=1,
        max_replica_count=1,
    )
    print(f"[vertex] batch job: {job.resource_name}", flush=True)


def model_artifact_uri(cfg: dict) -> str:
    bucket, prefix = parse_gcs_uri(cfg["staging_bucket_uri"])
    model_prefix = f"{prefix}/models" if prefix else "models"
    return f"gs://{bucket}/{model_prefix}/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--skip-train", action="store_true", help="Reuse model.joblib already on GCS")
    args = parser.parse_args()

    cfg = load_config(args.config)
    aiplatform.init(project=cfg["project_id"], location=cfg["region"])

    if args.skip_train:
        artifact_uri = model_artifact_uri(cfg)
        print(f"[vertex] reusing artifacts at {artifact_uri}", flush=True)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "models")
            train_model(model_dir)
            artifact_uri = upload_dir_to_gcs(model_dir, cfg["staging_bucket_uri"], "models")

    model = upload_explainable_model(cfg, artifact_uri)
    run_batch_job(cfg, model)


if __name__ == "__main__":
    main()
