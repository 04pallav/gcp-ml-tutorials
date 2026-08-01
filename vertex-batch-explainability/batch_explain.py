#!/usr/bin/env python3
"""Vertex batch explainability — load, train, explain (or `all`)."""

import argparse
import csv
import io
import os
import tempfile
from math import isnan

import google.auth
import joblib
from google.cloud import aiplatform, bigquery, storage
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 22 credit-bureau features from the FICO HELOC dataset (OpenML 45023).
FEATURES = [
    "ExternalRiskEstimate",
    "MSinceOldestTradeOpen",
    "MSinceMostRecentTradeOpen",
    "AverageMInFile",
    "NumSatisfactoryTrades",
    "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec",
    "PercentTradesNeverDelq",
    "MSinceMostRecentDelq",
    "MaxDelq2PublicRecLast12M",
    "NumTotalTrades",
    "NumTradesOpeninLast12M",
    "PercentInstallTrades",
    "MSinceMostRecentInqexcl7days",
    "NumInqLast6M",
    "NumInqLast6Mexcl7days",
    "NetFractionRevolvingBurden",
    "NetFractionInstallBurden",
    "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance",
    "NumBank2NatlTradesWHighUtilization",
    "PercentTradesWBalance",
]
TARGET = "RiskPerformance"  # 1 = bad (90+ days past due), 0 = good
# Blank cells in instances.csv; FICO raw data also uses -7/-8/-9 (see openml.org/d/45023).
MISSING = {"", "-7", "-8", "-9"}
CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest"
MODEL_NAME = "heloc-batch-explain"
DATASET = "ml_explainability"
INPUT_TABLE = "heloc_batch_input"
OUTPUT_TABLE = "heloc_batch_explanations"


def parse_float(raw: str) -> float:
    """CSV values are strings; map missing sentinels to NaN for sklearn imputation."""
    return float("nan") if raw in MISSING else float(raw)


def read_csv(path: str) -> list[dict[str, str]]:
    """Read instances.csv from a local path or gs:// URI (Step 3 uses GCS)."""
    if path.startswith("gs://"):
        bucket, _, blob = path.removeprefix("gs://").partition("/")
        text = storage.Client().bucket(bucket).blob(blob).download_as_text()
        return list(csv.DictReader(io.StringIO(text)))
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def model_uri(bucket_uri: str) -> tuple[str, str]:
    """Turn gs://your-bucket/vertex-batch-explain into bucket name + models/ artifact URI."""
    bucket, _, prefix = bucket_uri.removeprefix("gs://").partition("/")
    prefix = f"{prefix.rstrip('/')}/models" if prefix else "models"
    return bucket, f"gs://{bucket}/{prefix}/"


def load(args) -> None:
    """Step 3 — write feature columns to BigQuery (label stays in CSV, not loaded)."""
    rows = [
        # NaN → NULL in BigQuery; Vertex batch job reads this table later.
        {n: (None if isnan(v := parse_float(r[n])) else v) for n in FEATURES}
        for r in read_csv(args.instances)
    ]
    table = f"{args.project_id}.{DATASET}.{INPUT_TABLE}"
    bq = bigquery.Client(project=args.project_id)
    bq.create_dataset(f"{args.project_id}.{DATASET}", exists_ok=True)
    bq.load_table_from_json(
        rows,
        table,
        job_config=bigquery.LoadJobConfig(
            schema=[bigquery.SchemaField(n, "FLOAT") for n in FEATURES],
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ),
    ).result()
    print(f"[load] {len(rows)} rows → {table}", flush=True)


def train(args) -> None:
    """Step 4 — fit logistic regression and upload model.joblib to GCS."""
    rows = read_csv(args.instances)
    X = [[parse_float(r[n]) for n in FEATURES] for r in rows]  # feature matrix
    y = [int(r[TARGET].strip() == "1") for r in rows]  # 1 = bad, 0 = good
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # fill NaN from blank cells
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]).fit(X_train, y_train)
    print(
        f"[train] ROC AUC train={roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]):.3f} "
        f"test={roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.3f}",
        flush=True,
    )

    bucket, uri = model_uri(args.bucket_uri)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.joblib")
        joblib.dump(model, path)
        storage.Client().bucket(bucket).blob("models/model.joblib").upload_from_filename(path)
    print(f"[train] uploaded {uri}model.joblib", flush=True)


def explain(args) -> None:
    """Step 5 — register model on Vertex AI and run batch prediction with explanations."""
    _, uri = model_uri(args.bucket_uri)
    models = list(aiplatform.Model.list(
        filter=f'display_name="{MODEL_NAME}"', order_by="create_time desc"
    ))
    if models:
        model = models[0]
        print(f"[explain] reusing {model.resource_name}", flush=True)
    else:
        model = aiplatform.Model.upload(
            display_name=MODEL_NAME,
            artifact_uri=uri,
            serving_container_image_uri=CONTAINER,
            explanation_parameters=aiplatform.explain.ExplanationParameters(
                {"sampled_shapley_attribution": {"path_count": 10}},
            ),
            # BAG_OF_FEATURES: BigQuery rows are 22 unnamed floats; map index → feature name.
            explanation_metadata=aiplatform.explain.ExplanationMetadata(
                inputs={"input": {"index_feature_mapping": FEATURES, "encoding": "BAG_OF_FEATURES"}},
                outputs={"probability": {}},
            ),
        )
        print(f"[explain] registered {model.resource_name}", flush=True)

    job = model.batch_predict(
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
    print(f"[explain] job started: {job.resource_name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["load", "train", "explain", "all"])
    parser.add_argument("--project-id")
    parser.add_argument("--bucket-uri")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--instances", default="instances.csv")
    args = parser.parse_args()
    # Picks up project from Cloud Shell after: gcloud config set project ...
    _, default_project = google.auth.default()
    args.project_id = args.project_id or default_project
    if not args.project_id:
        parser.error("Pass --project-id or run: gcloud config set project YOUR_PROJECT_ID")
    if args.step in ("train", "explain", "all") and not args.bucket_uri:
        parser.error("--bucket-uri required for train/explain/all")

    aiplatform.init(project=args.project_id, location=args.region)
    steps = {"load": load, "train": train, "explain": explain}
    for fn in steps.values() if args.step == "all" else [steps[args.step]]:
        fn(args)


if __name__ == "__main__":
    main()
