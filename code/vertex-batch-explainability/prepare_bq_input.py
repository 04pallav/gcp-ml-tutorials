#!/usr/bin/env python3
"""Load instances.json into the BigQuery input table."""

import argparse
import json

from google.cloud import bigquery

from heloc_data import FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dataset", default="ml_explainability")
    parser.add_argument("--table", default="heloc_batch_input")
    parser.add_argument("--instances", default="instances.json")
    args = parser.parse_args()

    with open(args.instances) as f:
        instances = json.load(f)["instances"]

    table_id = f"{args.project_id}.{args.dataset}.{args.table}"
    client = bigquery.Client(project=args.project_id)
    client.create_dataset(f"{args.project_id}.{args.dataset}", exists_ok=True)

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
    print(f"Loaded {len(rows)} rows to {table_id}", flush=True)


if __name__ == "__main__":
    main()
