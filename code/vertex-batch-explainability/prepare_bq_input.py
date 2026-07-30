#!/usr/bin/env python3
"""Load instances.json into the BigQuery input table."""

import argparse
import json

from google.cloud import bigquery

from heloc_data import FEATURE_NAMES


def load_config(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--instances", default="instances.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    with open(args.instances) as f:
        instances = json.load(f)["instances"]

    project_id = cfg["project_id"]
    dataset = cfg["bq_dataset"]
    table = cfg["bq_input_table"]
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
    print(f"Loaded {len(rows)} rows to {table_id}", flush=True)


if __name__ == "__main__":
    main()
