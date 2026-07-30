#!/usr/bin/env python3
"""Train HELOC sklearn model and save model.joblib + feature_names.json."""

import argparse
import json
import os

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heloc_data import FEATURE_NAMES, load_heloc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--row-limit", type=int, default=None)
    args = parser.parse_args()

    X, y = load_heloc()
    if args.row_limit:
        X, _, y, _ = train_test_split(X, y, train_size=args.row_limit, random_state=42, stratify=y)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)

    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.output_dir, "model.joblib"))
    with open(os.path.join(args.output_dir, "feature_names.json"), "w") as f:
        json.dump({"feature_names": FEATURE_NAMES, "explanation_output_key": "probability"}, f, indent=2)

    print(f"Saved model to {args.output_dir}/", flush=True)


if __name__ == "__main__":
    main()
