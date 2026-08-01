"""FICO HELOC dataset (OpenML data_id=45023), vendored as heloc.csv."""

from __future__ import annotations

import csv
from math import nan
from pathlib import Path

FEATURE_NAMES = [
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

TARGET_COLUMN = "RiskPerformance"
DATA_PATH = Path(__file__).with_name("heloc.csv")
_MISSING = {"", "-7", "-8", "-9"}


def feature_value(raw: str) -> float:
    if raw in _MISSING:
        return nan
    return float(raw)


def load_heloc() -> tuple[list[list[float]], list[int]]:
    features: list[list[float]] = []
    labels: list[int] = []
    with DATA_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            features.append([feature_value(row[name]) for name in FEATURE_NAMES])
            labels.append(1 if row[TARGET_COLUMN].strip() == "1" else 0)
    return features, labels
