"""FICO HELOC dataset (OpenML data_id=45023), vendored as heloc.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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


def load_heloc() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES].replace([-7, -8, -9], np.nan)
    y = (df[TARGET_COLUMN].astype(str) == "1").astype(int)
    return X, y
