"""FICO HELOC dataset (OpenML data_id=45023)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

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


def load_heloc():
    df = fetch_openml(data_id=45023, as_frame=True, parser="auto").frame
    X = df[FEATURE_NAMES].replace([-7, -8, -9], np.nan)
    y = (df["RiskPerformance"].astype(str) == "1").astype(int)
    return X, y
