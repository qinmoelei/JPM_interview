from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.llm.pdf_extract import ExtractConfig, extract_from_pdf
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


RATING_BUCKETS = [
    ("AAA", 3.0),
    ("AA", 2.6),
    ("A", 2.2),
    ("BBB", 1.8),
    ("BB", 1.4),
    ("B", 1.0),
    ("CCC", -np.inf),
]


def _rating_from_score(score: float) -> str:
    for label, cutoff in RATING_BUCKETS:
        if score >= cutoff:
            return label
    return "CCC"


def _compute_features(states: pd.DataFrame) -> pd.DataFrame:
    sales = states["sales"]
    cogs = states["cogs"]
    sga = states["sga"]
    dep = states["dep"]
    interest = states["interest_expense"]
    cash = states["cash"]
    ar = states["ar"]
    inv = states["inventory"]
    ap = states["ap"]
    ppe = states["ppe"]
    debt = states["debt"]
    equity = states["equity"]

    total_assets = cash + ar + inv + ppe
    working_capital = cash + ar + inv - ap
    ebit = sales - cogs - sga - dep

    features = pd.DataFrame(index=states.index)
    features["wc_to_assets"] = working_capital / total_assets.replace(0, np.nan)
    features["ebit_to_assets"] = ebit / total_assets.replace(0, np.nan)
    features["debt_to_assets"] = debt / total_assets.replace(0, np.nan)
    features["equity_to_assets"] = equity / total_assets.replace(0, np.nan)
    features["sales_to_assets"] = sales / total_assets.replace(0, np.nan)
    features["interest_coverage"] = ebit / interest.replace(0, np.nan)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _z_score(features: pd.DataFrame) -> pd.Series:
    return (
        1.2 * features["wc_to_assets"]
        + 3.3 * features["ebit_to_assets"]
        + 0.6 * features["equity_to_assets"] / features["debt_to_assets"].replace(0, np.nan)
        + 1.0 * features["sales_to_assets"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_rating_dataset(proc_dir: Path, tickers: Sequence[str]) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        path = proc_dir / f"{tk}_states.csv"
        if not path.exists():
            continue
        states = pd.read_csv(path, index_col=0)
        features = _compute_features(states)
        z = _z_score(features)
        for idx in features.index:
            row = features.loc[idx].to_dict()
            row["ticker"] = tk
            row["period"] = idx
            row["z_score"] = float(z.loc[idx])
            row["rating_label"] = _rating_from_score(float(z.loc[idx]))
            rows.append(row)
    return pd.DataFrame(rows)


def train_rating_models(dataset: pd.DataFrame) -> Dict[str, object]:
    feature_cols = [
        "wc_to_assets",
        "ebit_to_assets",
        "debt_to_assets",
        "equity_to_assets",
        "sales_to_assets",
        "interest_coverage",
    ]
    X = dataset[feature_cols].to_numpy(dtype=float)
    y = dataset["rating_label"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2024, stratify=y)
    clf = LogisticRegression(max_iter=500, multi_class="multinomial")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    # Ordinal-style regression on numeric rating buckets
    rating_to_num = {label: i for i, (label, _) in enumerate(RATING_BUCKETS)}
    y_num = dataset["rating_label"].map(rating_to_num).to_numpy(dtype=float)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_num, test_size=0.25, random_state=2024)
    reg = LinearRegression()
    reg.fit(X_train2, y_train2)
    pred_num = reg.predict(X_test2)
    pred_labels = [list(rating_to_num.keys())[int(round(x))] if 0 <= x < len(rating_to_num) else "CCC" for x in pred_num]
    acc = accuracy_score(y_test2, [rating_to_num[p] if p in rating_to_num else 0 for p in pred_labels])

    return {
        "multiclass_accuracy": float(accuracy_score(y_test, preds)),
        "multiclass_report": report,
        "ordinal_accuracy": float(acc),
        "model": clf,
        "ordinal_model": reg,
        "feature_cols": feature_cols,
    }


def score_pdf_report(
    pdf_path: Path,
    company: str,
    *,
    model: Optional[LogisticRegression] = None,
    feature_cols: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    payload = extract_from_pdf(pdf_path, company, cfg=ExtractConfig())
    items = payload["items"]
    # Build a minimal feature row using extracted items
    revenue = items.get("revenue") or 0.0
    cogs = items.get("cogs") or 0.0
    sga = items.get("sga") or 0.0
    dep = items.get("depreciation") or 0.0
    interest = items.get("interest_expense") or 0.0
    cash = 0.0
    ar = 0.0
    inv = items.get("inventory") or 0.0
    ap = 0.0
    ppe = 0.0
    debt = items.get("total_debt") or 0.0
    equity = items.get("total_equity") or 0.0
    total_assets = items.get("total_assets") or 0.0
    current_assets = items.get("current_assets") or 0.0
    current_liabilities = items.get("current_liabilities") or 0.0

    working_capital = current_assets - current_liabilities
    ebit = revenue - cogs - sga - dep
    features = {
        "wc_to_assets": working_capital / total_assets if total_assets else 0.0,
        "ebit_to_assets": ebit / total_assets if total_assets else 0.0,
        "debt_to_assets": debt / total_assets if total_assets else 0.0,
        "equity_to_assets": equity / total_assets if total_assets else 0.0,
        "sales_to_assets": revenue / total_assets if total_assets else 0.0,
        "interest_coverage": ebit / interest if interest else 0.0,
    }
    z_score = _z_score(pd.DataFrame([features])).iloc[0]
    heuristic_rating = _rating_from_score(float(z_score))

    model_rating = None
    if model is not None and feature_cols is not None:
        X = np.array([[features[col] for col in feature_cols]], dtype=float)
        model_rating = str(model.predict(X)[0])

    return {
        "company": company,
        "features": features,
        "z_score": float(z_score),
        "heuristic_rating": heuristic_rating,
        "model_rating": model_rating or heuristic_rating,
        "extraction_meta": payload["meta"],
    }
