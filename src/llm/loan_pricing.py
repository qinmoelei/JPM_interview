from __future__ import annotations

"""Loan spread pricing, resale forecast, and interval estimation."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


# LendingClub is used as a public proxy for term loan pricing.
LENDING_CLUB_URL = "https://resources.lendingclub.com/LoanStats3a.csv.zip"
FRED_DGS2 = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2"
FRED_DGS5 = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS5"


def download_lending_club(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    import urllib.request

    LOGGER.info("Downloading LendingClub data...")
    urllib.request.urlretrieve(LENDING_CLUB_URL, out_path)
    return out_path


def _load_lending_club(zip_path: Path, nrows: int = 40000) -> pd.DataFrame:
    df = pd.read_csv(zip_path, compression="zip", skiprows=1, low_memory=False, nrows=nrows)
    df = df.rename(columns=lambda c: c.strip())
    return df


def _load_fred_yields() -> pd.DataFrame:
    dgs2 = pd.read_csv(FRED_DGS2)
    dgs5 = pd.read_csv(FRED_DGS5)
    date_col = "observation_date"
    dgs2[date_col] = pd.to_datetime(dgs2[date_col])
    dgs5[date_col] = pd.to_datetime(dgs5[date_col])
    df = pd.merge(dgs2, dgs5, on=date_col, how="outer").sort_values(date_col)
    df = df.rename(columns={date_col: "DATE", "DGS2": "y2", "DGS5": "y5"}).ffill()
    return df


def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float) / 100.0
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)
    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    cols = [
        "int_rate",
        "term_months",
        "loan_amnt",
        "annual_inc",
        "dti",
        "grade",
        "sub_grade",
        "home_ownership",
        "purpose",
        "issue_date",
    ]
    return df[cols].dropna()


def _attach_yields(df: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    # Join Treasury yields to compute loan spreads.
    df = df.copy()
    df = pd.merge_asof(
        df.sort_values("issue_date"),
        yields.sort_values("DATE"),
        left_on="issue_date",
        right_on="DATE",
        direction="backward",
    )
    df["treasury_yield"] = np.where(df["term_months"] <= 36, df["y2"], df["y5"]) / 100.0
    df["spread"] = df["int_rate"] - df["treasury_yield"]
    return df


def train_spread_models(df: pd.DataFrame) -> Dict[str, object]:
    features = df.drop(columns=["spread", "issue_date", "DATE", "y2", "y5"])
    features = pd.get_dummies(features, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        features, df["spread"], test_size=0.25, random_state=2024
    )
    lin = Ridge(alpha=1.0)
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_rmse = float(np.sqrt(mean_squared_error(y_test, lin_pred)))
    lin_mae = mean_absolute_error(y_test, lin_pred)

    gbdt = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08)
    gbdt.fit(X_train, y_train)
    gbdt_pred = gbdt.predict(X_test)
    gbdt_rmse = float(np.sqrt(mean_squared_error(y_test, gbdt_pred)))
    gbdt_mae = mean_absolute_error(y_test, gbdt_pred)

    feature_importance = {}
    if hasattr(gbdt, "feature_importances_"):
        feature_importance = dict(zip(features.columns, gbdt.feature_importances_.tolist()))

    return {
        "linear": {"rmse": float(lin_rmse), "mae": float(lin_mae)},
        "gbdt": {"rmse": float(gbdt_rmse), "mae": float(gbdt_mae)},
        "feature_importance": feature_importance,
        "models": {"linear": lin, "gbdt": gbdt},
        "feature_columns": list(features.columns),
    }


def evaluate_private_borrower(df: pd.DataFrame) -> Dict[str, float]:
    features = df.drop(columns=["spread", "issue_date", "DATE", "y2", "y5"])
    full = pd.get_dummies(features, drop_first=True)
    no_market = features.drop(columns=["grade", "sub_grade"])
    no_market = pd.get_dummies(no_market, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(full, df["spread"], test_size=0.25, random_state=2024)
    X_train_nm, X_test_nm, _, _ = train_test_split(no_market, df["spread"], test_size=0.25, random_state=2024)

    lin = Ridge(alpha=1.0).fit(X_train, y_train)
    lin_nm = Ridge(alpha=1.0).fit(X_train_nm, y_train)
    mae_full = mean_absolute_error(y_test, lin.predict(X_test))
    mae_nm = mean_absolute_error(y_test, lin_nm.predict(X_test_nm))
    return {"mae_full": float(mae_full), "mae_no_market": float(mae_nm)}


def build_resale_dataset(df: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    # Approximate 1m resale return using yield changes and duration proxy.
    df = df.copy()
    df["next_month"] = df["issue_date"] + pd.offsets.MonthEnd(1)
    next_yields = pd.merge_asof(
        df[["next_month"]].sort_values("next_month"),
        yields.sort_values("DATE"),
        left_on="next_month",
        right_on="DATE",
        direction="backward",
    )
    y_now = np.where(df["term_months"] <= 36, df["y2"], df["y5"]) / 100.0
    y_next = np.where(df["term_months"] <= 36, next_yields["y2"], next_yields["y5"]) / 100.0
    df["delta_yield"] = (y_next - y_now).astype(float)
    df["return_1m"] = df["int_rate"] / 12.0 - df["delta_yield"] * (df["term_months"] / 12.0) * 0.5
    df["price_1m"] = 100.0 * (1.0 + df["return_1m"])
    return df


def train_resale_model(df: pd.DataFrame) -> Dict[str, object]:
    features = df.drop(columns=["return_1m", "price_1m", "issue_date", "DATE", "y2", "y5", "next_month"])
    features = pd.get_dummies(features, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        features, df["return_1m"], test_size=0.25, random_state=2024
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = mean_absolute_error(y_test, preds)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "model": model,
        "feature_columns": list(features.columns),
        "y_test": y_test.to_numpy(),
        "preds": preds,
    }


def conformal_interval(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.05) -> float:
    # Simple conformal half-width based on absolute residual quantile.
    residuals = np.abs(y_true - y_pred)
    return float(np.quantile(residuals, 1 - alpha))


def evaluate_interval(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> Dict[str, float]:
    lower = y_pred - q
    upper = y_pred + q
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    avg_width = float(np.mean(upper - lower))
    return {"coverage": float(coverage), "avg_width": avg_width}


def run_loan_pricing_pipeline(out_dir: Path, nrows: int = 40000) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_lending_club(out_dir / "LoanStats3a.csv.zip")
    raw = _load_lending_club(zip_path, nrows=nrows)
    yields = _load_fred_yields()
    base = _attach_yields(_prep_features(raw), yields)

    spread_metrics = train_spread_models(base)
    spread_metrics_out = {k: v for k, v in spread_metrics.items() if k not in {"models"}}
    private_gap = evaluate_private_borrower(base)

    resale_df = build_resale_dataset(base, yields)
    resale_metrics = train_resale_model(resale_df)
    q = conformal_interval(resale_metrics["y_test"], resale_metrics["preds"])
    interval_report = evaluate_interval(resale_metrics["y_test"], resale_metrics["preds"], q)
    resale_metrics_out = {"rmse": resale_metrics["rmse"], "mae": resale_metrics["mae"]}

    output = {
        "spread_model_metrics": spread_metrics_out,
        "private_borrower_gap": private_gap,
        "resale_metrics": resale_metrics_out,
        "pi_coverage": interval_report,
        "pi_half_width": q,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "spread_model_metrics.json").write_text(json.dumps(spread_metrics_out, indent=2))
    (out_dir / "private_borrower_gap.json").write_text(json.dumps(private_gap, indent=2))
    (out_dir / "return_1m_metrics.json").write_text(json.dumps(resale_metrics_out, indent=2))
    (out_dir / "pi_coverage_report.json").write_text(json.dumps(interval_report, indent=2))
    return output
