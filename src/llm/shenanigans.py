from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.datahandler.data_sanity_check import standardize_columns
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _safe_series(df: pd.DataFrame, key: str) -> pd.Series:
    if key in df.index:
        values = df.loc[key]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[0]
        return values.astype(float)
    return pd.Series(dtype=float)


def _load_statement(raw_dir: Path, ticker: str, statement: str) -> Optional[pd.DataFrame]:
    path = raw_dir / f"{ticker}_{statement}_annual.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df = standardize_columns(df, statement=statement)
    return df.apply(pd.to_numeric, errors="coerce")


def compute_red_flags(ticker: str, raw_dir: Path) -> Optional[Dict[str, object]]:
    is_df = _load_statement(raw_dir, ticker, "IS")
    bs_df = _load_statement(raw_dir, ticker, "BS")
    cf_df = _load_statement(raw_dir, ticker, "CF")
    if is_df is None or bs_df is None or is_df.empty or bs_df.empty:
        return None

    sales = _safe_series(is_df, "sales")
    net_income = _safe_series(is_df, "ni")
    ar = _safe_series(bs_df, "ar")
    inv = _safe_series(bs_df, "inv_stock")
    cfo = _safe_series(cf_df, "operating_cash_flow")

    df = pd.DataFrame(
        {
            "sales": sales,
            "net_income": net_income,
            "ar": ar,
            "inventory": inv,
            "cfo": cfo,
        }
    ).dropna(how="all")
    if df.shape[1] == 0 or df.shape[0] < 2:
        return None

    df = df.sort_index(axis=0)
    df["ar_sales"] = df["ar"] / df["sales"].replace(0, np.nan)
    df["inv_sales"] = df["inventory"] / df["sales"].replace(0, np.nan)
    df["cfo_to_ni"] = df["cfo"] / df["net_income"].replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    def _delta_ratio(series: pd.Series) -> float:
        if series.dropna().shape[0] < 2:
            return 0.0
        return float(series.iloc[-1] - series.iloc[-2])

    flags = {
        "ar_spike": bool(_delta_ratio(df["ar_sales"]) > 0.15),
        "inventory_spike": bool(_delta_ratio(df["inv_sales"]) > 0.15),
        "cfo_vs_ni_gap": bool((df["cfo_to_ni"].iloc[-1] < 0.5) if df["cfo_to_ni"].shape[0] else False),
    }
    return {
        "ticker": ticker,
        "latest_period": str(df.index[-1]),
        "metrics": {
            "ar_sales": float(df["ar_sales"].iloc[-1]) if df["ar_sales"].shape[0] else None,
            "inv_sales": float(df["inv_sales"].iloc[-1]) if df["inv_sales"].shape[0] else None,
            "cfo_to_ni": float(df["cfo_to_ni"].iloc[-1]) if df["cfo_to_ni"].shape[0] else None,
        },
        "flags": flags,
    }


def run_shenanigans_scan(tickers: Sequence[str], raw_dir: Path) -> List[Dict[str, object]]:
    results = []
    for tk in tickers:
        try:
            item = compute_red_flags(tk, raw_dir)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", tk, exc)
            item = None
        if item:
            results.append(item)
    return results
