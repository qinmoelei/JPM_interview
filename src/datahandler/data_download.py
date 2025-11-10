from __future__ import annotations
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys 

sys.path.append('.')
from src.datahandler.data_sanity_check import standardize_columns
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)

def _import_yf():
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        raise RuntimeError("yfinance is required to download data. Install with `pip install yfinance`.") from e
    return yf

def fetch_financials_for_ticker(ticker: str, frequency: str="annual") -> Dict[str, pd.DataFrame]:
    """Download IS, BS, CF for a ticker using yfinance; return standardized DataFrames (columns=period end)."""
    yf = _import_yf()
    t = yf.Ticker(ticker)

    if frequency == "annual":
        is_df = t.financials            # Income Statement (annual)
        bs_df = t.balance_sheet         # Balance Sheet (annual)
        cf_df = t.cashflow              # Cash Flow Statement (annual)
    elif frequency == "quarterly":
        is_df = t.quarterly_financials
        bs_df = t.quarterly_balance_sheet
        cf_df = t.quarterly_cashflow
    else:
        raise ValueError("frequency must be 'annual' or 'quarterly'")

    # yfinance returns rows as line items, columns as period end. Ensure datetime index.
    def _fix(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        # Convert column names to pandas datetime if possible
        try:
            df.columns = pd.to_datetime(df.columns)
        except Exception:
            pass
        df.index = df.index.astype(str)
        return df

    is_df, bs_df, cf_df = map(_fix, [is_df, bs_df, cf_df])
    # Standardize column names for key items (best-effort across Yahoo variants)
    is_std = standardize_columns(is_df, statement="IS")
    bs_std = standardize_columns(bs_df, statement="BS")
    cf_std = standardize_columns(cf_df, statement="CF")
    return {"IS": is_std, "BS": bs_std, "CF": cf_std}

def save_financials(bundle: Dict[str, pd.DataFrame], out_dir: str, ticker: str, frequency: str) -> None:
    ensure_dir(out_dir)
    for k, df in bundle.items():
        path = os.path.join(out_dir, f"{ticker}_{k}_{frequency}.csv")
        df.to_csv(path)

def download_universe(tickers: List[str], out_dir: str, frequency: str="annual") -> None:
    for tk in tickers:
        logger.info(f"Downloading {tk} ({frequency})")
        try:
            bundle = fetch_financials_for_ticker(tk, frequency=frequency)
            save_financials(bundle, out_dir, tk, frequency)
        except Exception as e:
            logger.error(f"Failed {tk}: {e}")

