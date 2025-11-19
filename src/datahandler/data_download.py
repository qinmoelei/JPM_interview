from __future__ import annotations
import argparse
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional, Sequence
import sys 

sys.path.append('.')
from src.datahandler.data_sanity_check import standardize_columns
from src.utils.io import ensure_dir, load_config, get_default_config_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

def _import_yf():
    """Import yfinance lazily so that offline preprocess steps stay usable."""
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        raise RuntimeError("yfinance is required to download data. Install with `pip install yfinance`.") from e
    return yf

def fetch_financials_for_ticker(ticker: str, frequency: str="annual") -> Dict[str, pd.DataFrame]:
    """Download IS, BS, CF for a ticker using yfinance; return standardized DataFrames (columns=period end).

    Example:
        >>> bundle = fetch_financials_for_ticker("AAPL", frequency="annual")  # doctest: +SKIP
        >>> sorted(bundle.keys())  # doctest: +SKIP
        ['BS', 'CF', 'IS']
    """
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
        """Convert columns to datetime + normalize string indices for later joins."""
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
    """Persist each statement under `{ticker}_{statement}_{frequency}.csv`."""
    ensure_dir(out_dir)
    for k, df in bundle.items():
        path = os.path.join(out_dir, f"{ticker}_{k}_{frequency}.csv")
        df.to_csv(path)

def download_universe(tickers: List[str], out_dir: str, frequency: str="annual") -> None:
    """Loop tickers, fetch statements, and write CSVs; logs failures instead of crashing."""
    for tk in tickers:
        logger.info(f"Downloading {tk} ({frequency})")
        try:
            bundle = fetch_financials_for_ticker(tk, frequency=frequency)
            save_financials(bundle, out_dir, tk, frequency)
        except Exception as e:
            logger.error(f"Failed {tk}: {e}")

def run_download_pipeline(config_path: Optional[str] = None, override_frequency: Optional[str] = None) -> str:
    """Load config and execute the download stage. Returns the path that was used."""
    cfg_path = config_path or get_default_config_path()
    cfg = load_config(cfg_path)
    raw_dir = cfg.paths["raw_dir"]
    ensure_dir(raw_dir)
    frequency = override_frequency or cfg.frequency
    download_universe(cfg.tickers, out_dir=raw_dir, frequency=frequency)
    logger.info("Download stage finished.")
    return cfg_path

def main(cli_args: Optional[Sequence[str]] = None) -> None:
    """Simple CLI entry point (invoked by script/00_preprocess.py)."""
    ap = argparse.ArgumentParser(description="Download raw financial statements for the configured tickers.")
    ap.add_argument(
        "--config",
        default=get_default_config_path(),
        help="Path to the YAML config file. Defaults to %(default)s or $JPM_CONFIG_PATH if set.",
    )
    ap.add_argument("--frequency", choices=["annual", "quarterly"], default=None, help="Override frequency for download.")
    args = ap.parse_args(cli_args)
    run_download_pipeline(args.config, override_frequency=args.frequency)

if __name__ == "__main__":
    main()
