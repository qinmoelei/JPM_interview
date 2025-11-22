from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.model.dynamics_tf import DRIVER_ORDER, STATE_ORDER, inverse_sequence
from src.model.simulator import AccountingSimulator
from src.utils.io import ensure_dir, get_default_config_path, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

EPS = 1e-8

IS_CANDIDATES: Dict[str, List[str]] = {
    "sales": ["sales", "total_revenue", "revenue"],
    "cogs": ["cogs", "reconciled_cost_of_revenue", "cost_of_revenue"],
    "sga": [
        "opex",
        "operating_expense",
        "operating_expenses",
        "selling_general_and_administration",
        "selling_general_and_administration_expenses",
        "sga_expense",
    ],
    "dep": [
        "reconciled_depreciation",
        "depreciation_amortization_depletion",
        "depreciation_and_amortization",
    ],
    "interest_expense": ["interest_expense", "int_exp", "interest_expense_non_operating"],
    "tax_expense": ["tax_provision", "income_tax_expense", "provision_for_income_taxes"],
    "retained_earnings": ["re", "retained_earnings"],
}

BS_CANDIDATES: Dict[str, List[str]] = {
    "cash": ["cash", "cash_cash_equivalents", "cash_and_equivalents", "cash_cash_equivalents_and_short_term_investments"],
    "ar": ["accounts_receivable_t", "receivables", "accounts_receivable", "ar"],
    "inventory": ["inventory", "inv_stock"],
    "ap": ["accounts_payable", "payables", "payables_and_accrued_expenses"],
    "ppe": ["net_ppe", "property_plant_equipment_net", "property_plant_and_equipment_net"],
    "debt": ["total_interest_bearing_debt_t", "total_debt", "net_debt", "long_term_debt", "short_long_term_debt_total"],
    "equity": ["stockholders_equity", "total_equity_gross_minority_interest", "common_stock_equity"],
    "retained_earnings": ["re", "retained_earnings"],
}

CF_CANDIDATES: Dict[str, List[str]] = {
    "dividends": ["cash_dividends_paid", "common_stock_dividend_paid", "dividends_paid"],
}


@dataclass
class SimulationFrames:
    ticker: str
    years: List[pd.Timestamp]
    states: pd.DataFrame
    drivers: pd.DataFrame
    perfect_mae: float
    perfect_mse: float


def _read_statement(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    df.columns = pd.to_datetime(df.columns)
    return df.sort_index(axis=1).apply(pd.to_numeric, errors="coerce")


def _align_frames(*frames: pd.DataFrame) -> Tuple[pd.Index, List[pd.DataFrame]]:
    all_columns = sorted({col for frame in frames for col in frame.columns})
    aligned = [frame.reindex(columns=all_columns) for frame in frames]
    return pd.Index(all_columns), aligned


def _first_available(df: pd.DataFrame, candidates: List[str], default: float = 0.0) -> np.ndarray:
    for name in candidates:
        if name in df.index:
            values = df.loc[name]
            if isinstance(values, pd.DataFrame):
                values = values.iloc[0]
            values = values.to_numpy(dtype=float)
            if np.all(np.isnan(values)):
                continue
            return np.nan_to_num(values, nan=default)
    return np.full(df.shape[1], default, dtype=float)


def _drop_inactive_prefix(series: Dict[str, np.ndarray], years: pd.Index) -> Tuple[Dict[str, np.ndarray], pd.Index]:
    if not years.size:
        return series, years
    core_keys = ["sales", "cogs", "sga", "dep", "ar", "inventory", "ap", "ppe", "cash", "debt"]
    stacked = []
    for key in core_keys:
        stacked.append(np.abs(series.get(key, np.zeros_like(series["sales"]))))
    summed = np.sum(stacked, axis=0)
    non_zero = np.where(summed > EPS)[0]
    if non_zero.size == 0:
        return series, years
    start = int(non_zero[0])
    if start <= 0:
        return series, years
    trimmed = {k: v[start:] for k, v in series.items()}
    return trimmed, years[start:]


def _prepare_series(is_df: pd.DataFrame, bs_df: pd.DataFrame, cf_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    sales = _first_available(is_df, IS_CANDIDATES["sales"])
    cogs = _first_available(is_df, IS_CANDIDATES["cogs"])
    sga = _first_available(is_df, IS_CANDIDATES["sga"])
    dep = _first_available(is_df, IS_CANDIDATES["dep"], default=0.0)
    interest_expense = _first_available(is_df, IS_CANDIDATES["interest_expense"], default=0.0)
    tax_expense = _first_available(is_df, IS_CANDIDATES["tax_expense"], default=0.0)
    re_raw = _first_available(is_df, IS_CANDIDATES["retained_earnings"], default=np.nan)

    cash = _first_available(bs_df, BS_CANDIDATES["cash"])
    ar = _first_available(bs_df, BS_CANDIDATES["ar"])
    inventory = _first_available(bs_df, BS_CANDIDATES["inventory"])
    ap = _first_available(bs_df, BS_CANDIDATES["ap"])
    ppe = _first_available(bs_df, BS_CANDIDATES["ppe"])
    debt = _first_available(bs_df, BS_CANDIDATES["debt"], default=0.0)
    eq_raw = _first_available(bs_df, BS_CANDIDATES["equity"], default=np.nan)
    re_bs = _first_available(bs_df, BS_CANDIDATES["retained_earnings"], default=np.nan)

    dividends = np.abs(_first_available(cf_df, CF_CANDIDATES["dividends"], default=0.0))

    eq_model = cash + ar + inventory + ppe - ap - debt
    n_periods = eq_model.shape[0]
    if n_periods == 0:
        return {
            "sales": sales,
            "cogs": cogs,
            "sga": sga,
            "dep": dep,
            "ar": ar,
            "inventory": inventory,
            "ap": ap,
            "ppe": ppe,
            "cash": cash,
            "debt": debt,
            "equity": eq_model,
            "retained_earnings": np.array([], dtype=float),
            "tax_expense": tax_expense,
            "interest_expense": interest_expense,
            "dividends": dividends,
        }

    ebit = sales - cogs - sga - dep
    ebt = ebit - interest_expense
    ni = ebt - tax_expense

    re_source = re_bs if np.any(np.isfinite(re_bs)) else re_raw
    re_clean = np.zeros_like(eq_model)
    re_clean[0] = re_source[0] if np.isfinite(re_source[0]) else eq_model[0]
    for t in range(1, len(eq_model)):
        re_clean[t] = re_clean[t - 1] + ni[t] - dividends[t]

    return {
        "sales": sales,
        "cogs": cogs,
        "sga": sga,
        "dep": dep,
        "ar": ar,
        "inventory": inventory,
        "ap": ap,
        "ppe": ppe,
        "cash": cash,
        "debt": debt,
        "equity": eq_model,
        "retained_earnings": re_clean,
        "tax_expense": tax_expense,
        "interest_expense": interest_expense,
        "dividends": dividends,
    }


def _build_state_frame(series: Dict[str, np.ndarray], years: pd.Index) -> pd.DataFrame:
    values = np.vstack(
        [
            series["sales"],
            series["cogs"],
            series["sga"],
            series["dep"],
            series["ar"],
            series["inventory"],
            series["ap"],
            series["ppe"],
            series["cash"],
            series["debt"],
            series["equity"],
            series["retained_earnings"],
            series["tax_expense"],
            series["interest_expense"],
            series["dividends"],
        ]
    ).T
    df = pd.DataFrame(values, index=years, columns=STATE_ORDER)
    df.index.name = "date"
    return df


def _build_driver_frame(states: np.ndarray, years: pd.Index) -> pd.DataFrame:
    drivers = inverse_sequence(states, eps=EPS)
    if drivers.shape[0] == 0:
        return pd.DataFrame(columns=DRIVER_ORDER)
    driver_index = pd.Index(years[1:], name="date")
    return pd.DataFrame(drivers, index=driver_index, columns=DRIVER_ORDER)


def _perfect_reconstruction(states: pd.DataFrame, drivers: pd.DataFrame) -> Tuple[float, float]:
    if states.shape[0] < 2 or drivers.empty:
        return float("nan"), float("nan")
    simulator = AccountingSimulator()
    predicted = simulator.roll(states.iloc[0].to_numpy(dtype=float), drivers.to_numpy(dtype=float))
    target = states.iloc[1:].to_numpy(dtype=float)
    diff = predicted - target
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    return mae, mse


def build_simulation_frames(ticker: str, raw_root: Path, frequency: str) -> SimulationFrames:
    is_path = raw_root / f"{ticker}_IS_{frequency}.csv"
    bs_path = raw_root / f"{ticker}_BS_{frequency}.csv"
    cf_path = raw_root / f"{ticker}_CF_{frequency}.csv"
    is_df = _read_statement(is_path)
    bs_df = _read_statement(bs_path)
    cf_df = _read_statement(cf_path)
    years, (is_aligned, bs_aligned, cf_aligned) = _align_frames(is_df, bs_df, cf_df)
    series = _prepare_series(is_aligned, bs_aligned, cf_aligned)
    series, years = _drop_inactive_prefix(series, years)
    if years.size < 2:
        return SimulationFrames(ticker=ticker, years=list(years), states=pd.DataFrame(), drivers=pd.DataFrame(), perfect_mae=float("nan"), perfect_mse=float("nan"))
    states = _build_state_frame(series, years)
    drivers = _build_driver_frame(states.to_numpy(dtype=float), years)
    perfect_mae, perfect_mse = _perfect_reconstruction(states, drivers)
    return SimulationFrames(
        ticker=ticker,
        years=list(years),
        states=states,
        drivers=drivers,
        perfect_mae=perfect_mae,
        perfect_mse=perfect_mse,
    )


def _save_npz(out_dir: Path, frames: SimulationFrames) -> None:
    npz_path = out_dir / f"{frames.ticker}_simulation.npz"
    np.savez(
        npz_path,
        years=np.array([str(idx) for idx in frames.states.index]),
        states=frames.states.to_numpy(dtype=float),
        drivers=frames.drivers.to_numpy(dtype=float),
    )


def _frequency_to_variant(freq: str, override: Optional[str]) -> str:
    if override:
        return override
    mapping = {"annual": "year", "annuals": "year", "year": "year", "quarterly": "quarter", "quarter": "quarter"}
    return mapping.get(freq.lower(), freq)


def run_preprocessing_pipeline(
    config_path: Optional[str] = None,
    variant: Optional[str] = None,
    override_frequency: Optional[str] = None,
    **_: Sequence[str],
) -> str:
    cfg_path = config_path or get_default_config_path()
    cfg = load_config(cfg_path)
    raw_root = Path(cfg.paths.get("raw_dir", "data/raw"))
    proc_root = Path(cfg.paths.get("proc_dir", "data/processed"))

    frequency = override_frequency or cfg.frequency
    out_dir = proc_root / _frequency_to_variant(frequency, variant)
    ensure_dir(out_dir)

    summary: Dict[str, Dict[str, object]] = {}
    for ticker in cfg.tickers:
        try:
            frames = build_simulation_frames(ticker, raw_root, frequency)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
            continue
        if frames.states.empty or frames.drivers.empty:
            logger.warning("Skipping %s: insufficient data after trimming", ticker)
            continue
        frames.states.to_csv(out_dir / f"{frames.ticker}_states.csv", index_label="date")
        frames.drivers.to_csv(out_dir / f"{frames.ticker}_drivers.csv", index_label="date")
        _save_npz(out_dir, frames)
        summary[ticker] = {
            "years": [str(idx) for idx in frames.states.index],
            "transitions": len(frames.drivers),
            "perfect_mae": frames.perfect_mae,
            "perfect_mse": frames.perfect_mse,
        }
        logger.info(
            "Prepared %s (%d transitions) | perfect MAE=%.3e MSE=%.3e",
            ticker,
            len(frames.drivers),
            frames.perfect_mae,
            frames.perfect_mse,
        )

    summary_path = out_dir / "simulation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Simulation-ready data stored under %s", out_dir)
    return str(out_dir)
