from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, get_default_config_path, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

STATE_COLUMNS = [
    "sales",
    "cogs",
    "sga",
    "dep",
    "ar",
    "inventory",
    "ap",
    "ppe",
    "cash",
    "debt",
    "equity",
    "retained_earnings",
]

# Drivers follow the same order used by the simulator / accounting layer.
DRIVER_COLUMNS = [
    "growth",
    "gross_margin",
    "sga_ratio",
    "dep_rate",
    "dso",
    "dio",
    "dpo",
    "capex_ratio",
    "tax",
    "interest_rate",
    "payout",
]

EPS = 1e-9

IS_CANDIDATES: Dict[str, List[str]] = {
    "sales": ["sales", "total_revenue", "revenue"],
    "cogs": ["cogs", "reconciled_cost_of_revenue", "cost_of_revenue"],
    "sga": [
        "selling_general_and_administration",
        "selling_general_and_administration_expenses",
        "sga_expense",
    ],
    "dep": [
        "reconciled_depreciation",
        "depreciation_amortization_depletion",
        "depreciation_and_amortization",
    ],
    "int_exp": ["int_exp", "interest_expense", "interest_expense_non_operating"],
    "int_inc": ["int_inc", "interest_income", "interest_income_non_operating"],
    "tax_rate": ["tax_rate_for_calcs"],
    "net_income": [
        "net_income_from_continuing_operations",
        "net_income_from_continuing_operation_net_minority_interest",
        "net_income",
    ],
}

BS_CANDIDATES: Dict[str, List[str]] = {
    "cash": ["cash", "cash_cash_equivalents_and_short_term_investments"],
    "sti": ["other_short_term_investments"],
    "ar": ["receivables", "ar"],
    "inventory": ["inv_stock", "inventory"],
    "ap": ["payables", "accounts_payable", "payables_and_accrued_expenses"],
    "ppe": ["net_ppe", "property_plant_equipment_net", "property_plant_and_equipment_net"],
    "debt": ["total_debt", "net_debt"],
    "equity": [
        "stockholders_equity",
        "total_equity_gross_minority_interest",
        "common_stock_equity",
    ],
    "retained_earnings": ["re", "retained_earnings"],
}

CF_CANDIDATES: Dict[str, List[str]] = {
    "capex": ["capex", "capital_expenditure"],
    "dep_cf": ["dep_cf", "depreciation_amortization_depletion"],
    "dividends": ["cash_dividends_paid", "common_stock_dividend_paid", "dividends_paid"],
    "buybacks": ["repurchase_of_capital_stock"],
    "debt_issuance": ["issuance_of_debt", "net_long_term_debt_issuance"],
    "debt_repayment": ["repayment_of_debt", "lt_debt_payments"],
    "short_term_debt": ["net_short_term_debt_issuance", "short_term_debt_payments"],
}


@dataclass
class SimulationFrames:
    ticker: str
    years: List[pd.Timestamp]
    states: pd.DataFrame
    drivers: pd.DataFrame


def _read_statement(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    df.columns = pd.to_datetime(df.columns)
    df = df.sort_index(axis=1)
    return df.apply(pd.to_numeric, errors="coerce")


def _align_frames(*frames: pd.DataFrame) -> Tuple[pd.Index, List[pd.DataFrame]]:
    all_columns = sorted({c for frame in frames for c in frame.columns})
    aligned = [frame.reindex(columns=all_columns) for frame in frames]
    return pd.Index(all_columns), aligned


def _first_available(df: pd.DataFrame, candidates: List[str], default: float = 0.0) -> np.ndarray:
    for name in candidates:
        if name in df.index:
            values = df.loc[name].to_numpy(dtype=float)
            if np.all(np.isnan(values)):
                continue
            return np.nan_to_num(values, nan=default)
    return np.full(df.shape[1], default, dtype=float)


def _safe_ratio(num: float, denom: float, fallback: float = 0.0) -> float:
    if abs(denom) < EPS:
        return fallback
    return num / denom


def _prepare_series(is_df: pd.DataFrame, bs_df: pd.DataFrame, cf_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    sales = _first_available(is_df, IS_CANDIDATES["sales"])
    cogs = _first_available(is_df, IS_CANDIDATES["cogs"])
    sga = _first_available(is_df, IS_CANDIDATES["sga"])
    dep_is = _first_available(is_df, IS_CANDIDATES["dep"], default=0.0)
    int_exp = _first_available(is_df, IS_CANDIDATES["int_exp"], default=0.0)
    int_inc = _first_available(is_df, IS_CANDIDATES["int_inc"], default=0.0)
    tax_rate = _first_available(is_df, IS_CANDIDATES["tax_rate"], default=np.nan)
    net_income = _first_available(is_df, IS_CANDIDATES["net_income"], default=0.0)

    cash = _first_available(bs_df, BS_CANDIDATES["cash"])
    ar = _first_available(bs_df, BS_CANDIDATES["ar"])
    inventory = _first_available(bs_df, BS_CANDIDATES["inventory"])
    ap = _first_available(bs_df, BS_CANDIDATES["ap"])
    ppe = _first_available(bs_df, BS_CANDIDATES["ppe"])
    debt = _first_available(bs_df, BS_CANDIDATES["debt"], default=0.0)
    equity = _first_available(bs_df, BS_CANDIDATES["equity"], default=0.0)
    retained = _first_available(bs_df, BS_CANDIDATES["retained_earnings"], default=0.0)

    capex = _first_available(cf_df, CF_CANDIDATES["capex"], default=0.0)
    dep_cf = _first_available(cf_df, CF_CANDIDATES["dep_cf"], default=np.nan)
    dividends = np.abs(_first_available(cf_df, CF_CANDIDATES["dividends"], default=0.0))

    dep = np.where(np.isnan(dep_cf) | (dep_cf == 0), dep_is, dep_cf)
    int_cost = np.maximum(int_exp - int_inc, 0.0)
    pretax = sales - cogs - sga - dep - int_cost
    with np.errstate(invalid="ignore"):
        tax_eff = np.where(
            pretax > EPS,
            np.clip((pretax - (net_income + int_cost)) / pretax, 0.0, 0.6),
            np.nan,
        )
    fallback_tax = np.nanmedian(np.where(np.isfinite(tax_eff), tax_eff, np.nan))
    if not np.isfinite(fallback_tax):
        fallback_tax = 0.25
    tax = np.where(
        np.isfinite(tax_eff),
        tax_eff,
        np.clip(np.nan_to_num(tax_rate, nan=fallback_tax), 0.0, 0.6),
    )

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
        "equity": equity,
        "retained_earnings": retained,
        "capex": capex,
        "tax": tax,
        "int_cost": int_cost,
        "dividends": dividends,
        "net_income": net_income,
    }


def _build_state_frame(series: Dict[str, np.ndarray], years: pd.Index) -> pd.DataFrame:
    values = np.vstack([
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
    ]).T
    df = pd.DataFrame(values, index=years, columns=STATE_COLUMNS)
    df.index.name = "date"
    return df


def _build_driver_frame(series: Dict[str, np.ndarray], years: pd.Index) -> pd.DataFrame:
    if len(years) < 2:
        return pd.DataFrame(columns=DRIVER_COLUMNS)
    drivers = []
    index = []
    for idx in range(len(years) - 1):
        prev_idx = idx
        next_idx = idx + 1
        sales_prev = series["sales"][prev_idx]
        sales_next = series["sales"][next_idx]
        cogs_next = series["cogs"][next_idx]
        sga_next = series["sga"][next_idx]
        dep_next = series["dep"][next_idx]
        ar_next = series["ar"][next_idx]
        inv_next = series["inventory"][next_idx]
        ap_next = series["ap"][next_idx]
        capex_next = np.abs(series["capex"][next_idx])
        tax_next = series["tax"][next_idx]
        int_cost_next = series["int_cost"][next_idx]
        ni_next = series["net_income"][next_idx]
        debt_prev = series["debt"][prev_idx]
        ppe_prev = series["ppe"][prev_idx]
        payout_base = max(abs(ni_next), EPS)

        growth = (sales_next - sales_prev) / (abs(sales_prev) + EPS)
        gross_margin = 1.0 - _safe_ratio(cogs_next, sales_next, fallback=0.0)
        sga_ratio = _safe_ratio(sga_next, sales_next, fallback=0.0)
        dep_rate = _safe_ratio(dep_next, ppe_prev, fallback=0.0)
        dso = _safe_ratio(ar_next, sales_next, fallback=0.0) * 365.0
        dio = _safe_ratio(inv_next, cogs_next if abs(cogs_next) > EPS else sales_next, fallback=0.0) * 365.0
        dpo = _safe_ratio(ap_next, cogs_next if abs(cogs_next) > EPS else sales_next, fallback=0.0) * 365.0
        capex_ratio = _safe_ratio(capex_next, sales_next, fallback=0.0)
        tax = float(np.clip(tax_next, 0.0, 0.6))
        interest_rate = np.clip(_safe_ratio(int_cost_next, debt_prev, fallback=0.0), 0.0, 0.4)
        payout = np.clip(series["dividends"][next_idx] / payout_base, 0.0, 2.0)

        drivers.append([
            growth,
            gross_margin,
            sga_ratio,
            dep_rate,
            dso,
            dio,
            dpo,
            capex_ratio,
            tax,
            interest_rate,
            payout,
        ])
        index.append(years[next_idx])
    df = pd.DataFrame(drivers, index=pd.Index(index, name="date"), columns=DRIVER_COLUMNS)
    return df


def build_simulation_frames(ticker: str, raw_root: Path, frequency: str) -> SimulationFrames:
    is_path = raw_root / f"{ticker}_IS_{frequency}.csv"
    bs_path = raw_root / f"{ticker}_BS_{frequency}.csv"
    cf_path = raw_root / f"{ticker}_CF_{frequency}.csv"
    is_df = _read_statement(is_path)
    bs_df = _read_statement(bs_path)
    cf_df = _read_statement(cf_path)
    years, (is_aligned, bs_aligned, cf_aligned) = _align_frames(is_df, bs_df, cf_df)
    series = _prepare_series(is_aligned, bs_aligned, cf_aligned)
    states = _build_state_frame(series, years)
    drivers = _build_driver_frame(series, years)
    return SimulationFrames(ticker=ticker, years=list(years), states=states, drivers=drivers)


def _save_npz(out_dir: Path, frames: SimulationFrames) -> None:
    npz_path = out_dir / f"{frames.ticker}_simulation.npz"
    np.savez(
        npz_path,
        years=np.array([str(idx) for idx in frames.states.index]),
        states=frames.states.to_numpy(dtype=float),
        drivers=frames.drivers.to_numpy(dtype=float),
    )


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
    out_dir = proc_root / (variant or "simulation")
    ensure_dir(out_dir)

    summary: Dict[str, Dict[str, object]] = {}
    frequency = override_frequency or cfg.frequency
    for ticker in cfg.tickers:
        try:
            frames = build_simulation_frames(ticker, raw_root, frequency)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
            continue
        if frames.states.empty or frames.drivers.empty:
            logger.warning("Skipping %s: not enough data for simulation", ticker)
            continue
        frames.states.to_csv(out_dir / f"{ticker}_states.csv", index_label="date")
        frames.drivers.to_csv(out_dir / f"{ticker}_drivers.csv", index_label="date")
        _save_npz(out_dir, frames)
        summary[ticker] = {
            "years": [str(idx) for idx in frames.states.index],
            "transitions": len(frames.drivers),
        }
        logger.info("Prepared simulation inputs for %s (%d transitions)", ticker, len(frames.drivers))

    summary_path = out_dir / "simulation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Simulation-ready data stored under %s", out_dir)
    return str(out_dir)
