from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.datahandler.data_sanity_check import standardize_columns
from src.datahandler.features import compute_drivers
from src.utils.io import ensure_dir, get_default_config_path, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


FILL_POLICIES_IS = {
    "sales": "interp",
    "cogs": "interp",
    "opex": "interp",
    "dep": "zero",
    "ebt": "interp",
    "ni": "interp",
    "int_exp": "zero",
    "int_inc": "zero",
}

FILL_POLICIES_BS = {
    "cash": "ffill",
    "cash_and_sti": "ffill",
    "sti": "ffill",
    "ppe_net": "ffill",
    "assets": "constraint",
    "liab_total": "constraint",
    "equity_total": "constraint",
    "debt_st": "zero",
    "debt_lt": "ffill",
    "ar": "ffill",
    "ap": "ffill",
    "inv_stock": "ffill",
    "re": "ffill",
    "pic": "ffill",
}

FILL_POLICIES_CF = {
    "capex": "zero",
    "dep_cf": "zero",
    "lt_debt_payments": "zero",
    "lt_debt_issuance": "zero",
    "equity_issuance": "zero",
    "dividends_paid": "zero",
    "buyback": "zero",
}

STATE_ORDER = [
    ("cash", "cash"),
    ("inv_short", "sti"),
    ("ppe_net", "ppe_net"),
    ("debt_st", "debt_st"),
    ("debt_lt", "debt_lt"),
    ("re", "re"),
    ("pic", "pic"),
    ("ar", "ar"),
    ("ap", "ap"),
    ("inv_stock", "inv_stock"),
]

DRIVER_COLUMNS = [
    "Sales",
    "COGS",
    "Opex",
    "Dep",
    "Capex",
    "rST",
    "rLT",
    "rINV",
    "AmortLT",
    "Cbar",
    "EI",
    "Div",
    "tau",
    "DSO",
    "DPO",
    "DIO",
]

COVARIATE_EXTRA_COLUMNS = [
    "Capex_Sales", "Capex_PPE", "Capex_Dep", "Capex_Cashbound", "Cash_Buffer_Ratio",
    "DSO_Share", "DPO_Share", "DIO_Share"
]


@dataclass
class TickerData:
    ticker: str
    states: pd.DataFrame
    drivers: pd.DataFrame
    scale: float
    state_shift: pd.Series
    state_scale: pd.Series


def _read_statement(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing statement file: {path}")
    df = pd.read_csv(path, index_col=0)
    df.columns = pd.to_datetime(df.columns)
    return df


def _apply_fill(df: pd.DataFrame, policies: Dict[str, str]) -> pd.DataFrame:
    df = df.copy().apply(pd.to_numeric, errors="coerce")
    # make sure required rows exist
    for idx, policy in policies.items():
        if idx not in df.index:
            df.loc[idx] = np.nan
        series = df.loc[idx]
        if policy == "zero":
            df.loc[idx] = series.fillna(0.0)
        elif policy == "ffill":
            df.loc[idx] = series.sort_index().ffill().bfill().fillna(0.0)
        elif policy == "interp":
            df.loc[idx] = (
                series.sort_index().interpolate(limit_direction="both").bfill().ffill().fillna(0.0)
            )
        elif policy == "constraint":
            # handled later
            continue
    if not df.index.is_unique:
        df = df.groupby(level=0).mean()
    return df


def _enforce_bs_constraints(bs_df: pd.DataFrame) -> pd.DataFrame:
    bs_df = bs_df.copy()
    assets = bs_df.loc["assets"] if "assets" in bs_df.index else None
    liab = bs_df.loc["liab_total"] if "liab_total" in bs_df.index else None
    equity = bs_df.loc["equity_total"] if "equity_total" in bs_df.index else None
    if assets is None and liab is not None and equity is not None:
        bs_df.loc["assets"] = liab + equity
    if liab is None and assets is not None and equity is not None:
        bs_df.loc["liab_total"] = assets - equity
    if equity is None and assets is not None and liab is not None:
        bs_df.loc["equity_total"] = assets - liab
    return bs_df


def _derive_dep(is_df: pd.DataFrame, cf_df: pd.DataFrame) -> pd.Series:
    if "dep" in is_df.index:
        series = is_df.loc["dep"]
        if isinstance(series, pd.DataFrame):
            series = series.sum(axis=0)
        return series
    if "dep_cf" in cf_df.index:
        series = cf_df.loc["dep_cf"]
        if isinstance(series, pd.DataFrame):
            series = series.sum(axis=0)
        return series.abs()
    return pd.Series(0.0, index=is_df.columns)


def _build_states(bs_df: pd.DataFrame) -> pd.DataFrame:
    data = {}
    for state_name, source in STATE_ORDER:
        if source in bs_df.index:
            series = bs_df.loc[source]
            if isinstance(series, pd.DataFrame):
                series = series.sum(axis=0)
        elif source == "sti" and "cash_and_sti" in bs_df.index and "cash" in bs_df.index:
            series = (bs_df.loc["cash_and_sti"] - bs_df.loc["cash"]).clip(lower=0.0)
        else:
            series = pd.Series(0.0, index=bs_df.columns)
        data[state_name] = series.fillna(0.0)
    states = pd.DataFrame(data)
    states.index = bs_df.columns
    states.sort_index(inplace=True)
    return states


def _safe_div(num: pd.Series, denom: pd.Series, default: float) -> pd.Series:
    ratio = num.copy()
    ratio[:] = default
    valid = denom.replace(0.0, np.nan)
    ratio = (num / valid).fillna(default)
    return ratio


def _row_or_default(df: pd.DataFrame, name: str, index: pd.Index, default: float = 0.0) -> pd.Series:
    if name in df.index:
        series = df.loc[name]
        if isinstance(series, pd.DataFrame):
            series = series.sum(axis=0)
        return series
    return pd.Series(default, index=index)


def _logit_transform(series: pd.Series, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
    span = upper - lower
    clipped = series.clip(lower + 1e-6, upper - 1e-6)
    normalized = (clipped - lower) / span
    return np.log(normalized / (1 - normalized))


def _build_drivers(
    ticker: str,
    is_df: pd.DataFrame,
    bs_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    is_df = is_df.copy()
    cf_df = cf_df.copy()
    capex = (-_row_or_default(cf_df, "capex", is_df.columns)).clip(lower=0.0)
    dep = _derive_dep(is_df, cf_df)
    features = compute_drivers(is_df, bs_df, cf_df)
    default_idx = is_df.columns
    dso = features.get("DSO", pd.Series(45.0, index=default_idx)).replace([np.inf, -np.inf], np.nan).fillna(45.0)
    dpo = features.get("DPO", pd.Series(50.0, index=default_idx)).replace([np.inf, -np.inf], np.nan).fillna(50.0)
    dio = features.get("DIO", pd.Series(60.0, index=default_idx)).replace([np.inf, -np.inf], np.nan).fillna(60.0)
    cbar = (features.get("cash_target_ratio", pd.Series(0.05, index=is_df.columns)) * is_df.loc["sales"].abs()).fillna(0.0)

    int_exp = _row_or_default(is_df, "int_exp", is_df.columns).abs()
    int_inc = _row_or_default(is_df, "int_inc", is_df.columns).abs()
    debt_st = _row_or_default(bs_df, "debt_st", is_df.columns).replace(0.0, np.nan)
    debt_lt = _row_or_default(bs_df, "debt_lt", is_df.columns).replace(0.0, np.nan)
    rST = (int_exp * 0.35 / debt_st).replace([np.inf, -np.inf], np.nan).fillna(0.05)
    rLT = (int_exp * 0.65 / debt_lt).replace([np.inf, -np.inf], np.nan).fillna(0.04)
    rINV = _safe_div(int_inc, states["inv_short"] + 1e-6, 0.02)

    amort_lt = _row_or_default(cf_df, "lt_debt_payments", is_df.columns).abs()
    equity_inj = (_row_or_default(cf_df, "equity_issuance", is_df.columns) - _row_or_default(cf_df, "buyback", is_df.columns).abs()).clip(lower=0.0)
    dividends = _row_or_default(cf_df, "dividends_paid", is_df.columns).abs()

    ebt = is_df.loc["ebt"] if "ebt" in is_df.index else _row_or_default(is_df, "ni", is_df.columns)
    taxes = _row_or_default(is_df, "tax", is_df.columns).fillna(0.0)
    tau = _safe_div(taxes, ebt.replace(0.0, np.nan), 0.25).clip(0.0, 0.45)

    opex_series = _row_or_default(is_df, "opex", is_df.columns, default=np.nan)
    if opex_series.isna().all():
        opex_series = is_df.loc["sales"] * 0.3

    driver_dict = {
        "Sales": is_df.loc["sales"],
        "COGS": is_df.loc["cogs"],
        "Opex": opex_series,
        "Dep": dep,
        "Capex": capex,
        "rST": rST,
        "rLT": rLT,
        "rINV": rINV,
        "AmortLT": amort_lt,
        "Cbar": cbar,
        "EI": equity_inj,
        "Div": dividends,
        "tau": tau,
        "DSO": dso,
        "DPO": dpo,
        "DIO": dio,
    }
    drivers = pd.concat(driver_dict, axis=1)
    drivers.sort_index(inplace=True)
    drivers.fillna(0.0, inplace=True)
    missing_cols = [col for col in DRIVER_COLUMNS if col not in drivers.columns]
    for col in missing_cols:
        drivers[col] = 0.0
    drivers = drivers[DRIVER_COLUMNS]
    return drivers


def _determine_scale(is_df: pd.DataFrame, bs_df: pd.DataFrame) -> float:
    sales = is_df.loc["sales"] if "sales" in is_df.index else None
    assets = bs_df.loc["assets"] if "assets" in bs_df.index else None
    values = []
    if sales is not None:
        values.append(sales.abs().median())
    if assets is not None:
        values.append(assets.abs().median())
    scale = max([v for v in values if v and not np.isnan(v)] or [1.0])
    return float(max(scale, 1.0))


def _compute_covariates(
    drivers: pd.DataFrame,
    states: pd.DataFrame,
    sales: pd.Series,
    capex: pd.Series,
    dep: pd.Series,
    cbar: pd.Series,
    dividends: pd.Series,
    taxes: pd.Series,
) -> pd.DataFrame:
    eps = 1e-6
    cov = pd.DataFrame(index=drivers.index)
    capex_abs = capex.abs()
    cov["Capex_Sales"] = (capex_abs / (sales.abs() + eps)).clip(0.0, 1.0)
    cov["Capex_PPE"] = (capex_abs / (states["ppe_net"].abs() + eps)).clip(0.0, 1.0)
    cov["Capex_Dep"] = (capex_abs / (dep.abs() + eps)).clip(0.0, 2.0)
    cfo_proxy = (sales - drivers["COGS"] - drivers["Opex"] - taxes).fillna(0.0)
    capex_upper = (states["cash"] + cfo_proxy - dividends - cbar).clip(lower=eps)
    cov["Capex_Cashbound"] = (capex_abs / (capex_upper + eps)).clip(0.0, 1.2)
    operating_cost = (drivers["COGS"].abs() + drivers["Opex"].abs() + eps)
    cov["Cash_Buffer_Ratio"] = (states["cash"] / operating_cost).clip(0.0, 2.0)
    cov["DSO_Share"] = (drivers["DSO"] / 365.0).clip(0.0, 1.0)
    cov["DPO_Share"] = (drivers["DPO"] / 365.0).clip(0.0, 1.0)
    cov["DIO_Share"] = (drivers["DIO"] / 365.0).clip(0.0, 1.0)
    cov.replace([np.inf, -np.inf], np.nan, inplace=True)
    cov.fillna(0.0, inplace=True)
    return cov


def _normalize(states: pd.DataFrame,
               drivers: pd.DataFrame,
               method: str,
               scale: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series]]:
    meta: Dict[str, pd.Series] = {}
    if method == "zscore":
        states_mean = states.mean()
        states_std = states.std().replace(0.0, 1.0)
        meta["state_shift"] = states_mean
        meta["state_scale"] = states_std
        states_norm = (states - states_mean) / states_std
        drivers_mean = drivers.mean()
        drivers_std = drivers.std().replace(0.0, 1.0)
        drivers_norm = (drivers - drivers_mean) / drivers_std
    else:  # scale
        states_shift = pd.Series(0.0, index=states.columns)
        states_scale = pd.Series(scale, index=states.columns)
        meta["state_shift"] = states_shift
        meta["state_scale"] = states_scale
        states_norm = states / scale
        flow_cols = ["Sales", "COGS", "Opex", "Dep", "Capex", "AmortLT", "Cbar", "EI", "Div"]
        drivers_norm = drivers.copy()
        drivers_norm[flow_cols] = drivers_norm[flow_cols] / scale
    return states_norm, drivers_norm, meta


def preprocess_ticker(ticker: str,
                      raw_dir: Path,
                      proc_dir: Path,
                      frequency: str,
                      norm_method: str) -> Optional[TickerData]:
    try:
        is_path = raw_dir / f"{ticker}_IS_{frequency}.csv"
        bs_path = raw_dir / f"{ticker}_BS_{frequency}.csv"
        cf_path = raw_dir / f"{ticker}_CF_{frequency}.csv"
        is_df = standardize_columns(_read_statement(is_path), "IS")
        bs_df = standardize_columns(_read_statement(bs_path), "BS")
        cf_df = standardize_columns(_read_statement(cf_path), "CF")
    except FileNotFoundError as err:
        logger.warning("%s", err)
        return None

    for name, frame in (("IS", is_df), ("BS", bs_df), ("CF", cf_df)):
        if frame.empty or frame.shape[1] == 0:
            logger.warning("%s statement for %s is empty; skipping ticker", name, ticker)
            return None

    for name, frame in (("IS", is_df), ("BS", bs_df), ("CF", cf_df)):
        if frame.empty or frame.shape[1] == 0:
            logger.warning("%s statement for %s is empty; skipping ticker", name, ticker)
            return None

    is_df = _apply_fill(is_df, FILL_POLICIES_IS)
    bs_df = _enforce_bs_constraints(_apply_fill(bs_df, FILL_POLICIES_BS))
    cf_df = _apply_fill(cf_df, FILL_POLICIES_CF)

    states = _build_states(bs_df)
    drivers = _build_drivers(ticker, is_df, bs_df, cf_df, states)
    common_index = states.index.intersection(drivers.index)
    states = states.loc[common_index]
    drivers = drivers.loc[common_index]

    scale = _determine_scale(is_df, bs_df)
    states_norm, drivers_norm, norm_meta = _normalize(states, drivers, norm_method, scale)
    sales_series = is_df.loc["sales"].reindex(common_index)
    dep_series = _derive_dep(is_df, cf_df).reindex(common_index)
    tax_series = _row_or_default(is_df, "tax", is_df.columns).reindex(common_index).fillna(0.0)
    covariates = _compute_covariates(
        drivers_norm,
        states_norm,
        sales_series,
        drivers["Capex"],
        dep_series,
        drivers["Cbar"],
        drivers["Div"],
        tax_series,
    )
    covariates = pd.concat([drivers_norm, covariates], axis=1)

    ensure_dir(str(proc_dir))
    states.to_csv(proc_dir / f"{ticker}_states_raw.csv")
    drivers.to_csv(proc_dir / f"{ticker}_drivers_raw.csv")
    states_norm.to_csv(proc_dir / f"{ticker}_states_normalized.csv")
    drivers_norm.to_csv(proc_dir / f"{ticker}_drivers_normalized.csv")
    covariates.to_csv(proc_dir / f"{ticker}_covariates.csv")
    meta_path = proc_dir / f"{ticker}_meta.json"
    meta_payload = {"scale": scale, "rows": len(states_norm), "norm_method": norm_method}
    meta_payload.update({k: {col: float(val[col]) for col in val.index} for k, val in norm_meta.items()})
    meta_path.write_text(json.dumps(meta_payload, indent=2))

    return TickerData(
        ticker=ticker,
        states=states_norm,
        drivers=covariates,
        scale=scale,
        state_shift=norm_meta["state_shift"],
        state_scale=norm_meta["state_scale"],
    )


def _assemble_training_dataset(
    ticker_data: List[TickerData], seq_len: int
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
    np.ndarray,
    np.ndarray,
]:
    sequences_states: List[np.ndarray] = []
    sequences_covs: List[np.ndarray] = []
    retained: List[str] = []
    norm_shifts: List[np.ndarray] = []
    norm_scales: List[np.ndarray] = []
    if not ticker_data:
        raise RuntimeError("No ticker data provided to assemble the training dataset.")
    min_history = min(len(item.states) for item in ticker_data)
    effective_seq_len = min(seq_len, min_history)
    for item in ticker_data:
        if len(item.states) < effective_seq_len:
            logger.debug(
                "Skipping %s due to insufficient history (%s < %s)",
                item.ticker,
                len(item.states),
                effective_seq_len,
            )
            continue
        states_slice = item.states.sort_index().tail(effective_seq_len)
        drivers_slice = item.drivers.sort_index().tail(effective_seq_len)
        sequences_states.append(states_slice.to_numpy(dtype=np.float32))
        sequences_covs.append(drivers_slice.to_numpy(dtype=np.float32))
        retained.append(item.ticker)
        norm_shifts.append(item.state_shift.to_numpy(dtype=np.float32))
        norm_scales.append(item.state_scale.to_numpy(dtype=np.float32))
    if not sequences_states:
        raise RuntimeError("No ticker has enough history to assemble the training dataset.")
    states_arr = np.stack(sequences_states, axis=0)
    covs_arr = np.stack(sequences_covs, axis=0)
    targets_arr = states_arr.copy()
    cov_columns = list(ticker_data[0].drivers.columns)
    state_shift_arr = np.stack(norm_shifts, axis=0)
    state_scale_arr = np.stack(norm_scales, axis=0)
    return states_arr, covs_arr, targets_arr, retained, cov_columns, state_shift_arr, state_scale_arr


def run_preprocessing_pipeline(config_path: Optional[str] = None,
                               variant: str = "base",
                               norm_method: str = "scale",
                               override_frequency: Optional[str] = None) -> Path:
    cfg_path = config_path or get_default_config_path()
    cfg = load_config(cfg_path)
    frequency = override_frequency or cfg.frequency
    raw_dir = Path(cfg.paths["raw_dir"])
    proc_root = Path(cfg.paths["proc_dir"])
    proc_dir = proc_root / variant
    ensure_dir(str(proc_dir))
    ticker_results: List[TickerData] = []
    for ticker in cfg.tickers:
        logger.info("Preprocessing %s", ticker)
        td = preprocess_ticker(ticker, raw_dir, proc_dir, frequency, norm_method)
        if td:
            ticker_results.append(td)
    seq_len = int(cfg.training.get("window", 6) + cfg.training.get("horizon", 4))
    (states_arr,
     covs_arr,
     targets_arr,
     tickers,
     cov_columns,
     state_shift_arr,
     state_scale_arr) = _assemble_training_dataset(ticker_results, seq_len)
    dataset_path = proc_dir / "training_data.npz"
    np.savez(
        dataset_path,
        states=states_arr,
        covs=covs_arr,
        target_states=targets_arr,
        tickers=np.array(tickers, dtype="U10"),
        cov_columns=np.array(cov_columns, dtype="U30"),
        state_shift=state_shift_arr,
        state_scale=state_scale_arr,
    )
    summary = {
        "tickers": tickers,
        "seq_len_requested": seq_len,
        "seq_len_effective": states_arr.shape[1],
        "num_samples": len(tickers),
        "states_shape": states_arr.shape,
        "covs_shape": covs_arr.shape,
        "covariate_columns": cov_columns,
        "variant": variant,
        "norm_method": norm_method,
        "frequency": frequency,
    }
    (proc_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Saved normalized dataset to %s", dataset_path)
    return dataset_path
