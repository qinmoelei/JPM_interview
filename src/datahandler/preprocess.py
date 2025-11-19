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
    splits: pd.Series
    transition_splits: pd.Series


def _read_statement(path: Path) -> pd.DataFrame:
    """Load one Yahoo CSV statement.

    Args:
        path: Location of the CSV exported by `01_download.py`.

    Returns:
        DataFrame indexed by line-item names with datetime columns.

    Example:
        >>> sample_path = Path("data/AAPL_IS_annual.csv")
        >>> statement = _read_statement(sample_path)
        >>> statement.columns.dtype  # doctest: +SKIP
        dtype('<M8[ns]')
    """
    # Guardrail so downstream logic fails fast when a file is missing.
    if not path.exists():
        raise FileNotFoundError(f"Missing statement file: {path}")
    # Read the CSV while keeping line items as the index.
    df = pd.read_csv(path, index_col=0)
    # Convert the string-formatted period columns into datetimes for sorting/filtering.
    df.columns = pd.to_datetime(df.columns)
    return df


def _apply_fill(df: pd.DataFrame, policies: Dict[str, str]) -> pd.DataFrame:
    """Impute missing rows/values according to simple deterministic policies.

    Args:
        df: Statement frame indexed by standardized line-item names.
        policies: Mapping from row name to fill rule (`zero`, `ffill`, `interp`, `constraint`).

    Returns:
        DataFrame with all required rows present and NaNs handled.

    Example:
        >>> policies = {"sales": "interp", "cash": "ffill"}
        >>> data = pd.DataFrame({"2020": [np.nan], "2021": [20.0]}, index=["sales"])
        >>> filled = _apply_fill(data, policies)
        >>> float(filled.loc["sales", "2020"])
        20.0
    """
    # Work with numeric copies to avoid mutating caller data and enforce float ops.
    df = df.copy().apply(pd.to_numeric, errors="coerce")
    # Ensure each required row exists before applying its fill rule.
    for idx, policy in policies.items():
        if idx not in df.index:
            df.loc[idx] = np.nan
        series = df.loc[idx]
        # Apply the requested rule per policy (zero, forward fill, interpolation, constraint).
        if policy == "zero":
            df.loc[idx] = series.fillna(0.0)
        elif policy == "ffill":
            df.loc[idx] = series.sort_index().ffill().bfill().fillna(0.0)
        elif policy == "interp":
            df.loc[idx] = (
                series.sort_index().interpolate(limit_direction="both").bfill().ffill().fillna(0.0)
            )
        elif policy == "constraint":
            # Constraints (balance sheet identities) are enforced by later helpers.
            continue
    # Collapse duplicate indices (happens if the source exported duplicates).
    if not df.index.is_unique:
        df = df.groupby(level=0).mean()
    return df


def _enforce_bs_constraints(bs_df: pd.DataFrame) -> pd.DataFrame:
    """Backfill `assets`, `liab_total`, `equity_total` when two of the three are present.

    Example:
        >>> df = pd.DataFrame({"2021": [100, 40]}, index=["assets", "liab_total"])
        >>> enforced = _enforce_bs_constraints(df)
        >>> float(enforced.loc["equity_total", "2021"])
        60.0
    """
    # Work on a copy so original frames remain untouched.
    bs_df = bs_df.copy()
    # Grab optional series references for each equation component.
    assets = bs_df.loc["assets"] if "assets" in bs_df.index else None
    liab = bs_df.loc["liab_total"] if "liab_total" in bs_df.index else None
    equity = bs_df.loc["equity_total"] if "equity_total" in bs_df.index else None
    # When one element is missing, reconstruct via Assets = Liabilities + Equity.
    if assets is None and liab is not None and equity is not None:
        bs_df.loc["assets"] = liab + equity
    if liab is None and assets is not None and equity is not None:
        bs_df.loc["liab_total"] = assets - equity
    if equity is None and assets is not None and liab is not None:
        bs_df.loc["equity_total"] = assets - liab
    return bs_df


def _derive_dep(is_df: pd.DataFrame, cf_df: pd.DataFrame) -> pd.Series:
    """Use IS depreciation when available, otherwise CF depreciation proxy.

    Example:
        >>> is_df = pd.DataFrame({"2021": [5.0]}, index=["dep"])
        >>> cf_df = pd.DataFrame({"2021": [7.0]}, index=["dep_cf"])
        >>> float(_derive_dep(is_df, cf_df)["2021"])
        5.0
    """
    # Prefer income statement depreciation because it maps directly to drivers.
    if "dep" in is_df.index:
        series = is_df.loc["dep"]
        if isinstance(series, pd.DataFrame):
            series = series.sum(axis=0)
        return series
    # Fallback to cash-flow depreciation proxy (absolute value removes sign flips).
    if "dep_cf" in cf_df.index:
        series = cf_df.loc["dep_cf"]
        if isinstance(series, pd.DataFrame):
            series = series.sum(axis=0)
        return series.abs()
    # If neither exists, default to zeros aligned with the IS columns.
    return pd.Series(0.0, index=is_df.columns)


def _build_states(bs_df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the core 10+2 balance-sheet state vector for one ticker.

    Example:
        >>> bs = pd.DataFrame({"2021": [5, 2]}, index=["cash", "debt_st"])
        >>> states = _build_states(bs)
        >>> float(states.loc["2021", "cash"])
        5.0
    """
    # Build each required state by iterating over STATE_ORDER metadata.
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
        # Guarantee no NaNs so later concatenations remain numeric.
        data[state_name] = series.fillna(0.0)
    states = pd.DataFrame(data)
    # Align with the original chronological order before deriving extras.
    states.index = bs_df.columns
    states.sort_index(inplace=True)
    # Append derived "other" accounts to close the balance-sheet identity.
    states = _append_other_accounts(states, bs_df)
    return states


def _append_other_accounts(states: pd.DataFrame, bs_df: pd.DataFrame) -> pd.DataFrame:
    """Construct the residual 'other' assets and liabilities/equity components.

    Args:
        states: Current core state matrix sorted by period.
        bs_df: Full balance-sheet table with total assets/liabilities/equity.

    Returns:
        States augmented with `other_assets` and `other_liab_equity`.

    Example:
        >>> core = pd.DataFrame({"cash": [5.0]}, index=pd.to_datetime(["2021"]))
        >>> bs = pd.DataFrame({"2021": [15.0, 15.0]}, index=["assets", "liab_total"])
        >>> augmented = _append_other_accounts(core, bs)
        >>> float(augmented.loc["2021-01-01", "other_assets"])
        10.0
    """
    states = states.copy()

    # Helper to normalize rows to Series regardless of Yahoo quirks.
    def _to_series(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
        if key not in df.index:
            return None
        series = df.loc[key]
        if isinstance(series, pd.DataFrame):
            series = series.sum(axis=0)
        return series

    # Pull the total assets/liabilities/equity, backfilling when missing.
    assets_full = _to_series(bs_df, "assets")
    liab_full = _to_series(bs_df, "liab_total")
    equity_full = _to_series(bs_df, "equity_total")

    if assets_full is None and liab_full is not None and equity_full is not None:
        assets_full = liab_full + equity_full
    if assets_full is None:
        assets_full = pd.Series(0.0, index=states.index)
    else:
        assets_full = assets_full.reindex(states.index).ffill().bfill().fillna(0.0)

    if liab_full is not None and equity_full is not None:
        liab_equity_full = (liab_full + equity_full).reindex(states.index).ffill().bfill().fillna(0.0)
    else:
        liab_equity_full = assets_full.copy()

    # Compute the portion already represented by explicit state columns.
    core_assets = (states[["cash", "inv_short", "ppe_net", "ar", "inv_stock"]]).sum(axis=1)
    core_liab_equity = (states[["debt_st", "debt_lt", "ap", "pic", "re"]]).sum(axis=1)

    # Residual "other" accounts ensure the totals still match.
    other_assets = (assets_full - core_assets).fillna(0.0)
    other_liab_equity = (liab_equity_full - core_liab_equity).fillna(0.0)

    states["other_assets"] = other_assets
    states["other_liab_equity"] = other_liab_equity
    return states


def _assign_splits(index: pd.DatetimeIndex,
                   train_end_year: Optional[int],
                   val_end_year: Optional[int],
                   train_frac: float,
                   val_frac: float) -> pd.Series:
    """Label each timestamp as train/val/test in chronological order.

    Args:
        index: Periods for one ticker.
        train_end_year: Optional calendar cutoff for the training window.
        val_end_year: Optional calendar cutoff for the validation window.
        train_frac: Fallback fraction for train share when calendar cutoffs absent.
        val_frac: Fallback fraction for validation share.

    Returns:
        Series aligned with `index` containing `train`/`val`/`test`.
    
    Example:
        >>> idx = pd.to_datetime(["2018", "2019", "2020", "2021"])
        >>> labels = _assign_splits(idx, train_end_year=2019, val_end_year=2020, train_frac=0.5, val_frac=0.25)
        >>> labels.tolist()
        ['train', 'train', 'val', 'test']
    """
    if len(index) == 0:
        return pd.Series(dtype="object")
    # Work with sorted copies to ensure chronological splits.
    sorted_idx = index.sort_values()
    years = pd.Series(sorted_idx.year, index=sorted_idx)

    def _assign_by_fraction() -> pd.Series:
        n = len(sorted_idx)
        if n == 1:
            return pd.Series(["train"], index=sorted_idx)
        if n == 2:
            return pd.Series(["train", "test"], index=sorted_idx)
        # Translate requested fractions into integer counts.
        train_count = max(1, int(round(n * train_frac)))
        val_count = max(1, int(round(n * val_frac)))
        if train_count + val_count >= n:
            val_count = max(1, min(val_count, n - 1))
            train_count = max(1, n - val_count - 1)
        labels = ["train"] * train_count + ["val"] * val_count + ["test"] * (n - train_count - val_count)
        series = pd.Series(labels, index=sorted_idx)
        return series

    # Prefer deterministic year cutoffs when available.
    use_years = (train_end_year is not None) or (val_end_year is not None)
    if use_years:
        train_cut = train_end_year if train_end_year is not None else years.min()
        val_cut = val_end_year if val_end_year is not None else train_cut
        train_mask = years <= train_cut
        val_mask = (years > train_cut) & (years <= val_cut)
        labels = pd.Series("test", index=sorted_idx)
        if train_mask.any():
            labels.loc[train_mask.index[train_mask]] = "train"
        if val_mask.any():
            labels.loc[val_mask.index[val_mask]] = "val"
        has_train = (labels == "train").any()
        has_val = (labels == "val").any()
        has_test = (labels == "test").any()
        if has_train and has_val and has_test:
            return labels
    # Fall back to fraction-based assignment when dates are insufficient.
    return _assign_by_fraction()


def _safe_div(num: pd.Series, denom: pd.Series, default: float) -> pd.Series:
    """Divide with graceful fallback when denominator hits zero.

    Example:
        >>> num = pd.Series([10.0, 5.0])
        >>> denom = pd.Series([2.0, 0.0])
        >>> _safe_div(num, denom, default=1.0).tolist()
        [5.0, 1.0]
    """
    # Start with the default value everywhere.
    ratio = num.copy()
    ratio[:] = default
    # Replace zeros before division to avoid warnings.
    valid = denom.replace(0.0, np.nan)
    # Perform the actual division, then backfill missing entries with the default.
    ratio = (num / valid).fillna(default)
    return ratio


def _row_or_default(df: pd.DataFrame, name: str, index: pd.Index, default: float = 0.0) -> pd.Series:
    """Return row `name` if it exists, otherwise a constant-valued Series.

    Example:
        >>> df = pd.DataFrame({"2020": [3.0]}, index=["sales"])
        >>> _row_or_default(df, "cogs", df.columns, default=1.0)["2020"]
        1.0
    """
    if name in df.index:
        series = df.loc[name]
        if isinstance(series, pd.DataFrame):
            series = series.sum(axis=0)
        return series
    return pd.Series(default, index=index)


def _logit_transform(series: pd.Series, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
    """Apply a bounded logit so that ratios remain finite.

    Example:
        >>> data = pd.Series([0.2, 0.8])
        >>> transformed = _logit_transform(data, 0.0, 1.0)
        >>> transformed.round(2).tolist()
        [-1.39, 1.39]
    """
    # Compute span to normalize the range into (0, 1).
    span = upper - lower
    # Clip extreme ratios to avoid infinite logits.
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
    """Assemble exogenous drivers (flows + policy ratios) for one ticker.

    Example:
        >>> idx = pd.to_datetime(["2020"])
        >>> is_df = pd.DataFrame({"2020": [100, 60, 20, 10]}, index=["sales", "cogs", "opex", "dep"])
        >>> bs_df = pd.DataFrame({"2020": [5, 8]}, index=["debt_st", "debt_lt"])
        >>> cf_df = pd.DataFrame({"2020": [-15]}, index=["capex"])
        >>> states = pd.DataFrame({"cash": [10.0], "inv_short": [2.0]}, index=idx)
        >>> drivers = _build_drivers("TEST", is_df, bs_df, cf_df, states)
        >>> drivers.loc["2020", "Sales"]
        100.0
    """
    # Operate on copies so we can mutate without surprising callers.
    is_df = is_df.copy()
    cf_df = cf_df.copy()
    # Capex is negative cash flow; flip the sign and keep only positive spend.
    capex = (-_row_or_default(cf_df, "capex", is_df.columns)).clip(lower=0.0)
    # Depreciation is stitched together from IS/CF (ensuring absolute value).
    dep = _derive_dep(is_df, cf_df)
    # Feature engineering from the separate module (e.g., DSO, target cash).
    features = compute_drivers(is_df, bs_df, cf_df)
    default_idx = is_df.columns
    # Default to reasonable WC days when features are unavailable.
    dso = features.get("DSO", pd.Series(45.0, index=default_idx)).replace([np.inf, -np.inf], np.nan).fillna(45.0)
    dpo = features.get("DPO", pd.Series(50.0, index=default_idx)).replace([np.inf, -np.inf], np.nan).fillna(50.0)
    dio = features.get("DIO", pd.Series(60.0, index=default_idx)).replace([np.inf, -np.inf], np.nan).fillna(60.0)
    # Cash buffer target ratio multiplied by absolute sales yields nominal dollars.
    cbar = (features.get("cash_target_ratio", pd.Series(0.05, index=is_df.columns)) * is_df.loc["sales"].abs()).fillna(0.0)

    # Interest expense/income plus debt balances produce cost-of-debt proxies.
    int_exp = _row_or_default(is_df, "int_exp", is_df.columns).abs()
    int_inc = _row_or_default(is_df, "int_inc", is_df.columns).abs()
    debt_st = _row_or_default(bs_df, "debt_st", is_df.columns).replace(0.0, np.nan)
    debt_lt = _row_or_default(bs_df, "debt_lt", is_df.columns).replace(0.0, np.nan)
    rST = (int_exp * 0.35 / debt_st).replace([np.inf, -np.inf], np.nan).fillna(0.05)
    rLT = (int_exp * 0.65 / debt_lt).replace([np.inf, -np.inf], np.nan).fillna(0.04)
    rINV = _safe_div(int_inc, states["inv_short"] + 1e-6, 0.02)

    # Financing flows that describe how debt/equity gets serviced.
    amort_lt = _row_or_default(cf_df, "lt_debt_payments", is_df.columns).abs()
    equity_inj = (_row_or_default(cf_df, "equity_issuance", is_df.columns) - _row_or_default(cf_df, "buyback", is_df.columns).abs()).clip(lower=0.0)
    dividends = _row_or_default(cf_df, "dividends_paid", is_df.columns).abs()

    # Tax rate proxy derived from ebt vs actual tax payments.
    ebt = is_df.loc["ebt"] if "ebt" in is_df.index else _row_or_default(is_df, "ni", is_df.columns)
    taxes = _row_or_default(is_df, "tax", is_df.columns).fillna(0.0)
    tau = _safe_div(taxes, ebt.replace(0.0, np.nan), 0.25).clip(0.0, 0.45)

    # Opex missing for some companies; approximate it when required.
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
    # Consolidate all drivers into one aligned DataFrame.
    drivers = pd.concat(driver_dict, axis=1)
    drivers.sort_index(inplace=True)
    drivers.fillna(0.0, inplace=True)
    # Some tickers miss exotic drivers; insert zero columns to maintain schema.
    missing_cols = [col for col in DRIVER_COLUMNS if col not in drivers.columns]
    for col in missing_cols:
        drivers[col] = 0.0
    drivers = drivers[DRIVER_COLUMNS]
    return drivers


def _determine_scale(is_df: pd.DataFrame,
                     bs_df: pd.DataFrame,
                     train_index: pd.DatetimeIndex) -> float:
    """Estimate a single scale factor (median sales/assets) using only train periods.

    Example:
        >>> idx = pd.to_datetime(["2020", "2021"])
        >>> is_df = pd.DataFrame([100.0, 120.0], index=["sales"], columns=idx)
        >>> bs_df = pd.DataFrame([200.0, 220.0], index=["assets"], columns=idx)
        >>> _determine_scale(is_df, bs_df, train_index=idx)
        170.0
    """
    if train_index is None or len(train_index) == 0:
        raise ValueError("Training index is required to determine scale without leakage.")

    def _subset(series: Optional[pd.Series]) -> Optional[pd.Series]:
        if series is None:
            return None
        # Restrict to training periods and drop NaNs to avoid contaminating medians.
        sub = series.reindex(train_index).dropna()
        if sub.empty:
            return None
        return sub

    # Collect candidate scale proxies from both statements when available.
    sales = _subset(is_df.loc["sales"] if "sales" in is_df.index else None)
    assets = _subset(bs_df.loc["assets"] if "assets" in bs_df.index else None)
    values = []
    if sales is not None:
        values.append(sales.abs().median())
    if assets is not None:
        values.append(assets.abs().median())
    if not values:
        raise ValueError("Unable to compute scale from training window.")
    # Pick the maximum to bias toward more conservative scaling.
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
    """Generate secondary ratios (growth, capacity, buffer) that are global-normalized.

    Example:
        >>> idx = pd.to_datetime(["2020"])
        >>> drivers = pd.DataFrame({"COGS": [60.0], "Opex": [20.0], "DPO": [40.0], "DSO": [50.0], "DIO": [70.0]}, index=idx)
        >>> states = pd.DataFrame({"cash": [10.0], "ppe_net": [30.0]}, index=idx)
        >>> covs = _compute_covariates(drivers, states, sales=pd.Series([100.0], index=idx), capex=pd.Series([15.0], index=idx), dep=pd.Series([10.0], index=idx), cbar=pd.Series([5.0], index=idx), dividends=pd.Series([2.0], index=idx), taxes=pd.Series([3.0], index=idx))
        >>> covs.columns.tolist()
        ['Capex_Sales', 'Capex_PPE', 'Capex_Dep', 'Capex_Cashbound', 'Cash_Buffer_Ratio', 'DSO_Share', 'DPO_Share', 'DIO_Share']
    """
    eps = 1e-6
    cov = pd.DataFrame(index=drivers.index)
    capex_abs = capex.abs()
    # Express Capex relative to key magnitudes (sales/PPE/dep) to capture intensity.
    cov["Capex_Sales"] = (capex_abs / (sales.abs() + eps)).clip(0.0, 1.0)
    cov["Capex_PPE"] = (capex_abs / (states["ppe_net"].abs() + eps)).clip(0.0, 1.0)
    cov["Capex_Dep"] = (capex_abs / (dep.abs() + eps)).clip(0.0, 2.0)
    # Estimate a capex affordability ceiling using cash + CFO proxy.
    cfo_proxy = (sales - drivers["COGS"] - drivers["Opex"] - taxes).fillna(0.0)
    capex_upper = (states["cash"] + cfo_proxy - dividends - cbar).clip(lower=eps)
    cov["Capex_Cashbound"] = (capex_abs / (capex_upper + eps)).clip(0.0, 1.2)
    # Cash buffer ratio measures resilience vs ongoing costs.
    operating_cost = (drivers["COGS"].abs() + drivers["Opex"].abs() + eps)
    cov["Cash_Buffer_Ratio"] = (states["cash"] / operating_cost).clip(0.0, 2.0)
    # Normalize working-capital days into 0-1 shares for modeling convenience.
    cov["DSO_Share"] = (drivers["DSO"] / 365.0).clip(0.0, 1.0)
    cov["DPO_Share"] = (drivers["DPO"] / 365.0).clip(0.0, 1.0)
    cov["DIO_Share"] = (drivers["DIO"] / 365.0).clip(0.0, 1.0)
    # Remove infinities and default to zeros in rare edge cases.
    cov.replace([np.inf, -np.inf], np.nan, inplace=True)
    cov.fillna(0.0, inplace=True)
    return cov


def _augment_with_state_history(states: pd.DataFrame,
                                covariates: pd.DataFrame,
                                rolling_window: int = 4) -> pd.DataFrame:
    """Append lag, rolling mean, and lag/mean ratios (input to the driver MLP).

    Args:
        states: Cleaned state trajectories.
        covariates: Driver covariates before augmentation.
        rolling_window: Number of past periods to average (uses causal rolling mean).

    Returns:
        Covariate frame with lag/rolling features aligned with `states`.

    Example:
        >>> idx = pd.to_datetime(["2020", "2021"])
        >>> states = pd.DataFrame({"cash": [5.0, 7.0]}, index=idx)
        >>> covs = pd.DataFrame({"Capex_Sales": [0.1, 0.2]}, index=idx)
        >>> augmented = _augment_with_state_history(states, covs, rolling_window=2)
        >>> sorted([col for col in augmented.columns if col.startswith("lag_cash")])[0]
        'lag_cash'
    """
    rolling_window = max(1, int(rolling_window))
    # Shift by one period so models only see historical state information.
    lagged = states.shift(1)
    # Rolling mean summarizes the trailing trajectory in a causal manner.
    rolling = lagged.rolling(window=rolling_window, min_periods=1).mean()
    lagged = lagged.fillna(0.0)
    rolling = rolling.fillna(0.0)

    eps = 1e-6
    # Compare lagged values against trend to highlight sudden changes.
    ratio = lagged / (rolling.abs() + eps)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Construct new feature columns for lag, rolling mean, and ratios.
    lag_cols = {f"lag_{col}": lagged[col] for col in states.columns}
    roll_cols = {f"roll_mean_{rolling_window}_{col}": rolling[col] for col in states.columns}
    ratio_cols = {f"lag_to_roll_{col}": ratio[col] for col in states.columns}
    extra = pd.DataFrame({**lag_cols, **roll_cols, **ratio_cols})
    extra = extra.reindex(covariates.index).fillna(0.0)
    covariates = pd.concat([covariates, extra], axis=1)
    return covariates


def _normalize(states: pd.DataFrame,
               drivers: pd.DataFrame,
               method: str,
               scale: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series]]:
    """Normalize states/drivers either via per-ticker scale or z-score (legacy).

    Example:
        >>> states = pd.DataFrame({"cash": [10.0, 20.0]})
        >>> drivers = pd.DataFrame({"Sales": [100.0, 150.0], "COGS": [60.0, 90.0]})
        >>> s_norm, d_norm, meta = _normalize(states, drivers, method="scale", scale=100.0)
        >>> float(s_norm.iloc[0]["cash"])
        0.1
    """
    meta: Dict[str, pd.Series] = {}
    if method == "zscore":
        # Compute per-column shift/scale to throw away units.
        states_mean = states.mean()
        states_std = states.std().replace(0.0, 1.0)
        meta["state_shift"] = states_mean
        meta["state_scale"] = states_std
        states_norm = (states - states_mean) / states_std
        drivers_mean = drivers.mean()
        drivers_std = drivers.std().replace(0.0, 1.0)
        drivers_norm = (drivers - drivers_mean) / drivers_std
    else:  # scale
        # Share a single scale for every state dimension to retain relative ratios.
        states_shift = pd.Series(0.0, index=states.columns)
        states_scale = pd.Series(scale, index=states.columns)
        meta["state_shift"] = states_shift
        meta["state_scale"] = states_scale
        states_norm = states / scale
        # Only flow-like columns need scaling; policy ratios already unitless.
        flow_cols = ["Sales", "COGS", "Opex", "Dep", "Capex", "AmortLT", "Cbar", "EI", "Div"]
        drivers_norm = drivers.copy()
        drivers_norm[flow_cols] = drivers_norm[flow_cols] / scale
    return states_norm, drivers_norm, meta


def preprocess_ticker(ticker: str,
                      raw_dir: Path,
                      proc_dir: Path,
                      frequency: str,
                      norm_method: str,
                      state_history_window: int,
                      train_end_year: Optional[int],
                      val_end_year: Optional[int],
                      train_frac: float,
                      val_frac: float) -> Optional[TickerData]:
    """Full per-ticker preprocessing pipeline used by CLI scripts.

    Args:
        ticker: Equity ticker symbol.
        raw_dir: Folder containing downloaded Yahoo statements.
        proc_dir: Where normalized assets/meta should be persisted.
        frequency: `annual` or `quarterly`.
        norm_method: Currently `scale` (default) or `zscore` for legacy runs.
        state_history_window: Rolling window size for lag/rolling features.
        train_end_year: Optional explicit cutoff for the training portion.
        val_end_year: Optional explicit cutoff for validation portion.
        train_frac: Fallback training share when cutoffs absent.
        val_frac: Fallback validation share when cutoffs absent.

    Returns:
        TickerData bundle with normalized states, covariates, and masks; `None` on failure.

    Example:
        >>> td = preprocess_ticker("AAPL", Path("data/raw"), Path("data/proc"), "annual", "scale", 4, None, None, 0.6, 0.2)  # doctest: +SKIP
        >>> td.ticker  # doctest: +SKIP
        'AAPL'
    """
    try:
        # Load the three core financial statements.
        is_path = raw_dir / f"{ticker}_IS_{frequency}.csv"
        bs_path = raw_dir / f"{ticker}_BS_{frequency}.csv"
        cf_path = raw_dir / f"{ticker}_CF_{frequency}.csv"
        is_df = standardize_columns(_read_statement(is_path), "IS")
        bs_df = standardize_columns(_read_statement(bs_path), "BS")
        cf_df = standardize_columns(_read_statement(cf_path), "CF")
    except FileNotFoundError as err:
        logger.warning("%s", err)
        return None

    # Bail if Yahoo delivered an empty CSV.
    for name, frame in (("IS", is_df), ("BS", bs_df), ("CF", cf_df)):
        if frame.empty or frame.shape[1] == 0:
            logger.warning("%s statement for %s is empty; skipping ticker", name, ticker)
            return None

    for name, frame in (("IS", is_df), ("BS", bs_df), ("CF", cf_df)):
        if frame.empty or frame.shape[1] == 0:
            logger.warning("%s statement for %s is empty; skipping ticker", name, ticker)
            return None

    # Deterministic imputations bring statements to a common schema.
    is_df = _apply_fill(is_df, FILL_POLICIES_IS)
    bs_df = _enforce_bs_constraints(_apply_fill(bs_df, FILL_POLICIES_BS))
    cf_df = _apply_fill(cf_df, FILL_POLICIES_CF)

    # Build states/drivers and synchronize them on a shared index.
    states = _build_states(bs_df)
    drivers = _build_drivers(ticker, is_df, bs_df, cf_df, states)
    split_labels = _assign_splits(states.index, train_end_year, val_end_year, train_frac, val_frac)
    common_index = states.index.intersection(drivers.index)
    states = states.loc[common_index]
    drivers = drivers.loc[common_index]
    split_labels = split_labels.loc[common_index]
    transition_labels = split_labels.iloc[1:].copy()
    train_index = split_labels[split_labels == "train"].index
    if len(train_index) == 0:
        logger.warning("No training samples for %s before %s; skipping ticker.", ticker, train_end_year)
        return None

    # Derive scale and normalization metadata exclusively from training periods.
    scale = _determine_scale(is_df, bs_df, train_index)
    states_norm, drivers_norm, norm_meta = _normalize(states, drivers, norm_method, scale)
    # Build auxiliary covariates, reusing original (non-normalized) flows where required.
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
    covariates = _augment_with_state_history(states_norm, covariates, rolling_window=state_history_window)
    covariates = covariates.loc[common_index]

    # Persist intermediate CSVs/metadata for debugging or downstream analysis.
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
        splits=split_labels,
        transition_splits=transition_labels,
    )


def _windows_with_split_masks(states: pd.DataFrame,
                              covariates: pd.DataFrame,
                              transition_splits: pd.Series,
                              seq_len: int) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """Create sliding windows plus per-split transition masks to avoid leakage.

    Args:
        states: Normalized state trajectories sorted in time.
        covariates: Driver features aligned with `states`.
        transition_splits: Series with length `len(states) - 1`, label of each transition.
        seq_len: Total steps per window (history + horizon).

    Returns:
        Dict per split with lists for `states`, `covs`, `targets`, and `mask`.

    Example:
        >>> idx = pd.to_datetime(["2020", "2021", "2022"])
        >>> states = pd.DataFrame({"cash": [1.0, 2.0, 3.0]}, index=idx)
        >>> covs = states.copy()
        >>> splits = pd.Series(["train", "test"], index=idx[1:])
        >>> windows = _windows_with_split_masks(states, covs, splits, seq_len=2)
        >>> len(windows["train"]["states"])
        1
    """
    splits = ("train", "val", "test")
    result = {split: {"states": [], "covs": [], "targets": [], "mask": []} for split in splits}
    if len(states) < seq_len or len(transition_splits) == 0:
        return result
    # Convert to numpy for efficient slicing inside sliding window loop.
    states_np = states.to_numpy(dtype=np.float32)
    covs_np = covariates.to_numpy(dtype=np.float32)
    transition_values = transition_splits.to_numpy(dtype=object)
    max_start = len(states) - seq_len + 1
    transitions_per_window = seq_len - 1
    for start in range(max_start):
        end = start + seq_len
        trans_slice = transition_values[start:start + transitions_per_window]
        if len(trans_slice) != transitions_per_window:
            continue
        state_window = states_np[start:end]
        cov_window = covs_np[start:end]
        for split in splits:
            # Mask shows which transitions in the window belong to the split.
            mask = (trans_slice == split).astype(np.float32)
            if mask.sum() <= 0:
                continue
            result[split]["states"].append(state_window)
            result[split]["covs"].append(cov_window)
            result[split]["targets"].append(state_window.copy())
            result[split]["mask"].append(mask)
    return result


def _assemble_training_dataset(
    ticker_data: List[TickerData], seq_len: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """Stack ticker-level sliding windows plus masks into numpy arrays.

    Example:
        >>> idx = pd.to_datetime(["2020", "2021", "2022"])
        >>> states = pd.DataFrame({"cash": [0.1, 0.2, 0.3]}, index=idx)
        >>> drivers = states.copy()
        >>> td = TickerData("T1", states, drivers, 1.0, states.iloc[0], states.iloc[0].abs() + 1, pd.Series(["train", "train", "test"], index=idx), pd.Series(["train", "test"], index=idx[1:]))
        >>> dataset = _assemble_training_dataset([td], seq_len=2)
        >>> dataset["states_train"].shape[0] >= 1
        True
    """
    if not ticker_data:
        raise RuntimeError("No ticker data provided to assemble the training dataset.")

    splits = ("train", "val", "test")
    split_states = {split: [] for split in splits}
    split_covs = {split: [] for split in splits}
    split_targets = {split: [] for split in splits}
    split_masks = {split: [] for split in splits}
    split_tickers = {split: [] for split in splits}
    split_shifts = {split: [] for split in splits}
    split_scales = {split: [] for split in splits}
    transition_counts: Dict[str, Dict[str, int]] = {}

    for item in ticker_data:
        # Align covariates/states and derive sliding windows per split.
        states_df = item.states.sort_index()
        covs_df = item.drivers.sort_index().reindex(states_df.index)
        covs_df.fillna(0.0, inplace=True)
        transitions_series = item.transition_splits.reindex(states_df.index[1:])
        windows_by_split = _windows_with_split_masks(states_df, covs_df, transitions_series, seq_len)
        shift_arr = item.state_shift.to_numpy(dtype=np.float32)
        scale_arr = item.state_scale.to_numpy(dtype=np.float32)
        ticker_counts = {split: 0 for split in splits}
        for split in splits:
            payload = windows_by_split.get(split, {})
            states_windows = payload.get("states", [])
            if not states_windows:
                continue
            covs_windows = payload["covs"]
            targets_windows = payload["targets"]
            masks_windows = payload["mask"]
            # Append windows and metadata for the split.
            split_states[split].extend(states_windows)
            split_covs[split].extend(covs_windows)
            split_targets[split].extend(targets_windows)
            split_masks[split].extend(masks_windows)
            split_tickers[split].extend([item.ticker] * len(states_windows))
            split_shifts[split].extend([shift_arr.copy() for _ in range(len(states_windows))])
            split_scales[split].extend([scale_arr.copy() for _ in range(len(states_windows))])
            ticker_counts[split] += int(sum(float(np.sum(mask)) for mask in masks_windows))
        if any(ticker_counts.values()):
            transition_counts[item.ticker] = ticker_counts

    if not any(split_states[split] for split in splits):
        raise RuntimeError("No ticker has enough history to assemble the training dataset.")

    state_columns = list(ticker_data[0].states.columns)
    cov_columns = list(ticker_data[0].drivers.columns)

    def _stack_sequences(sequences: List[np.ndarray], default_shape: Tuple[int, int]) -> np.ndarray:
        if sequences:
            return np.stack(sequences, axis=0).astype(np.float32)
        return np.empty((0, *default_shape), dtype=np.float32)

    def _stack_meta(meta: List[np.ndarray], state_dim: int) -> np.ndarray:
        if meta:
            return np.stack(meta, axis=0).astype(np.float32)
        return np.empty((0, state_dim), dtype=np.float32)

    mask_len = seq_len - 1
    assembled = {
        "state_columns": state_columns,
        "cov_columns": cov_columns,
        "transition_counts": transition_counts,
        "mask_len": mask_len,
    }
    for split in splits:
        states_list = split_states[split]
        covs_list = split_covs[split]
        targets_list = split_targets[split]
        default_state_shape = states_list[0].shape if states_list else (seq_len, len(state_columns))
        default_cov_shape = covs_list[0].shape if covs_list else (seq_len, len(cov_columns))
        # Stack into contiguous tensors for the Trainer dataloader.
        assembled[f"states_{split}"] = _stack_sequences(states_list, default_state_shape)
        assembled[f"covs_{split}"] = _stack_sequences(covs_list, default_cov_shape)
        assembled[f"targets_{split}"] = _stack_sequences(targets_list, default_state_shape)
        tickers = split_tickers[split]
        assembled[f"tickers_{split}"] = np.array(tickers, dtype="U12") if tickers else np.empty((0,), dtype="U12")
        assembled[f"state_shift_{split}"] = _stack_meta(split_shifts[split], len(state_columns))
        assembled[f"state_scale_{split}"] = _stack_meta(split_scales[split], len(state_columns))
        masks = split_masks[split]
        if masks:
            assembled[f"mask_{split}"] = np.stack(masks, axis=0).astype(np.float32)
        else:
            assembled[f"mask_{split}"] = np.empty((0, mask_len), dtype=np.float32)

    return assembled


def run_preprocessing_pipeline(config_path: Optional[str] = None,
                               variant: str = "base",
                               norm_method: str = "scale",
                               override_frequency: Optional[str] = None) -> Path:
    """CLI-friendly preprocessing entry point. Returns the npz path for downstream training.

    Example:
        >>> dataset_path = run_preprocessing_pipeline("configs/config.yaml", variant="debug")  # doctest: +SKIP
        >>> dataset_path.name  # doctest: +SKIP
        'training_data.npz'
    """
    # Read configuration (tickers, paths, training params).
    cfg_path = config_path or get_default_config_path()
    cfg = load_config(cfg_path)
    frequency = override_frequency or cfg.frequency
    raw_dir = Path(cfg.paths["raw_dir"])
    proc_root = Path(cfg.paths["proc_dir"])
    proc_dir = proc_root / variant
    ensure_dir(str(proc_dir))
    ticker_results: List[TickerData] = []
    training_cfg = cfg.training or {}
    # Hyperparameters for sliding window assembly.
    window = int(training_cfg.get("window", 6))
    horizon = int(training_cfg.get("horizon", 1))
    rolling_window = int(training_cfg.get("rolling_mean_window", 4))
    splits_cfg = training_cfg.get("splits") or {}
    train_end_year = splits_cfg.get("train_end_year")
    val_end_year = splits_cfg.get("val_end_year")
    train_end_year = int(train_end_year) if train_end_year is not None else None
    val_end_year = int(val_end_year) if val_end_year is not None else None
    if (train_end_year is not None and val_end_year is not None) and (val_end_year < train_end_year):
        raise ValueError("val_end_year must be >= train_end_year in training.splits configuration.")
    train_frac = float(splits_cfg.get("train_frac", 0.6))
    val_frac = float(splits_cfg.get("val_frac", 0.2))
    # Preprocess each ticker individually and collect successful outputs.
    for ticker in cfg.tickers:
        logger.info("Preprocessing %s", ticker)
        td = preprocess_ticker(
            ticker,
            raw_dir,
            proc_dir,
            frequency,
            norm_method,
            state_history_window=rolling_window,
            train_end_year=train_end_year,
            val_end_year=val_end_year,
            train_frac=train_frac,
            val_frac=val_frac,
        )
        if td:
            ticker_results.append(td)
    seq_len = int(window + horizon)
    # Convert ticker-level outputs into numpy arrays w/ masks per split.
    assembled = _assemble_training_dataset(ticker_results, seq_len)
    splits = ("train", "val", "test")
    dataset_path = proc_dir / "training_data.npz"
    payload = {
        "state_columns": np.array(assembled["state_columns"], dtype="U24"),
        "cov_columns": np.array(assembled["cov_columns"], dtype="U60"),
        "mask_len": np.int32(assembled["mask_len"]),
    }
    for split in splits:
        payload[f"states_{split}"] = assembled[f"states_{split}"]
        payload[f"covs_{split}"] = assembled[f"covs_{split}"]
        payload[f"targets_{split}"] = assembled[f"targets_{split}"]
        payload[f"tickers_{split}"] = assembled[f"tickers_{split}"]
        payload[f"state_shift_{split}"] = assembled[f"state_shift_{split}"]
        payload[f"state_scale_{split}"] = assembled[f"state_scale_{split}"]
        payload[f"mask_{split}"] = assembled[f"mask_{split}"]
    np.savez(dataset_path, **payload)
    split_counts = {split: int(payload[f"states_{split}"].shape[0]) for split in splits}
    num_sequences = int(sum(split_counts.values()))
    all_tickers = sorted(
        set(
            ticker
            for split in splits
            for ticker in assembled[f"tickers_{split}"].tolist()
        )
    )
    summary = {
        "tickers": all_tickers,
        "transitions_per_ticker": assembled["transition_counts"],
        "seq_len_requested": seq_len,
        "seq_len_effective": seq_len,
        "num_samples": num_sequences,
        "split_counts": split_counts,
        "mask_len": assembled["mask_len"],
        "covariate_columns": assembled["cov_columns"],
        "state_columns": assembled["state_columns"],
        "variant": variant,
        "norm_method": norm_method,
        "frequency": frequency,
        "window": window,
        "horizon": horizon,
        "rolling_mean_window": rolling_window,
        "train_end_year": train_end_year,
        "val_end_year": val_end_year,
        "train_frac": train_frac,
        "val_frac": val_frac,
    }
    (proc_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Saved normalized dataset to %s", dataset_path)
    return dataset_path
