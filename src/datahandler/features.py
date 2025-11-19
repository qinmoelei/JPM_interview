from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

def compute_drivers(is_df: pd.DataFrame, bs_df: pd.DataFrame, cf_df: pd.DataFrame) -> pd.DataFrame:
    """Compute driver series: EBITDA margin, cost ratios, Capex ratio, and working-capital days (rough).

    Example:
        >>> idx = pd.to_datetime(["2020"])
        >>> is_df = pd.DataFrame({"2020": [100.0, 60.0]}, index=["sales", "cogs"])
        >>> bs_df = pd.DataFrame({"2020": [5.0, 4.0, 3.0]}, index=["cash", "ar", "ap"])
        >>> cf_df = pd.DataFrame({"2020": [-20.0]}, index=["capex"])
        >>> compute_drivers(is_df, bs_df, cf_df).columns[:2].tolist()
        ['ebitda_margin', 'cogs_ratio']
    """
    df = pd.DataFrame(index=is_df.columns)
    sales = is_df.loc["sales"] if "sales" in is_df.index else None
    cogs = is_df.loc["cogs"] if "cogs" in is_df.index else None
    dep  = is_df.loc["dep"] if "dep" in is_df.index else None
    opex = None
    for candidate in ["opex", "operating_expense", "operating_expenses", "selling_general_administrative"]:
        if candidate in is_df.index:
            opex = is_df.loc[candidate]; break
    if sales is not None and cogs is not None:
        gross_profit = sales - cogs
    else:
        gross_profit = None

    # EBITDA proxy
    if sales is not None and cogs is not None and opex is not None:
        ebitda = sales - cogs - opex
        df["ebitda_margin"] = (ebitda / sales).replace([np.inf, -np.inf], np.nan)
    # Ratios
    if sales is not None and cogs is not None:
        df["cogs_ratio"] = (cogs / sales).replace([np.inf, -np.inf], np.nan)
    if sales is not None and opex is not None:
        df["opex_ratio"] = (opex / sales).replace([np.inf, -np.inf], np.nan)

    # Capex ratio using CF (Yahoo CF capex is negative outflow)
    if "capex" in cf_df.index and sales is not None:
        df["capex_ratio"] = (-cf_df.loc["capex"] / sales).replace([np.inf, -np.inf], np.nan)

    # Working capital days (rough, end-of-period stocks against annual flows)
    try:
        ar = bs_df.loc["ar"]; ap = bs_df.loc["ap"]; inv = bs_df.loc["inv_stock"]
        cogs_flow = cogs if cogs is not None else None
        if sales is not None:
            df["DSO"] = 365.0 * ar / sales
        if cogs_flow is not None and cogs_flow.abs().max() > 0:
            df["DPO"] = 365.0 * ap / cogs_flow
            df["DIO"] = 365.0 * inv / cogs_flow
    except KeyError:
        pass

    # Cash target as a fraction of sales (policy proxy)
    if sales is not None and "cash" in bs_df.index:
        cash = bs_df.loc["cash"]
        df["cash_target_ratio"] = (cash / sales).clip(lower=0.0, upper=1.0)

    return df
