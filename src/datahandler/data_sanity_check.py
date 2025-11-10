from __future__ import annotations
import pandas as pd
from typing import Dict, Optional

# Mapping tables for Yahoo Finance line item name variants
MAP_IS = {
    "Total Revenue": "sales",
    "Operating Revenue": "sales",
    "Cost Of Revenue": "cogs",
    "Cost of Revenue": "cogs",
    "Operating Expense": "opex",
    "Operating Expenses": "opex",
    "Selling General Administrative": "sga",
    "Depreciation And Amortization": "dep",
    "Depreciation & amortization": "dep",
    "Pretax Income": "ebt",
    "Net Income": "ni",
    "Interest Expense": "int_exp",
    "Interest Income": "int_inc"
}

MAP_BS = {
    "Cash And Cash Equivalents": "cash",
    "Cash And Cash Equivalents And Short Term Investments": "cash_and_sti",
    "Short Term Investments": "sti",
    "Property Plant Equipment Net": "ppe_net",
    "Total Assets": "assets",
    "Total Liab": "liab_total",
    "Total Stockholder Equity": "equity_total",
    "Short Long Term Debt": "debt_st",
    "Short Term Debt": "debt_st",
    "Long Term Debt": "debt_lt",
    "Accounts Receivable": "ar",
    "Net Receivables": "ar",
    "Accounts Payable": "ap",
    "Inventory": "inv_stock",
    "Retained Earnings": "re",
    "Common Stock": "pic",
    "Capital Stock": "pic",
    "Additional Paid In Capital": "apic"
}

MAP_CF = {
    "Capital Expenditure": "capex",
    "Depreciation": "dep_cf",
    "Depreciation And Amortization": "dep_cf",
    "Investments": "investments_cf",
    "Change In Working Capital": "delta_wc_cf",
    "Repurchase Of Stock": "buyback",
    "Issuance Of Stock": "equity_issuance",
    "Long Term Debt Issuance": "lt_debt_issuance",
    "Long Term Debt Payments": "lt_debt_payments",
    "Dividends Paid": "dividends_paid"
}

def standardize_columns(df: pd.DataFrame, statement: str) -> pd.DataFrame:
    """Rename common lines to standardized keys (best-effort) and lowercase the rest."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if statement == "IS":
        mapper = MAP_IS
    elif statement == "BS":
        mapper = MAP_BS
    elif statement == "CF":
        mapper = MAP_CF
    else:
        mapper = {}
    new_index = []
    for idx in df.index:
        std = mapper.get(idx, None)
        if std is None:
            std = idx.lower().replace(" ", "_")
        new_index.append(std)
    df.index = new_index
    return df

def check_balance_identity(bs_df: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
    """Return a boolean Series over periods indicating whether Assets == Liabilities + Equity within tolerance."""
    # Yahoo sometimes includes NaNs; align safely
    cols = bs_df.columns
    A = bs_df.loc["assets"].reindex(cols)
    L = bs_df.loc["liab_total"].reindex(cols)
    E = bs_df.loc["equity_total"].reindex(cols)
    diff = (A - (L + E)).abs()
    ok = diff <= tol
    return ok

def check_required_lines(is_df: pd.DataFrame, bs_df: pd.DataFrame) -> Dict[str, bool]:
    keys = {
        "sales": "IS",
        "cogs": "IS",
        "dep": "IS",
        "assets": "BS",
        "liab_total": "BS",
        "equity_total": "BS",
        "ppe_net": "BS"
    }
    results = {}
    for k, st in keys.items():
        df = is_df if st == "IS" else bs_df
        results[k] = k in df.index
    return results
