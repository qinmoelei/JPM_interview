from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


def _safe_div(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom in (None, 0.0):
        return None
    return float(numer) / float(denom)


def cost_to_income(revenue: float | None, cogs: float | None, sga: float | None) -> float | None:
    if revenue is None:
        return None
    cost = 0.0
    if cogs is not None:
        cost += cogs
    if sga is not None:
        cost += sga
    return _safe_div(cost, revenue)


def quick_ratio(current_assets: float | None, inventory: float | None, current_liabilities: float | None) -> float | None:
    if current_assets is None or current_liabilities is None:
        return None
    inv = inventory or 0.0
    return _safe_div(current_assets - inv, current_liabilities)


def debt_to_equity(total_debt: float | None, total_equity: float | None) -> float | None:
    return _safe_div(total_debt, total_equity)


def debt_to_assets(total_debt: float | None, total_assets: float | None) -> float | None:
    return _safe_div(total_debt, total_assets)


def debt_to_capital(total_debt: float | None, total_equity: float | None) -> float | None:
    if total_debt is None or total_equity is None:
        return None
    return _safe_div(total_debt, total_debt + total_equity)


def debt_to_ebitda(total_debt: float | None, ebitda: float | None) -> float | None:
    return _safe_div(total_debt, ebitda)


def interest_coverage(ebit: float | None, interest_expense: float | None) -> float | None:
    return _safe_div(ebit, interest_expense)


def ebitda_from_items(
    operating_income: float | None,
    dep: float | None,
    amort: float | None = None,
) -> float | None:
    if operating_income is None and dep is None and amort is None:
        return None
    base = operating_income or 0.0
    return base + (dep or 0.0) + (amort or 0.0)


def ebit_from_items(
    operating_income: float | None,
    interest_expense: float | None,
    tax_expense: float | None,
    net_income: float | None,
) -> float | None:
    if operating_income is not None:
        return operating_income
    if net_income is None:
        return None
    return net_income + (interest_expense or 0.0) + (tax_expense or 0.0)


def compute_ratios(items: Mapping[str, float | None]) -> Dict[str, float | None]:
    revenue = items.get("revenue")
    cogs = items.get("cogs")
    sga = items.get("sga")
    net_income = items.get("net_income")
    interest_expense = items.get("interest_expense")
    tax_expense = items.get("tax_expense")
    operating_income = items.get("operating_income")
    dep = items.get("depreciation")
    amort = items.get("amortization")
    current_assets = items.get("current_assets")
    current_liabilities = items.get("current_liabilities")
    inventory = items.get("inventory")
    total_debt = items.get("total_debt")
    total_assets = items.get("total_assets")
    total_equity = items.get("total_equity")

    ebit = ebit_from_items(operating_income, interest_expense, tax_expense, net_income)
    ebitda = ebitda_from_items(operating_income, dep, amort)

    return {
        "net_income": net_income,
        "cost_to_income": cost_to_income(revenue, cogs, sga),
        "quick_ratio": quick_ratio(current_assets, inventory, current_liabilities),
        "debt_to_equity": debt_to_equity(total_debt, total_equity),
        "debt_to_assets": debt_to_assets(total_debt, total_assets),
        "debt_to_capital": debt_to_capital(total_debt, total_equity),
        "debt_to_ebitda": debt_to_ebitda(total_debt, ebitda),
        "interest_coverage": interest_coverage(ebit, interest_expense),
        "ebit": ebit,
        "ebitda": ebitda,
    }
