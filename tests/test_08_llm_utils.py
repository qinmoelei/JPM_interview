import numpy as np

from src.llm.json_utils import safe_json_loads, coerce_float
from src.llm.ratios import compute_ratios


def test_safe_json_loads_with_code_fence():
    text = "```json\n{\"a\": 1, \"b\": 2}\n```"
    data = safe_json_loads(text)
    assert data["a"] == 1
    assert data["b"] == 2


def test_coerce_float_percent():
    assert coerce_float("12.5%") == 0.125


def test_compute_ratios_basic():
    items = {
        "revenue": 100.0,
        "cogs": 40.0,
        "sga": 10.0,
        "net_income": 20.0,
        "interest_expense": 5.0,
        "tax_expense": 3.0,
        "operating_income": 50.0,
        "depreciation": 4.0,
        "current_assets": 60.0,
        "inventory": 10.0,
        "current_liabilities": 30.0,
        "total_assets": 120.0,
        "total_equity": 40.0,
        "total_debt": 50.0,
    }
    ratios = compute_ratios(items)
    assert np.isclose(ratios["cost_to_income"], 0.5)
    assert np.isclose(ratios["quick_ratio"], (60.0 - 10.0) / 30.0)
    assert np.isclose(ratios["debt_to_equity"], 50.0 / 40.0)
