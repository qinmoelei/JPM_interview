from __future__ import annotations
import pandas as pd
import pytest

from src.datahandler.features import compute_drivers


def test_compute_drivers_returns_expected_ratios():
    period = pd.Timestamp("2024-12-31")
    is_df = pd.DataFrame(
        {
            period: [100.0, 60.0, 30.0, 10.0],
        },
        index=["sales", "cogs", "opex", "dep"],
    )
    bs_df = pd.DataFrame(
        {
            period: [15.0, 10.0, 5.0, 8.0],
        },
        index=["cash", "ar", "ap", "inv_stock"],
    )
    cf_df = pd.DataFrame({period: [-20.0]}, index=["capex"])

    drivers = compute_drivers(is_df, bs_df, cf_df)

    assert pytest.approx(drivers.loc[period, "ebitda_margin"]) == 0.1
    assert pytest.approx(drivers.loc[period, "cogs_ratio"]) == 0.6
    assert pytest.approx(drivers.loc[period, "opex_ratio"]) == 0.3
    assert pytest.approx(drivers.loc[period, "capex_ratio"]) == 0.2
    assert pytest.approx(drivers.loc[period, "DSO"]) == 36.5
    assert pytest.approx(drivers.loc[period, "DPO"]) == 30.4166666, "DPO should divide AP by COGS"
    assert pytest.approx(drivers.loc[period, "DIO"]) == 48.6666666
    assert pytest.approx(drivers.loc[period, "cash_target_ratio"]) == 0.15


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
