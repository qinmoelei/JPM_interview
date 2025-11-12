import pandas as pd
from src.datahandler.data_sanity_check import standardize_columns, check_balance_identity

def test_balance_identity_pass():
    data = {
        "Total Assets": [100.0],
        "Total Liab": [60.0],
        "Total Stockholder Equity": [40.0]
    }
    df = pd.DataFrame(data, index=[pd.Timestamp("2024-12-31")]).T
    df = standardize_columns(df, statement="BS")
    ok = check_balance_identity(df)
    assert ok.iloc[0] == True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
