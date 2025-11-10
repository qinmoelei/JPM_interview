from __future__ import annotations
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from src.datahandler import data_download as dd
from src.utils.io import Config, get_default_config_path


def _mock_bundle() -> dict[str, pd.DataFrame]:
    return {
        "IS": pd.DataFrame({"2024-12-31": [1.0]}, index=["sales"]),
        "BS": pd.DataFrame({"2024-12-31": [2.0]}, index=["cash"]),
        "CF": pd.DataFrame({"2024-12-31": [3.0]}, index=["capex"]),
    }


def test_download_universe_persists_csv(tmp_path: Path) -> None:
    bundle = _mock_bundle()
    with patch.object(dd, "fetch_financials_for_ticker", return_value=bundle):
        dd.download_universe(["AAPL"], out_dir=str(tmp_path), frequency="annual")

    for statement in bundle:
        expected = tmp_path / f"AAPL_{statement}_annual.csv"
        assert expected.exists()


def test_run_download_pipeline_uses_default_path(tmp_path: Path) -> None:
    cfg = Config(
        tickers=["AAPL"],
        macro_tickers=[],
        frequency="annual",
        start_year=2020,
        end_year=2024,
        paths={"raw_dir": str(tmp_path / "raw")},
        training={"epochs": 1},
    )

    used_args = {}

    def fake_load(path: str) -> Config:
        used_args["config_path"] = path
        return cfg

    def fake_download(tickers, out_dir, frequency):
        used_args["download"] = (tickers, out_dir, frequency)

    with patch.object(dd, "load_config", side_effect=fake_load), patch.object(dd, "download_universe", side_effect=fake_download):
        used_path = dd.run_download_pipeline()

    default_path = get_default_config_path()
    assert used_path == default_path
    assert used_args["config_path"] == default_path
    assert used_args["download"] == (cfg.tickers, cfg.paths["raw_dir"], cfg.frequency)
