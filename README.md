github Repo for JPM interview

# Q1P1   Vélez–Pareja-Style Financial Forecast (CB → IS → BS) with TensorFlow

This repository implements a **deterministic, no-plug, no-circularity** three-statement forecaster in the spirit of Vélez‑Pareja.
It uses **Yahoo Finance** (via `yfinance`) to obtain **income statements**, **balance sheets**, and **cash flow statements**,
and trains **driver networks** in TensorFlow to forecast exogenous drivers. The **accounting layer** is implemented as a
custom Keras layer that enforces the **cash-budget rules** and **accounting identities** by construction.

> **Order of computation**: exogenous drivers → **Cash Budget (CB)** (decide ST/LT financing, short-term investment, lock cash target) → **Income Statement (IS)** (interests on beginning balances, taxes) → **Balance Sheet (BS)** (update states and verify `Assets = Liabilities + Equity`).

## Project structure

```
src/
  datahandler/
    data_download.py        # yfinance download of IS, BS, CF + macros (optional)
    data_sanity_check.py    # schema checks + accounting identity checks
    features.py             # compute derived ratios/drivers
    dataset.py              # sequence builder for (x_t, y_t) windows
  model/
    cash_budget_layer.py    # deterministic Vélez–Pareja mapping f(x_{t-1}, u_t) → x_t
    losses.py               # identity penalties and data-fit losses
    metrics.py              # scale-aware metrics and identity violations
    trainer.py              # tf training loop and evaluation
  utils/
    io.py                   # config loading, paths
    logging.py              # basic logger
tests/
  test_cash_budget_layer.py # synthetic tests: identity holds, no circularity
  test_data_checks.py       # sanity checks on mocked Yahoo data
configs/
  config.yaml               # tickers, date ranges, training params
reports/
  report.tex                # LaTeX report (matches this codebase)
run_download.py             # CLI to download and cache raw financials
run_train.py                # CLI to train models
run_eval.py                 # CLI to evaluate and backtest
```

## Quick start

> **Note:** The environment here is offline; run the commands below *locally*.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U yfinance pandas numpy pydantic pyyaml tensorflow==2.* matplotlib
python run_download.py --config configs/config.yaml
python run_train.py --config configs/config.yaml
python run_eval.py  --config configs/config.yaml
```

## References

- Vélez‑Pareja, *Forecasting Financial Statements with No Plugs and No Circularity* (SSRN 1031735).  
- Vélez‑Pareja, *Constructing Consistent Financial Planning Models for Valuation* (SSRN 1455304).

