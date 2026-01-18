github Repo for JPM interview

# Q1P1   Vélez–Pareja-Style Financial Forecast (CB → IS → BS) with TensorFlow

Deterministic, no-plug, no-circularity three-statement forecaster inspired by Vélez‑Pareja.  
Pipeline: Yahoo Finance → preprocessing to states/drivers → deterministic simulator → driver baselines (perfect / sliding / AR1 / MLP).

**Order of computation**: exogenous drivers → Cash Budget (CB) → Income Statement (IS) → Balance Sheet (BS) while enforcing `Assets = Liabilities + Equity`.

## Project Structure

```
src/
  datahandler/
    data_download.py        # Yahoo download of IS/BS/CF
    data_sanity_check.py    # schema & identity checks
    preprocess.py           # cleaning/normalization to states/drivers
    features.py             # driver ratios
    dataset.py              # rolling windows
  model/
    dynamics_tf.py          # state/driver ordering and forward/inverse transforms
    simulator.py            # deterministic accounting simulator
  utils/
    io.py                   # config helpers
    logging.py              # logger
tests/
  test_02_data_checks.py       # sanity checks on mocked Yahoo data
  test_03_data_download.py     # download orchestration & defaults
  test_04_dataset.py           # sequence builder windows
  test_05_features.py          # driver/ratio calculations
  test_driver_metrics.py       # driver metric checker
configs/
  config.yaml                  # tickers, date ranges, training params
reports/
  report.tex                   # LaTeX report
script/
  00_download.py               # download raw statements
  01_preprocess.py             # build simulation-ready states/drivers
  02_train.py                  # deterministic simulator eval
  03_eval.py                   # optional custom eval
  04_driver_pipeline.py        # driver baselines (perfect/sliding/AR1/MLP)
```

- `configs/` holds download/preprocess/simulation parameters.
- `src/datahandler/` aligns Yahoo IS/BS/CF, computes robust line items and driver ratios.
- `src/model/` provides the deterministic accounting simulator and state/driver mappings.
- `script/` has the CLI workflow: download → preprocess → simulate → driver baselines.
- `tests/` contain lightweight unit tests for utilities.

## Workflow

1. Download – `python script/00_download.py --config configs/config.yaml` to fetch Yahoo annual/quarterly statements.
2. Preprocess – `python script/01_preprocess.py --config configs/config.yaml --variant <year|quarter>` to produce `data/processed/<variant>/*_states.csv`, `*_drivers.csv`, `*_simulation.npz`, plus `simulation_summary.json`.
3. Simulate (deterministic) – `PYTHONPATH=. python script/02_train.py --config configs/config.yaml --variant <year|quarter> [--max-tickers N]` to roll the simulator and write `results/simulation_<timestamp>/simulation_metrics.json`.
4. Driver baselines – `python script/04_driver_pipeline.py --config configs/config.yaml --variant <year|quarter> --results-subdir driver_experiments_<...>` to run perfect/sliding/AR1/MLP driver forecasts; outputs `analysis_*.json` and per-ticker metrics.
5. Evaluate (optional) – `python script/03_eval.py --config configs/config.yaml` for custom backtests.
6. 
7. conda run -n jpmc python script/06_pdf_statement_pipeline.py --out-dir results/part2_llm_run_e2i

Tip: set `PYTHONPATH=.` when running scripts directly.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U -r requirements.txt
python script/00_download.py --config configs/config.yaml
python script/01_preprocess.py --config configs/config.yaml --variant year
PYTHONPATH=. python script/02_train.py --config configs/config.yaml --variant year
python script/04_driver_pipeline.py --config configs/config.yaml --variant year --results-subdir driver_experiments_year
```

## Key Functions

- `data_download.download_universe` – stream-download Yahoo statements with logging/retries.
- `preprocess.build_simulation_frames` – load raw IS/BS/CF, pick robust line items, derive states + driver ratios, align years.
- `data/processed/<variant>/*_simulation.npz` – stores `states` (T×15) and `drivers` ((T-1)×13) for simulator use; `simulation_summary.json` tracks availability.
- `model.simulator.AccountingSimulator` – deterministic layer enforcing working-capital/financing logic and accounting identity.
- `script/04_driver_pipeline.py` – driver baselines (perfect/sliding/AR1/MLP), outputs `analysis_*.json` and per-ticker metrics.
- `get_default_config_path` – resolves `$JPM_CONFIG_PATH` or default config for every CLI.

## Results folder guide

- `results/simulation_<timestamp>/simulation_metrics.json`: deterministic simulation MAE/identity metrics per ticker.
- `results/driver_experiments_<variant>_<tag>/analysis_*.json`: summary of driver/state metrics for each method (perfect, sliding, AR1, MLP).
- `results/driver_experiments_<variant>_<tag>/<method>_<timestamp>_*/`: per-method run directory containing `learner.json` (metrics/config) and `per_ticker.json` (per-ticker driver/state errors).
