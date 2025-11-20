from __future__ import annotations

import argparse
import json
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from src.model.simulator import AccountingSimulator, evaluate_simulation, format_metrics
from src.utils.io import ensure_dir, get_default_config_path, load_config
from src.utils.logging import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = get_logger(__name__)


def _load_simulation_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    data = np.load(path)
    states = data["states"].astype(float)
    drivers = data["drivers"].astype(float)
    if states.shape[0] < 2 or drivers.shape[0] == 0:
        return None
    return {"states": states, "drivers": drivers}


def _simulate_ticker(ticker: str, data: Dict[str, np.ndarray], simulator: AccountingSimulator) -> Dict[str, float]:
    states = data["states"]
    drivers = data["drivers"]
    preds = simulator.roll(states[0], drivers)
    actual = states[1 : preds.shape[0] + 1]
    metrics = evaluate_simulation(actual, preds)
    return format_metrics(metrics)


def run_simulation_pipeline(
    config_path: Optional[str],
    variant: str,
    max_tickers: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    cfg = load_config(config_path or get_default_config_path())
    proc_root = Path(cfg.paths.get("proc_dir", "data/processed"))
    if not proc_root.is_absolute():
        proc_root = PROJECT_ROOT / proc_root
    proc_dir = proc_root / variant
    if not proc_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {proc_dir}")

    simulator = AccountingSimulator()
    results: Dict[str, Dict[str, float]] = {}
    tickers = cfg.tickers[: max_tickers or len(cfg.tickers)]
    for ticker in tickers:
        npz_path = proc_dir / f"{ticker}_simulation.npz"
        data = _load_simulation_npz(npz_path)
        if data is None:
            logger.warning("Skipping %s: no simulation inputs", ticker)
            continue
        results[ticker] = _simulate_ticker(ticker, data, simulator)
        logger.info(
            "%s – MAE %.2f | Balance gap %.2f",
            ticker,
            results[ticker]["mae"],
            results[ticker]["max_balance_gap"],
        )
    return results


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run deterministic accounting simulations for processed tickers.")
    parser.add_argument("--config", default=get_default_config_path(), help="Path to YAML config file.")
    parser.add_argument("--variant", default="simulation", help="Name of processed variant directory.")
    parser.add_argument("--max-tickers", type=int, default=None, help="Limit the number of tickers to simulate.")
    args = parser.parse_args(cli_args)

    results_dir = Path(load_config(args.config).paths.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    ensure_dir(results_dir)
    run_name = f"simulation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = results_dir / run_name
    ensure_dir(run_dir)

    results = run_simulation_pipeline(args.config, args.variant, args.max_tickers)
    (run_dir / "simulation_metrics.json").write_text(json.dumps(results, indent=2))
    logger.info("Stored simulation metrics for %d tickers in %s", len(results), run_dir)


if __name__ == "__main__":
    main()
