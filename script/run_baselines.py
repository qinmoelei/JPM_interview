from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.model.simulator import AccountingSimulator, evaluate_simulation, format_metrics
from src.utils.io import get_default_config_path, load_config
from src.utils.logging import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = get_logger(__name__)


def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    data = np.load(path)
    states = data["states"].astype(float)
    drivers = data["drivers"].astype(float)
    if states.shape[0] < 2 or drivers.shape[0] == 0:
        return None
    return {"states": states, "drivers": drivers}


def _apply_scenario(drivers: np.ndarray, growth_delta: Optional[float]) -> np.ndarray:
    if growth_delta is None:
        return drivers
    adj = drivers.copy()
    adj[:, 0] += growth_delta
    return adj


def simulate_tickers(
    config_path: Optional[str],
    variant: str,
    tickers: List[str],
    growth_delta: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    cfg = load_config(config_path or get_default_config_path())
    proc_root = Path(cfg.paths.get("proc_dir", "data/processed"))
    if not proc_root.is_absolute():
        proc_root = PROJECT_ROOT / proc_root
    proc_dir = proc_root / variant
    simulator = AccountingSimulator()
    metrics: Dict[str, Dict[str, float]] = {}
    for ticker in tickers:
        npz_path = proc_dir / f"{ticker}_simulation.npz"
        data = _load_npz(npz_path)
        if data is None:
            logger.warning("Skipping %s: missing simulation inputs", ticker)
            continue
        drivers = _apply_scenario(data["drivers"], growth_delta)
        preds = simulator.roll(data["states"][0], drivers)
        actual = data["states"][1 : preds.shape[0] + 1]
        metrics[ticker] = format_metrics(evaluate_simulation(actual, preds))
    return metrics


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Quick diagnostics for deterministic simulations.")
    parser.add_argument("--config", default=get_default_config_path(), help="Path to YAML config file.")
    parser.add_argument("--variant", default="simulation", help="Processed variant directory.")
    parser.add_argument("--tickers", nargs="*", help="Tickers to simulate; defaults to the first config entry.")
    parser.add_argument("--growth-delta", type=float, default=None, help="Additive adjustment applied to all growth drivers.")
    args = parser.parse_args(cli_args)

    cfg = load_config(args.config)
    tickers = args.tickers or cfg.tickers[:1]
    metrics = simulate_tickers(args.config, args.variant, tickers, args.growth_delta)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
