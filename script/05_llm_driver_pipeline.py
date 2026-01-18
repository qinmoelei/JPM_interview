from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.driver_workflow import (
    build_driver_batch,
    collect_targets,
    evaluate_driver_predictions,
    evaluate_state_predictions,
    load_driver_dataset,
)
from src.llm.driver_forecast import run_llm_driver_experiment, build_mlp_baseline
from src.llm.ensemble import blend_predictions, tune_weight
from src.llm.recommendations import generate_cfo_recommendation
from src.llm.robustness import run_robustness
from src.model.simulator import AccountingSimulator
from src.utils.io import get_default_config_path, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _parse_tickers(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _resolve_variant(cfg, variant: Optional[str], frequency: Optional[str]) -> str:
    if variant:
        return variant
    freq = frequency or cfg.frequency
    return "year" if freq == "annual" else "quarter"


def _filter_tickers(proc_dir: Path, tickers: Sequence[str], min_rows: int, max_abs: Optional[float]) -> List[str]:
    kept = []
    for tk in tickers:
        path = proc_dir / f"{tk}_drivers.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        if df.shape[0] < min_rows:
            continue
        if max_abs is not None and df.abs().to_numpy().max() > max_abs:
            continue
        kept.append(tk)
    return kept


def _maybe_cfo(
    out_dir: Path,
    ticker: str,
    proc_dir: Path,
    preds: dict,
) -> None:
    bucket = preds.get(ticker, {})
    if not bucket:
        LOGGER.warning("No predictions for %s; skip CFO recommendation.", ticker)
        return
    idx = sorted(int(i) for i in bucket.keys())[0]
    pred_driver = np.array(bucket[str(idx)], dtype=float)
    states = pd.read_csv(proc_dir / f"{ticker}_states.csv", index_col=0)
    drivers = pd.read_csv(proc_dir / f"{ticker}_drivers.csv", index_col=0)
    if idx >= len(states):
        LOGGER.warning("Index %s out of bounds for %s states.", idx, ticker)
        return
    prev_state = states.iloc[idx].to_numpy(dtype=float)
    last_driver = drivers.iloc[idx - 1].to_numpy(dtype=float)
    next_state = AccountingSimulator()._step(prev_state, pred_driver)
    rec = asyncio.run(
        generate_cfo_recommendation(
            ticker,
            prev_state,
            next_state,
            last_driver,
            pred_driver,
            temperature=0.2,
        )
    )
    (out_dir / "cfo_recommendation.json").write_text(json.dumps(rec, indent=2))


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="LLM driver forecast + ensemble + robustness pipeline.")
    ap.add_argument("--config", default=get_default_config_path(), help="Config path.")
    ap.add_argument("--variant", default=None, help="Processed subdir (year/quarter).")
    ap.add_argument("--frequency", choices=["annual", "quarterly"], default=None, help="Override frequency.")
    ap.add_argument("--tickers", default=None, help="Comma-separated tickers (override config).")
    ap.add_argument("--max-tickers", type=int, default=6, help="Max tickers to use.")
    ap.add_argument("--min-rows", type=int, default=4, help="Min rows for driver history.")
    ap.add_argument("--filter-max-abs", type=float, default=1000.0, help="Skip tickers with extreme drivers.")
    ap.add_argument("--window", type=int, default=3, help="History window for LLM.")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    ap.add_argument("--model", default=None, help="Override LLM model.")
    ap.add_argument("--max-calls", type=int, default=20, help="Limit total LLM calls.")
    ap.add_argument("--robust-runs", type=int, default=0, help="Run robustness repeats.")
    ap.add_argument("--robust-temperature", type=float, default=0.1, help="Temperature for robustness runs.")
    ap.add_argument("--cfo-ticker", default=None, help="Ticker for CFO recommendation.")
    ap.add_argument("--out-dir", default="results/part2_llm_run", help="Output directory.")
    args = ap.parse_args(cli_args)

    cfg = load_config(args.config)
    variant = _resolve_variant(cfg, args.variant, args.frequency)
    proc_dir = Path(cfg.paths["proc_dir"]) / variant
    tickers = _parse_tickers(args.tickers) or list(cfg.tickers)
    tickers = _filter_tickers(proc_dir, tickers, args.min_rows, args.filter_max_abs)
    tickers = tickers[: args.max_tickers]
    if not tickers:
        raise RuntimeError("No usable tickers after filtering.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tickers.json").write_text(json.dumps(tickers, indent=2))
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "tickers": tickers,
                "window": args.window,
                "temperature": args.temperature,
                "model": args.model,
                "max_calls": args.max_calls,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
        )
    )

    llm_payload = run_llm_driver_experiment(
        proc_dir,
        tickers,
        window=args.window,
        temperature=args.temperature,
        model=args.model,
        max_calls=args.max_calls,
        cache_path=out_dir / "llm_cache.json",
    )
    (out_dir / "llm_metrics.json").write_text(json.dumps(llm_payload["metrics"], indent=2))
    (out_dir / "llm_metadata.json").write_text(json.dumps(llm_payload["metadata"], indent=2))

    frames = load_driver_dataset(proc_dir, tickers)
    val_targets = collect_targets(frames, "val")
    test_targets = collect_targets(frames, "test")

    baseline_metrics = {}
    ensemble_metrics = {}
    try:
        mlp_val, mlp_test = build_mlp_baseline(frames)
        baseline_metrics = {
            "driver_val": evaluate_driver_predictions(mlp_val, val_targets),
            "driver_test": evaluate_driver_predictions(mlp_test, test_targets),
            "state_val": evaluate_state_predictions(mlp_val, frames),
            "state_test": evaluate_state_predictions(mlp_test, frames),
        }
        (out_dir / "mlp_baseline.json").write_text(json.dumps(baseline_metrics, indent=2))

        import numpy as np

        llm_val = {k: {int(i): np.array(v) for i, v in bucket.items()} for k, bucket in llm_payload["preds_val"].items()}
        llm_test = {k: {int(i): np.array(v) for i, v in bucket.items()} for k, bucket in llm_payload["preds_test"].items()}
        weight_info = tune_weight(llm_val, mlp_val, val_targets)
        best_w = weight_info["best_weight"]
        ensemble_val = blend_predictions(llm_val, mlp_val, best_w)
        ensemble_test = blend_predictions(llm_test, mlp_test, best_w)
        ensemble_metrics = {
            "driver_val": evaluate_driver_predictions(ensemble_val, val_targets),
            "driver_test": evaluate_driver_predictions(ensemble_test, test_targets),
            "state_val": evaluate_state_predictions(ensemble_val, frames),
            "state_test": evaluate_state_predictions(ensemble_test, frames),
            "weight_tuning": weight_info,
        }
        (out_dir / "ensemble_metrics.json").write_text(json.dumps(ensemble_metrics, indent=2))
    except ValueError as exc:
        LOGGER.warning("Skipping MLP/ensemble: %s", exc)

    if args.robust_runs > 0:
        robust = run_robustness(
            proc_dir,
            tickers,
            runs=args.robust_runs,
            temperature=args.robust_temperature,
            window=args.window,
            max_calls=min(args.max_calls, 10),
            cache_dir=out_dir / "robust_cache",
        )
        (out_dir / "robustness_summary.json").write_text(json.dumps(robust, indent=2))

    if args.cfo_ticker:
        _maybe_cfo(out_dir, args.cfo_ticker.upper(), proc_dir, llm_payload["preds_test"])


if __name__ == "__main__":
    main()
