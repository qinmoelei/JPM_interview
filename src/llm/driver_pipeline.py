from __future__ import annotations

"""End-to-end LLM driver forecasting pipeline used by the CLI wrapper."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.experiments.driver_workflow import (
    collect_targets,
    evaluate_driver_predictions,
    evaluate_state_predictions,
    load_driver_dataset,
)
from src.llm.driver_forecast import run_llm_driver_experiment
from src.llm.ensemble import blend_predictions, tune_driver_weights
from src.llm.recommendations import generate_cfo_recommendation
from src.llm.robustness import run_robustness
from src.model.dynamics_tf import DRIVER_ORDER
from src.model.simulator import AccountingSimulator
from src.utils.io import load_config
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


def _find_latest_analysis(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("analysis_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No analysis_*.json found under {results_dir}")
    return candidates[-1]


def _load_part1_runs(results_dir: Path) -> Dict[str, Path]:
    analysis_path = _find_latest_analysis(results_dir)
    rows = json.loads(analysis_path.read_text())
    runs = {}
    for row in rows:
        exp_id = row.get("exp_id")
        method = row.get("method")
        if not exp_id or not method:
            continue
        exp_dir = results_dir / exp_id
        if exp_dir.exists():
            runs[method] = exp_dir
    if not runs:
        raise FileNotFoundError(f"No experiment directories referenced by {analysis_path}")
    return runs


def _load_predictions(path: Path, tickers: Sequence[str]) -> Dict[str, Dict[int, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    driver_order = payload.get("driver_order", DRIVER_ORDER)
    order_index = {name: i for i, name in enumerate(driver_order)}
    preds_raw = payload.get("preds", {})
    preds: Dict[str, Dict[int, np.ndarray]] = {}
    for tk, bucket in preds_raw.items():
        if tk not in tickers:
            continue
        mapped: Dict[int, np.ndarray] = {}
        for idx, values in bucket.items():
            vec = np.array(values, dtype=float)
            if list(driver_order) != list(DRIVER_ORDER):
                reordered = np.zeros(len(DRIVER_ORDER), dtype=float)
                for j, name in enumerate(DRIVER_ORDER):
                    if name in order_index and order_index[name] < vec.size:
                        reordered[j] = vec[order_index[name]]
                vec = reordered
            mapped[int(idx)] = vec
        if mapped:
            preds[tk] = mapped
    return preds


def _weights_to_array(weights: Mapping[str, float]) -> np.ndarray:
    return np.array([weights.get(name, 0.5) for name in DRIVER_ORDER], dtype=float)


def _maybe_cfo(
    out_dir: Path,
    ticker: str,
    proc_dir: Path,
    preds: Mapping[str, Mapping[int, Sequence[float]]],
    prompt_log_path: Path,
) -> None:
    bucket = preds.get(ticker, {})
    if not bucket:
        LOGGER.warning("No predictions for %s; skip CFO recommendation.", ticker)
        return
    idx = sorted(int(i) for i in bucket.keys())[0]
    if idx <= 0:
        LOGGER.warning("Index %s is too early for CFO recommendation; skip %s.", idx, ticker)
        return
    key = str(idx) if str(idx) in bucket else idx
    pred_driver = np.array(bucket[key], dtype=float)
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
            prompt_log_path=prompt_log_path,
        )
    )
    (out_dir / "cfo_recommendation.json").write_text(json.dumps(rec, indent=2))


def run_llm_driver_pipeline(
    *,
    config_path: str,
    variant: Optional[str],
    frequency: Optional[str],
    tickers: Optional[str],
    max_tickers: int,
    min_rows: int,
    filter_max_abs: Optional[float],
    window: int,
    temperature: float,
    model: Optional[str],
    max_calls: Optional[int],
    robust_runs: int,
    robust_temperature: float,
    cfo_ticker: Optional[str],
    out_dir: str,
    baseline_results_dir: Optional[str],
) -> None:
    cfg = load_config(config_path)
    variant = _resolve_variant(cfg, variant, frequency)
    proc_dir = Path(cfg.paths["proc_dir"]) / variant
    tickers_list = _parse_tickers(tickers) or list(cfg.tickers)
    tickers_list = _filter_tickers(proc_dir, tickers_list, min_rows, filter_max_abs)
    tickers_list = tickers_list[:max_tickers]
    if not tickers_list:
        raise RuntimeError("No usable tickers after filtering.")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    (out_dir_path / "tickers.json").write_text(json.dumps(tickers_list, indent=2))
    (out_dir_path / "run_config.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "tickers": tickers_list,
                "window": window,
                "temperature": temperature,
                "model": model,
                "max_calls": max_calls,
                "baseline_results_dir": baseline_results_dir,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
        )
    )

    prompt_log_path = out_dir_path / "llm_prompt_log.md"
    reasoning_path = out_dir_path / "llm_reasoning.txt"
    llm_payload = run_llm_driver_experiment(
        proc_dir,
        tickers_list,
        window=window,
        temperature=temperature,
        model=model,
        max_calls=max_calls,
        cache_path=out_dir_path / "llm_cache.json",
        prompt_log_path=prompt_log_path,
        reasoning_path=reasoning_path,
    )
    (out_dir_path / "llm_metrics.json").write_text(json.dumps(llm_payload["metrics"], indent=2))
    (out_dir_path / "llm_metadata.json").write_text(json.dumps(llm_payload["metadata"], indent=2))

    frames = load_driver_dataset(proc_dir, tickers_list)
    val_targets = collect_targets(frames, "val")
    test_targets = collect_targets(frames, "test")

    llm_val = {k: {int(i): np.array(v, dtype=float) for i, v in bucket.items()} for k, bucket in llm_payload["preds_val"].items()}
    llm_test = {k: {int(i): np.array(v, dtype=float) for i, v in bucket.items()} for k, bucket in llm_payload["preds_test"].items()}

    if baseline_results_dir is None:
        baseline_results_dir = str(Path("results") / f"driver_experiments_{variant}_top")
    baseline_root = Path(baseline_results_dir)
    baseline_runs = _load_part1_runs(baseline_root)
    (out_dir_path / "baseline_index.json").write_text(
        json.dumps({method: str(path) for method, path in baseline_runs.items()}, indent=2)
    )

    baseline_metrics: Dict[str, object] = {}
    ensemble_metrics: Dict[str, object] = {}
    for method, exp_dir in baseline_runs.items():
        try:
            base_val = _load_predictions(exp_dir / "preds_val.json", tickers_list)
            base_test = _load_predictions(exp_dir / "preds_test.json", tickers_list)
        except FileNotFoundError as exc:
            LOGGER.warning("Missing predictions for %s (%s)", method, exc)
            continue
        baseline_metrics[method] = {
            "driver_val": evaluate_driver_predictions(base_val, val_targets),
            "driver_test": evaluate_driver_predictions(base_test, test_targets),
            "state_val": evaluate_state_predictions(base_val, frames),
            "state_test": evaluate_state_predictions(base_test, frames),
        }

        weight_info = tune_driver_weights(llm_val, base_val, val_targets)
        weight_vec = _weights_to_array(weight_info["weights"])
        ensemble_val = blend_predictions(llm_val, base_val, weight_vec)
        ensemble_test = blend_predictions(llm_test, base_test, weight_vec)
        ensemble_metrics[method] = {
            "weights": weight_info["weights"],
            "weight_grid": weight_info["grid"],
            "driver_val": evaluate_driver_predictions(ensemble_val, val_targets),
            "driver_test": evaluate_driver_predictions(ensemble_test, test_targets),
            "state_val": evaluate_state_predictions(ensemble_val, frames),
            "state_test": evaluate_state_predictions(ensemble_test, frames),
        }

    (out_dir_path / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2))
    (out_dir_path / "ensemble_metrics.json").write_text(json.dumps(ensemble_metrics, indent=2))

    if robust_runs > 0:
        robust = run_robustness(
            proc_dir,
            tickers_list,
            runs=robust_runs,
            temperature=robust_temperature,
            window=window,
            max_calls=min(max_calls or 10, 10),
            cache_dir=out_dir_path / "robust_cache",
            prompt_log_path=prompt_log_path,
        )
        (out_dir_path / "robustness_summary.json").write_text(json.dumps(robust, indent=2))

    if cfo_ticker:
        _maybe_cfo(out_dir_path, cfo_ticker.upper(), proc_dir, llm_payload["preds_test"], prompt_log_path)
