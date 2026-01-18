from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from src.experiments.driver_workflow import collect_targets, evaluate_driver_predictions, evaluate_state_predictions, load_driver_dataset
from src.llm.apiyi import load_apiyi_config
from src.llm.driver_forecast import LLMRunConfig, predict_llm_drivers


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _sdk_version() -> Optional[str]:
    try:
        import openai
    except Exception:
        return None
    return getattr(openai, "__version__", None)


def run_robustness(
    proc_dir: Path,
    tickers: Sequence[str],
    *,
    runs: int = 5,
    model: Optional[str] = None,
    temperature: float = 0.0,
    window: int = 3,
    max_calls: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, object]:
    frames = load_driver_dataset(proc_dir, tickers)
    targets = collect_targets(frames, "test")
    driver_mae = []
    driver_rel1 = []
    state_mae = []
    state_rel1 = []
    for r in range(runs):
        cache_path = None
        if cache_dir is not None:
            cache_path = cache_dir / f"robust_run_{r}.json"
        cfg = LLMRunConfig(
            model=model,
            temperature=temperature,
            window=window,
            max_calls=max_calls,
            cache_path=cache_path,
        )
        preds = predict_llm_drivers(frames, "test", cfg)
        d_metrics = evaluate_driver_predictions(preds, targets)
        s_metrics = evaluate_state_predictions(preds, frames)
        driver_mae.append(d_metrics.get("mae", float("nan")))
        driver_rel1.append(d_metrics.get("rel_l1", float("nan")))
        state_mae.append(s_metrics.get("mae", float("nan")))
        state_rel1.append(s_metrics.get("rel_l1", float("nan")))
    cfg = load_apiyi_config()
    return {
        "driver_mae": _mean_std(driver_mae),
        "driver_rel_l1": _mean_std(driver_rel1),
        "state_mae": _mean_std(state_mae),
        "state_rel_l1": _mean_std(state_rel1),
        "runs": runs,
        "model": model or cfg.model,
        "base_url": cfg.base_url,
        "sdk_version": _sdk_version(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
