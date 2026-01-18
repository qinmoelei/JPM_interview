from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np

from src.experiments.driver_workflow import evaluate_driver_predictions


def blend_predictions(
    preds_a: Mapping[str, Mapping[int, np.ndarray]],
    preds_b: Mapping[str, Mapping[int, np.ndarray]],
    weight: float,
) -> Dict[str, Dict[int, np.ndarray]]:
    blended: Dict[str, Dict[int, np.ndarray]] = {}
    w = float(weight)
    for ticker, bucket in preds_a.items():
        other = preds_b.get(ticker, {})
        merged: Dict[int, np.ndarray] = {}
        for idx, pred in bucket.items():
            if idx not in other:
                continue
            merged[idx] = w * pred + (1.0 - w) * other[idx]
        if merged:
            blended[ticker] = merged
    return blended


def tune_weight(
    preds_a: Mapping[str, Mapping[int, np.ndarray]],
    preds_b: Mapping[str, Mapping[int, np.ndarray]],
    targets: Mapping[str, Mapping[int, np.ndarray]],
    weights: Iterable[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> Dict[str, float]:
    best_weight = None
    best_mse = float("inf")
    results: Dict[str, float] = {}
    for w in weights:
        blended = blend_predictions(preds_a, preds_b, w)
        metrics = evaluate_driver_predictions(blended, targets)
        mse = metrics.get("mse", float("inf"))
        results[f"{w:.2f}"] = mse
        if mse < best_mse:
            best_mse = mse
            best_weight = w
    return {"best_weight": float(best_weight or 0.0), "best_mse": float(best_mse), "grid": results}
