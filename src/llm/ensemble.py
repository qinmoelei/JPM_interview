from __future__ import annotations

"""Driver-level ensemble utilities."""

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from src.experiments.driver_workflow import evaluate_driver_predictions
from src.model.dynamics_tf import DRIVER_ORDER


def blend_predictions(
    preds_a: Mapping[str, Mapping[int, np.ndarray]],
    preds_b: Mapping[str, Mapping[int, np.ndarray]],
    weight: float | Sequence[float] | np.ndarray,
) -> Dict[str, Dict[int, np.ndarray]]:
    blended: Dict[str, Dict[int, np.ndarray]] = {}
    w = np.asarray(weight, dtype=float)
    if w.ndim == 0:
        w = float(w)
    pred_size = None
    for bucket in preds_a.values():
        for pred in bucket.values():
            pred_size = pred.size
            break
        if pred_size is not None:
            break
    if isinstance(w, np.ndarray) and w.ndim == 1 and pred_size is not None:
        if w.size not in (1, pred_size):
            raise ValueError(f"Weight vector length {w.size} does not match predictions size {pred_size}.")
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
    # Grid-search weight on validation targets to minimize driver MSE.
    for w in weights:
        blended = blend_predictions(preds_a, preds_b, w)
        metrics = evaluate_driver_predictions(blended, targets)
        mse = metrics.get("mse", float("inf"))
        results[f"{w:.2f}"] = mse
        if mse < best_mse:
            best_mse = mse
            best_weight = w
    return {"best_weight": float(best_weight or 0.0), "best_mse": float(best_mse), "grid": results}


def tune_driver_weights(
    preds_a: Mapping[str, Mapping[int, np.ndarray]],
    preds_b: Mapping[str, Mapping[int, np.ndarray]],
    targets: Mapping[str, Mapping[int, np.ndarray]],
    weights: Iterable[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> Dict[str, object]:
    # Grid-search per-driver weights using validation targets.
    driver_values = {name: {"a": [], "b": [], "t": []} for name in DRIVER_ORDER}
    for ticker, bucket in preds_a.items():
        other = preds_b.get(ticker, {})
        tgt_bucket = targets.get(ticker, {})
        for idx, pred_a in bucket.items():
            pred_b = other.get(idx)
            tgt = tgt_bucket.get(idx)
            if pred_b is None or tgt is None:
                continue
            for j, name in enumerate(DRIVER_ORDER):
                driver_values[name]["a"].append(float(pred_a[j]))
                driver_values[name]["b"].append(float(pred_b[j]))
                driver_values[name]["t"].append(float(tgt[j]))

    best_weights: Dict[str, float] = {}
    grid: Dict[str, Dict[str, float]] = {}
    for name in DRIVER_ORDER:
        a = np.array(driver_values[name]["a"], dtype=float)
        b = np.array(driver_values[name]["b"], dtype=float)
        t = np.array(driver_values[name]["t"], dtype=float)
        if a.size == 0:
            best_weights[name] = 0.5
            grid[name] = {}
            continue
        best_w = None
        best_mse = float("inf")
        grid[name] = {}
        for w in weights:
            blended = w * a + (1.0 - w) * b
            mse = float(np.mean((blended - t) ** 2))
            grid[name][f"{w:.2f}"] = mse
            if mse < best_mse:
                best_mse = mse
                best_w = w
        best_weights[name] = float(best_w if best_w is not None else 0.5)
    return {"weights": best_weights, "grid": grid}
