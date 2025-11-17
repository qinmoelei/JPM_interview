from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datahandler.preprocess import DRIVER_COLUMNS
from src.model.trainer import VPForecaster
from src.model.metrics import identity_violation
from src.utils.io import get_default_config_path, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_states_series(proc_dir: Path, ticker: str) -> pd.DataFrame:
    """Load normalized state trajectories for plotting/baselines."""
    path = proc_dir / f"{ticker}_states_raw.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing normalized states for {ticker}: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df


def _rolling_arima(series: pd.Series) -> Tuple[List[float], List[float]]:
    """Rolling ARIMA(1,0,0) one-step forecasts -> (actual, prediction) lists."""
    actual: List[float] = []
    preds: List[float] = []
    values = series.astype(float)
    for end in range(3, len(values)):
        train = values.iloc[:end]
        target = values.iloc[end]
        try:
            model = ARIMA(train, order=(1, 0, 0))
            fit = model.fit()
            pred = fit.forecast()[0]
        except Exception as exc:
            logger.debug("ARIMA failed for %s: %s", series.name, exc)
            continue
        actual.append(float(target))
        preds.append(float(pred))
    return actual, preds


def _metrics(actual: List[float], preds: List[float]) -> Dict[str, float]:
    """Helper to compute MAE/RMSE/MAPE for the ARIMA baseline."""
    if not actual:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "count": 0}
    a = np.array(actual)
    p = np.array(preds)
    mae = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mape = float(np.mean(np.abs((a - p) / (np.abs(a) + 1e-6))))
    return {"mae": mae, "rmse": rmse, "mape": mape, "count": len(actual)}


def _evaluate_neural(dataset_path: Path,
                     weights_path: Path,
                     hidden_dim: int = 16,
                     split: str = "test") -> Dict[str, float]:
    """Load `model.weights.h5`, run inference on one split, and compute masked metrics."""
    data = np.load(dataset_path)
    try:
        states = data[f"states_{split}"].astype("float32")
        covs = data[f"covs_{split}"].astype("float32")
        targets = data[f"targets_{split}"].astype("float32")
        state_shift = data[f"state_shift_{split}"].astype("float32")
        state_scale = data[f"state_scale_{split}"].astype("float32")
        mask = data[f"mask_{split}"].astype("float32")
    except KeyError:
        raise KeyError(f"Split '{split}' not found in dataset {dataset_path}.")
    if states.size == 0 or mask.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "identity_violation": float("nan"), "count": 0}
    _, T, state_dim = states.shape
    cov_dim = covs.shape[-1]
    driver_dim = len(DRIVER_COLUMNS)

    model = VPForecaster(cov_dim=cov_dim, driver_dim=driver_dim, state_dim=state_dim, hidden=hidden_dim)
    _ = model((tf.zeros((1, T, state_dim)), tf.zeros((1, T, cov_dim))))
    model.load_weights(weights_path)
    preds, _ = model((states, covs), training=False)
    preds_np = preds.numpy()
    pred_len = preds_np.shape[1]
    target = targets[:, -pred_len:, :]
    scale_expanded = state_scale[:, None, :]
    shift_expanded = state_shift[:, None, :]
    preds_raw = preds_np * scale_expanded + shift_expanded
    target_raw = target * scale_expanded + shift_expanded
    pred_len = preds_np.shape[1]
    mask_slice = mask[:, -pred_len:]
    mask_expanded = mask_slice[..., None]
    denom = float(np.sum(mask_slice) * preds_raw.shape[-1])
    if denom == 0.0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "identity_violation": float("nan"), "count": 0}
    err = preds_raw - target_raw
    abs_err = np.abs(err) * mask_expanded
    sq_err = np.square(err) * mask_expanded
    mae = float(np.sum(abs_err) / denom)
    rmse = float(np.sqrt(np.sum(sq_err) / denom))
    mape = float(np.sum(abs_err / (np.abs(target_raw) + 1e-6)) / denom)
    mask_bool = mask_slice > 0
    pred_tensor = tf.convert_to_tensor(preds_raw)
    if np.any(mask_bool):
        pred_active = tf.boolean_mask(pred_tensor, mask_bool)
        iden = float(identity_violation(pred_active).numpy())
        active = int(np.sum(mask_slice))
    else:
        iden = float("nan")
        active = 0
    return {"mae": mae, "rmse": rmse, "mape": mape, "identity_violation": iden, "count": active}


def _plot_arima_samples(proc_dir: Path, tickers: List[str], out_path: Path) -> None:
    """Plot ARIMA vs. actual for up to four tickers (saved to PNG)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()
    plotted = 0
    for ticker, ax in zip(tickers, axes):
        try:
            states_df = _load_states_series(proc_dir, ticker)
        except FileNotFoundError:
            ax.axis('off')
            continue
        if len(states_df) < 4:
            ax.axis('off')
            continue
        col = "cash" if "cash" in states_df.columns else states_df.columns[0]
        series = states_df[col]
        actual, preds = _rolling_arima(series)
        if not actual:
            ax.axis('off')
            continue
        idx = range(len(actual))
        ax.plot(idx, actual, label="Actual", color="tab:blue")
        ax.plot(idx, preds, label="ARIMA", color="tab:orange")
        ax.set_title(f"{ticker} – {col}")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Normalized value")
        plotted += 1
    if plotted > 0:
        axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_baselines(
    config_path: Optional[str] = None,
    variant: str = "base",
    dataset_path: Optional[str] = None,
    run_dir: Optional[Path] = None,
    sample_plot_tickers: Optional[List[str]] = None,
    hidden_dim: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare ARIMA vs. neural VPForecaster and optionally save plots."""
    cfg_path = config_path or get_default_config_path()
    cfg = load_config(cfg_path)
    proc_root = Path(cfg.paths.get("proc_dir", "data/processed"))
    if not proc_root.is_absolute():
        proc_root = PROJECT_ROOT / proc_root
    proc_dir = proc_root / variant

    dataset_path = Path(dataset_path) if dataset_path else proc_dir / "training_data.npz"
    if not dataset_path.exists():
        raise FileNotFoundError("Run script/00_preprocess.py for the requested variant before evaluating baselines.")

    if run_dir:
        weights_path = Path(run_dir) / "model.weights.h5"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found in run dir: {weights_path}")
    else:
        weights = sorted(PROJECT_ROOT.glob("results/run_*/model.weights.h5"))
        if not weights:
            raise FileNotFoundError("No trained weights found under results/.")
        weights_path = weights[-1]

    arima_actual: List[float] = []
    arima_pred: List[float] = []
    max_tickers = 50  # limit runtime

    for ticker in cfg.tickers[:max_tickers]:
        try:
            states_df = _load_states_series(proc_dir, ticker)
        except FileNotFoundError:
            continue
        if len(states_df) < 4:
            continue
        col = "cash" if "cash" in states_df.columns else states_df.columns[0]
        actual, preds = _rolling_arima(states_df[col])
        arima_actual.extend(actual)
        arima_pred.extend(preds)

    arima_metrics = _metrics(arima_actual, arima_pred)
    if hidden_dim is None and run_dir:
        learner_path = Path(run_dir) / "learner.json"
        if learner_path.exists():
            try:
                learner_cfg = json.loads(learner_path.read_text())
                hidden_dim = int(learner_cfg.get("model", {}).get("hidden_dim", 16))
            except Exception:
                hidden_dim = 16
        else:
            hidden_dim = 16
    if hidden_dim is None:
        hidden_dim = 16

    neural_metrics = _evaluate_neural(dataset_path, weights_path, hidden_dim=hidden_dim)

    summary = {
        "config": cfg_path,
        "variant": variant,
        "weights": str(weights_path),
        "arima": arima_metrics,
        "neural": neural_metrics,
        "models": {
            "arima": {"order": [1, 0, 0], "library": "statsmodels.ARIMA"},
            "neural": {"driver_dim": len(DRIVER_COLUMNS), "hidden_dim": hidden_dim},
        },
    }

    target_dir = Path(run_dir) if run_dir else Path(cfg.paths.get("results_dir", "results"))
    if not target_dir.is_absolute():
        target_dir = PROJECT_ROOT / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "baseline_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved baseline comparison to %s", out_path)

    if not sample_plot_tickers and run_dir:
        learner_path = Path(run_dir) / "learner.json"
        if learner_path.exists():
            try:
                learner_cfg = json.loads(learner_path.read_text())
                sample_plot_tickers = learner_cfg.get("dataset_summary", {}).get("tickers", [])[:4]
            except Exception:
                sample_plot_tickers = None

    if sample_plot_tickers:
        plot_path = target_dir / "arima_samples.png"
        _plot_arima_samples(proc_dir, sample_plot_tickers, plot_path)

    return summary


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare ARIMA/GARCH baselines against the neural VPForecaster")
    parser.add_argument("--config", default=get_default_config_path(), help="Path to YAML config file.")
    parser.add_argument("--variant", default="base", help="Preprocessing variant under data/processed/.")
    parser.add_argument("--dataset", default=None, help="Optional explicit dataset .npz path.")
    parser.add_argument("--run-dir", default=None, help="Optional run directory containing model weights.")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Override neural hidden dimension for evaluation.")
    args = parser.parse_args(cli_args)
    summary = run_baselines(
        config_path=args.config,
        variant=args.variant,
        dataset_path=args.dataset,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        hidden_dim=args.hidden_dim,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
