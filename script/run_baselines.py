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
from arch import arch_model
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
    path = proc_dir / f"{ticker}_states_normalized.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing normalized states for {ticker}: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df


def _rolling_arima(series: pd.Series) -> Tuple[List[float], List[float]]:
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


def _rolling_garch(series: pd.Series) -> Tuple[List[float], List[float]]:
    actual: List[float] = []
    preds: List[float] = []
    values = series.astype(float)
    scale = 1000.0
    for end in range(10, len(values)):
        train = values.iloc[:end]
        target = values.iloc[end]
        try:
            am = arch_model(train * scale, mean="AR", lags=1, vol="Garch", p=1, q=1, dist="normal")
            res = am.fit(disp="off")
            forecast = res.forecast(horizon=1)
            pred = forecast.mean.iloc[-1, 0] / scale
        except Exception as exc:
            logger.debug("GARCH failed for %s: %s", series.name, exc)
            continue
        actual.append(float(target))
        preds.append(float(pred))
    return actual, preds


def _metrics(actual: List[float], preds: List[float]) -> Dict[str, float]:
    if not actual:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "count": 0}
    a = np.array(actual)
    p = np.array(preds)
    mae = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mape = float(np.mean(np.abs((a - p) / (np.abs(a) + 1e-6))))
    return {"mae": mae, "rmse": rmse, "mape": mape, "count": len(actual)}


def _evaluate_neural(dataset_path: Path, weights_path: Path) -> Dict[str, float]:
    data = np.load(dataset_path)
    states = data["states"].astype("float32")
    covs = data["covs"].astype("float32")
    _, T, state_dim = states.shape
    cov_dim = covs.shape[-1]
    driver_dim = len(DRIVER_COLUMNS)

    model = VPForecaster(cov_dim=cov_dim, driver_dim=driver_dim, state_dim=state_dim)
    _ = model((tf.zeros((1, T, state_dim)), tf.zeros((1, T, cov_dim))))
    model.load_weights(weights_path)
    preds, _ = model((states, covs), training=False)
    preds_np = preds.numpy()
    target = states[:, -preds.shape[1]:, :]
    err = preds_np - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mape = float(np.mean(np.abs(err) / (np.abs(target) + 1e-6)))
    iden = float(identity_violation(tf.convert_to_tensor(preds_np)).numpy())
    return {"mae": mae, "rmse": rmse, "mape": mape, "identity_violation": iden, "count": int(preds_np.size)}


def _plot_arima_samples(proc_dir: Path, tickers: List[str], out_path: Path) -> None:
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
) -> Dict[str, Dict[str, float]]:
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
    garch_actual: List[float] = []
    garch_pred: List[float] = []
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
        ga, gp = _rolling_garch(states_df[col])
        garch_actual.extend(ga)
        garch_pred.extend(gp)

    arima_metrics = _metrics(arima_actual, arima_pred)
    garch_metrics = _metrics(garch_actual, garch_pred)
    neural_metrics = _evaluate_neural(dataset_path, weights_path)

    summary = {
        "config": cfg_path,
        "variant": variant,
        "weights": str(weights_path),
        "arima": arima_metrics,
        "garch": garch_metrics,
        "neural": neural_metrics,
    }

    target_dir = Path(run_dir) if run_dir else Path(cfg.paths.get("results_dir", "results"))
    if not target_dir.is_absolute():
        target_dir = PROJECT_ROOT / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "baseline_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved baseline comparison to %s", out_path)

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
    args = parser.parse_args(cli_args)
    summary = run_baselines(
        config_path=args.config,
        variant=args.variant,
        dataset_path=args.dataset,
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
