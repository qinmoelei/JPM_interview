from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datahandler.preprocess import DRIVER_COLUMNS
from src.model.metrics import identity_violation
from src.model.trainer import VPForecaster, training_step
from src.utils.io import ensure_dir, get_default_config_path, load_config
from src.utils.logging import get_logger
from script.run_baselines import run_baselines

logger = get_logger(__name__)
SPLIT_NAMES = ("train", "val", "test")


def _prepare_paths(cfg, variant: str) -> tuple[Path, Path, Path, Path]:
    proc_root = Path(cfg.paths.get("proc_dir", "data/processed"))
    if not proc_root.is_absolute():
        proc_root = PROJECT_ROOT / proc_root
    proc_dir = proc_root / variant
    dataset_path = proc_dir / "training_data.npz"
    summary_path = proc_dir / "training_summary.json"

    results_root = Path(cfg.paths.get("results_dir", "results"))
    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root
    ensure_dir(str(results_root))
    return proc_dir, dataset_path, summary_path, results_root


def _load_split_batches(np_data: np.lib.npyio.NpzFile) -> Dict[str, Dict[str, np.ndarray]]:
    splits = {}
    for split in SPLIT_NAMES:
        splits[split] = {
            "states": np_data[f"states_{split}"].astype("float32"),
            "covs": np_data[f"covs_{split}"].astype("float32"),
            "targets": np_data[f"targets_{split}"].astype("float32"),
            "state_shift": np_data[f"state_shift_{split}"].astype("float32"),
            "state_scale": np_data[f"state_scale_{split}"].astype("float32"),
            "tickers": np_data[f"tickers_{split}"],
        }
    return splits


def _infer_dim(splits: Dict[str, Dict[str, np.ndarray]], key: str) -> int:
    for split in SPLIT_NAMES:
        arr = splits[split][key]
        if arr.size > 0:
            return arr.shape[-1]
    raise ValueError(f"Unable to infer dimension for {key}; no data available.")


def _evaluate_split(model: VPForecaster,
                    split_batch: Dict[str, np.ndarray]) -> Optional[Dict[str, float]]:
    states = split_batch["states"]
    if states.size == 0:
        return None
    covs = split_batch["covs"]
    state_shift = split_batch["state_shift"]
    state_scale = split_batch["state_scale"]
    targets = split_batch["targets"]
    pred_norm, _ = model((states, covs), training=False)
    pred_norm = pred_norm.numpy()
    pred_len = pred_norm.shape[1]
    target_norm = targets[:, -pred_len:, :]
    scale_expanded = state_scale[:, None, :]
    shift_expanded = state_shift[:, None, :]
    pred_raw = pred_norm * scale_expanded + shift_expanded
    target_raw = target_norm * scale_expanded + shift_expanded
    err = pred_raw - target_raw
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mape = float(np.mean(np.abs(err) / (np.abs(target_raw) + 1e-6)))
    iden = float(identity_violation(tf.convert_to_tensor(pred_raw)).numpy())
    return {
        "mae_raw": mae,
        "rmse_raw": rmse,
        "mape_raw": mape,
        "identity_violation": iden,
    }


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train the VPForecaster model with structured logging and baselines.")
    parser.add_argument("--config", default=get_default_config_path(), help="Path to YAML config file.")
    parser.add_argument("--variant", default="base", help="Name of preprocessing variant under data/processed/.")
    parser.add_argument("--experiment-tag", default=None, help="Optional tag to include in the results folder name.")
    args = parser.parse_args(cli_args)

    cfg = load_config(args.config)
    proc_dir, dataset_path, summary_path, results_root = _prepare_paths(cfg, args.variant)

    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found. Run script/00_preprocess.py --variant {args.variant} first.")

    data = np.load(dataset_path)
    splits_data = _load_split_batches(data)
    state_columns = data["state_columns"].astype("U24").tolist()
    cov_columns = data["cov_columns"].astype("U60").tolist()
    train_data = splits_data["train"]
    if train_data["states"].size == 0 or train_data["states"].ndim != 3:
        raise ValueError("Training split is empty or malformed; ensure preprocessing produced valid sequences.")

    training_cfg = cfg.training or {}
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    epochs = int(training_cfg.get("epochs", 50))
    weight_identity = float(training_cfg.get("weight_identity", 5.0))
    weight_growth = float(training_cfg.get("weight_growth", 1.0))
    weight_level = float(training_cfg.get("weight_level", 0.3))
    hidden_dim = int(training_cfg.get("hidden_dim", 16))
    state_dim = _infer_dim(splits_data, "states")
    cov_dim = _infer_dim(splits_data, "covs")
    driver_dim = len(DRIVER_COLUMNS)

    state_weight_cfg = training_cfg.get("state_loss_weights")
    state_weight_vector = None
    if state_weight_cfg:
        if isinstance(state_weight_cfg, dict):
            name_to_idx = {name: idx for idx, name in enumerate(state_columns)}
            weights = np.ones(state_dim, dtype="float32")
            for name, value in state_weight_cfg.items():
                idx = name_to_idx.get(name)
                if idx is None:
                    logger.warning("State weight provided for unknown column %s; skipping.", name)
                    continue
                weights[idx] = float(value)
            state_weight_vector = weights
        else:
            weights = np.array(state_weight_cfg, dtype="float32")
            if weights.shape[0] == state_dim:
                state_weight_vector = weights
            else:
                logger.warning("state_loss_weights length %s does not match state_dim %s; ignoring.", weights.shape[0], state_dim)

    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        for gpu in gpu_devices:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:  # pragma: no cover - safety fallback
                pass
        device_msg = f"GPU(s) detected: {[dev.name for dev in gpu_devices]}"
    else:
        device_msg = "No GPU detected; defaulting to CPU."
    logger.info(device_msg)

    model = VPForecaster(cov_dim=cov_dim, driver_dim=driver_dim, state_dim=state_dim, hidden=hidden_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.experiment_tag}_{timestamp}" if args.experiment_tag else f"run_{timestamp}"
    run_dir = results_root / run_name
    ensure_dir(str(run_dir))

    loss_history = []
    train_batch = {
        "states": train_data["states"],
        "covs": train_data["covs"],
        "target_states": train_data["targets"],
        "state_shift": train_data["state_shift"],
        "state_scale": train_data["state_scale"],
    }
    for epoch in range(epochs):
        logs = training_step(
            model,
            train_batch,
            optimizer,
            w_identity=weight_identity,
            growth_weight=weight_growth,
            level_weight=weight_level,
            state_weights=state_weight_vector,
        )
        record = {
            "epoch": epoch + 1,
            "losses": {
                "total": logs["loss"],
                "growth": logs["loss_growth"],
                "level": logs["loss_level"],
                "identity": logs["loss_id"],
            },
            "metrics": {
                "mae_states": logs.get("mae_states"),
                "mae_growth": logs.get("mae_growth"),
                "identity_violation": logs.get("identity_violation"),
            },
        }
        loss_history.append(record)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(
                "Epoch %d | loss=%.4f growth=%.4f level=%.4f identity=%.4f",
                epoch + 1,
                logs["loss"],
                logs["loss_growth"],
                logs["loss_level"],
                logs["loss_id"],
            )

    weights_path = run_dir / "model.weights.h5"
    model.save_weights(weights_path)

    (run_dir / "training_logs.json").write_text(json.dumps(loss_history, indent=2))

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    learner = {
        "config_path": args.config,
        "variant": args.variant,
        "dataset_summary": summary,
        "model": {
            "state_dim": state_dim,
            "driver_dim": driver_dim,
            "cov_dim": cov_dim,
            "hidden_dim": hidden_dim,
        },
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_identity": weight_identity,
            "weight_growth": weight_growth,
            "weight_level": weight_level,
            "state_loss_weights": state_weight_cfg,
        },
        "device": device_msg,
        "run_dir": str(run_dir),
        "run_name": run_name,
    }
    (run_dir / "learner.json").write_text(json.dumps(learner, indent=2))

    eval_metrics = {}
    for split in SPLIT_NAMES:
        metrics = _evaluate_split(model, splits_data[split])
        if metrics:
            eval_metrics[split] = metrics
    (run_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))

    # Classical baselines & plots stored in same run directory
    baseline_summary = run_baselines(
        config_path=args.config,
        variant=args.variant,
        dataset_path=str(dataset_path),
        run_dir=run_dir,
        sample_plot_tickers=summary.get("tickers", [])[:4],
        hidden_dim=hidden_dim,
    )
    (run_dir / "baseline_metrics.json").write_text(json.dumps(baseline_summary, indent=2))

    logger.info("Training artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()
