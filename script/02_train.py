from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datahandler.preprocess import DRIVER_COLUMNS
from src.model.trainer import VPForecaster, training_step
from src.utils.io import ensure_dir, get_default_config_path, load_config
from src.utils.logging import get_logger
from script.run_baselines import run_baselines

logger = get_logger(__name__)


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
    states = data["states"].astype("float32")
    covs = data["covs"].astype("float32")
    targets = data["target_states"].astype("float32")
    if states.ndim != 3:
        raise ValueError("states array must be 3-D [B, T, state_dim]")

    training_cfg = cfg.training or {}
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    epochs = int(training_cfg.get("epochs", 50))
    weight_identity = float(training_cfg.get("weight_identity", 1000.0))
    hidden_dim = int(training_cfg.get("hidden_dim", 16))

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

    cov_dim = covs.shape[-1]
    state_dim = states.shape[-1]
    driver_dim = len(DRIVER_COLUMNS)

    model = VPForecaster(cov_dim=cov_dim, driver_dim=driver_dim, state_dim=state_dim, hidden=hidden_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if args.experiment_tag:
        run_name = f"{args.experiment_tag}_{timestamp}"
    else:
        run_name = f"run_{timestamp}"
    run_dir = results_root / run_name
    ensure_dir(str(run_dir))

    loss_history = []
    batch = {"states": states, "covs": covs, "target_states": targets}
    for epoch in range(epochs):
        logs = training_step(model, batch, optimizer, w_identity=weight_identity)
        record = {
            "epoch": epoch + 1,
            "losses": {
                "total": logs["loss"],
                "fit": logs["loss_fit"],
                "identity": logs["loss_id"],
            },
            "metrics": {
                "mae_states": logs.get("mae_states"),
                "identity_violation": logs.get("identity_violation"),
            },
        }
        loss_history.append(record)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(
                "Epoch %d | loss=%.4f fit=%.4f identity=%.4f",
                epoch + 1,
                logs["loss"],
                logs["loss_fit"],
                logs["loss_id"],
            )

    weights_path = run_dir / "model.weights.h5"
    model.save_weights(weights_path)

    logs_path = run_dir / "training_logs.json"
    logs_path.write_text(json.dumps(loss_history, indent=2))

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
        },
        "device": device_msg,
        "run_dir": str(run_dir),
        "run_name": run_name,
    }
    (run_dir / "learner.json").write_text(json.dumps(learner, indent=2))

    # Classical baselines & plots stored in same run directory
    baseline_summary = run_baselines(
        config_path=args.config,
        variant=args.variant,
        dataset_path=str(dataset_path),
        run_dir=run_dir,
        sample_plot_tickers=summary.get("tickers", [])[:4],
    )
    (run_dir / "baseline_metrics.json").write_text(json.dumps(baseline_summary, indent=2))

    logger.info("Training artifacts saved to %s", run_dir)


if __name__ == "__main__":
    main()
