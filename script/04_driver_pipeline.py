from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import uuid
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.driver_workflow import (
    DriverMLP,
    DriverBatch,
    AR1Model,
    collect_targets,
    evaluate_driver_predictions,
    evaluate_state_predictions,
    load_driver_dataset,
    predict_sliding,
    build_driver_batch,
    save_experiment,
)
from src.model.simulator import AccountingSimulator
from src.utils.io import get_default_config_path, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _ensure_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if not path.is_absolute():
        path = root / path
    return path


def _count_predictions(preds: Mapping[str, Mapping[int, object]]) -> int:
    return sum(len(bucket) for bucket in preds.values())


def _make_exp_id(method: str) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{method}_{stamp}_{suffix}"


def _dataset_summary(frames, train_batch: DriverBatch) -> Dict[str, int]:
    val_targets = sum(len(frame.split.val) for frame in frames)
    test_targets = sum(len(frame.split.test) for frame in frames)
    return {
        "num_tickers": len(frames),
        "train_samples": int(train_batch.X.shape[0]),
        "val_targets": int(val_targets),
        "test_targets": int(test_targets),
    }


def _eval_and_save(
    exp_id: str,
    out_root: Path,
    method: str,
    params: Mapping[str, object],
    dataset_info: Mapping[str, object],
    preds_val,
    preds_test,
    val_targets,
    test_targets,
    frames,
    log_lines: Sequence[str],
    include_val: bool,
) -> Dict[str, object]:
    driver_val = evaluate_driver_predictions(preds_val, val_targets) if include_val else {}
    driver_test = evaluate_driver_predictions(preds_test, test_targets)
    state_val = evaluate_state_predictions(preds_val, frames) if include_val else {}
    state_test = evaluate_state_predictions(preds_test, frames)
    exp_dir = out_root / exp_id

    def per_ticker_driver(preds, targets):
        rows = []
        for tk, bucket in preds.items():
            tgt_bucket = targets.get(tk, {})
            errs = []
            rel1 = []
            rel2 = []
            for idx, pred in bucket.items():
                tgt = tgt_bucket.get(idx)
                if tgt is None:
                    continue
                diff = pred - tgt
                errs.append(diff ** 2)
                denom = np.abs(tgt) + 1e-8
                rel1.append(np.abs(diff) / denom)
                rel2.append((diff ** 2) / (denom ** 2))
            if not errs:
                continue
            mat = np.vstack(errs)
            mae_arr = np.sqrt(mat)
            rows.append(
                {
                    "ticker": tk,
                    "mse": float(mat.mean()),
                    "mae": float(mae_arr.mean()),
                    "rel_l1": float(np.vstack(rel1).mean()) if rel1 else float("nan"),
                    "rel_l2": float(np.vstack(rel2).mean()) if rel2 else float("nan"),
                    "n": len(errs),
                }
            )
        return rows

    def per_ticker_state(preds):
        rows = []
        for frame in frames:
            bucket = preds.get(frame.ticker)
            if not bucket:
                continue
            errs = []
            rel1 = []
            rel2 = []
            for idx, driver_pred in bucket.items():
                if idx + 1 >= frame.states.shape[0]:
                    continue
                prev_state = frame.states[idx]
                true_state = frame.states[idx + 1]
                next_state = AccountingSimulator()._step(prev_state, driver_pred)
                diff = next_state - true_state
                errs.append(diff ** 2)
                denom = np.abs(true_state) + 1e-8
                rel1.append(np.abs(diff) / denom)
                rel2.append((diff ** 2) / (denom ** 2))
            if not errs:
                continue
            mat = np.vstack(errs)
            mae_arr = np.sqrt(mat)
            rows.append(
                {
                    "ticker": frame.ticker,
                    "mse": float(mat.mean()),
                    "mae": float(mae_arr.mean()),
                    "rel_l1": float(np.vstack(rel1).mean()) if rel1 else float("nan"),
                    "rel_l2": float(np.vstack(rel2).mean()) if rel2 else float("nan"),
                    "n": len(errs),
                }
            )
        return rows

    payload = {
        "exp_id": exp_id,
        "method": method,
        "params": dict(params),
        "dataset": dict(dataset_info),
        "driver_metrics": {"val": driver_val, "test": driver_test} if include_val else {"test": driver_test},
        "state_metrics": {"val": state_val, "test": state_test} if include_val else {"test": state_test},
        "predictions": {
            "val": _count_predictions(preds_val),
            "test": _count_predictions(preds_test),
        },
    }
    save_experiment(exp_dir, payload, log_lines)
    per_ticker_payload = {
        "driver": {
            "val": per_ticker_driver(preds_val, val_targets) if include_val else [],
            "test": per_ticker_driver(preds_test, test_targets),
        },
        "state": {
            "val": per_ticker_state(preds_val) if include_val else [],
            "test": per_ticker_state(preds_test),
        },
    }
    (exp_dir / "per_ticker.json").write_text(json.dumps(per_ticker_payload, indent=2))
    LOGGER.info(
        "%s → driver MSE test=%.4e, state MSE test=%.4e | state MAE %.4e relL1 %.4e",
        method,
        driver_test.get("mse", float("nan")),
        state_test.get("mse", float("nan")),
        state_test.get("mae", float("nan")),
        state_test.get("rel_l1", float("nan")),
    )
    return {
        "method": method,
        "exp_id": exp_id,
        "driver_mse_test": driver_test.get("mse"),
        "driver_mae_test": driver_test.get("mae"),
        "driver_rel_l1_test": driver_test.get("rel_l1"),
        "driver_rel_l2_test": driver_test.get("rel_l2"),
        "state_mse_test": state_test.get("mse"),
        "state_mae_test": state_test.get("mae"),
        "state_rel_l1_test": state_test.get("rel_l1"),
        "state_rel_l2_test": state_test.get("rel_l2"),
    }


def run_pipeline(
    config_path: str,
    variant: str,
    results_subdir: str,
    window: int,
    history_lag: int,
) -> None:
    cfg = load_config(config_path)
    proc_root = _ensure_path(PROJECT_ROOT, cfg.paths.get("proc_dir", "data/processed"))
    proc_dir = proc_root / variant
    if not proc_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {proc_dir}")
    frames = load_driver_dataset(proc_dir, cfg.tickers, min_transitions=2)
    if not frames:
        raise RuntimeError("No usable tickers found for driver modeling.")
    train_batch = build_driver_batch(frames, "train", lag=history_lag)
    val_batch = build_driver_batch(frames, "val", lag=history_lag)
    test_batch = build_driver_batch(frames, "test", lag=history_lag)
    dataset_info = _dataset_summary(frames, train_batch)
    dataset_info.update({"variant": variant, "history_lag": history_lag})

    val_targets = collect_targets(frames, "val")
    test_targets = collect_targets(frames, "test")

    results_root = _ensure_path(PROJECT_ROOT, cfg.paths.get("results_dir", "results"))
    exp_root = results_root / results_subdir
    exp_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    # Perfect baseline
    perfect_val = val_targets
    perfect_test = test_targets
    summary_rows.append(
        _eval_and_save(
            exp_id=_make_exp_id("perfect"),
            out_root=exp_root,
            method="perfect",
            params={"description": "Ground-truth drivers"},
            dataset_info=dataset_info,
            preds_val=perfect_val,
            preds_test=perfect_test,
            val_targets=val_targets,
            test_targets=test_targets,
            frames=frames,
            log_lines=["Perfect driver baseline (no training)."],
            include_val=False,
        )
    )

    # Sliding window
    sliding_val = predict_sliding(frames, "val", window=window)
    sliding_test = predict_sliding(frames, "test", window=window)
    summary_rows.append(
        _eval_and_save(
            exp_id=_make_exp_id("sliding"),
            out_root=exp_root,
            method="sliding_mean",
            params={"window": window},
            dataset_info=dataset_info,
            preds_val=sliding_val,
            preds_test=sliding_test,
            val_targets=val_targets,
            test_targets=test_targets,
            frames=frames,
            log_lines=[f"Sliding mean per ticker with window={window}."],
            include_val=False,
        )
    )

    if train_batch.X.shape[0] > 0:
        ar_model = AR1Model()
        ar_model.fit(train_batch)
        ar_val = ar_model.predict(frames, "val", lag=history_lag)
        ar_test = ar_model.predict(frames, "test", lag=history_lag)
        summary_rows.append(
            _eval_and_save(
                exp_id=_make_exp_id("ar1"),
                out_root=exp_root,
                method="ar1",
                params={"history_lag": history_lag},
                dataset_info=dataset_info,
                preds_val=ar_val,
                preds_test=ar_test,
                val_targets=val_targets,
                test_targets=test_targets,
                frames=frames,
                log_lines=[
                    "Pooled AR(1) per driver dimension.",
                    f"Train samples: {train_batch.X.shape[0]}",
                ],
                include_val=False,
            )
        )
    else:
        LOGGER.warning("Skipping AR(1): no training samples available.")

    if train_batch.X.shape[0] > 0:
        mlp = DriverMLP(hidden_units=(32, 32), epochs=500, batch_size=16, patience=40)
        mlp.fit(train_batch, val_batch if val_batch.X.shape[0] else None)
        nn_val = mlp.predict(frames, "val", lag=history_lag)
        nn_test = mlp.predict(frames, "test", lag=history_lag)
        summary_rows.append(
            _eval_and_save(
                exp_id=_make_exp_id("mlp"),
                out_root=exp_root,
                method="small_mlp",
                params={"history_lag": history_lag, "hidden_units": [32, 32], "epochs": 500},
                dataset_info=dataset_info,
                preds_val=nn_val,
                preds_test=nn_test,
                val_targets=val_targets,
                test_targets=test_targets,
                frames=frames,
                log_lines=[
                    "Small pooled MLP with 2×32 hidden units.",
                    f"Train samples: {train_batch.X.shape[0]}",
                    f"Validation samples: {val_batch.X.shape[0]}",
                ],
                include_val=True,
            )
        )
    else:
        LOGGER.warning("Skipping MLP: no training samples available.")

    analysis_path = exp_root / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    analysis_path.write_text(json.dumps(summary_rows, indent=2))
    LOGGER.info("Wrote analysis summary to %s", analysis_path)


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Driver time-series modeling and structural backtests.")
    parser.add_argument("--config", default=get_default_config_path(), help="Path to YAML config.")
    parser.add_argument("--variant", default="accounting_tank", help="Processed data subdirectory.")
    parser.add_argument("--results-subdir", default="driver_experiments", help="Subdirectory under results/ for experiments.")
    parser.add_argument("--window", type=int, default=2, help="Sliding mean window length.")
    parser.add_argument("--history-lag", type=int, default=1, help="Number of lags for supervised driver samples.")
    args = parser.parse_args(cli_args)

    run_pipeline(
        config_path=args.config,
        variant=args.variant,
        results_subdir=args.results_subdir,
        window=args.window,
        history_lag=args.history_lag,
    )


if __name__ == "__main__":
    main()
