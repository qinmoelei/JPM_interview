from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.model.simulator import AccountingSimulator
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


NUM_STATES = 15
NUM_DRIVERS = 13


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class TickerFrame:
    ticker: str
    states: np.ndarray
    state_index: List[str]
    drivers: np.ndarray
    driver_index: List[str]
    split: SplitIndices


@dataclass
class DriverBatch:
    X: np.ndarray
    y: np.ndarray
    refs: List[Tuple[str, int]]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    return df


def _split_transitions(n_transitions: int) -> Optional[SplitIndices]:
    if n_transitions < 1:
        return None
    train_end = max(n_transitions - 2, 0)
    val_start = train_end
    val_end = max(n_transitions - 1, val_start)
    train_idx = np.arange(0, train_end, dtype=int)
    val_idx = np.arange(val_start, val_end, dtype=int)
    test_idx = np.arange(val_end, n_transitions, dtype=int)
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def load_ticker_frame(proc_dir: Path, ticker: str) -> Optional[TickerFrame]:
    states_path = proc_dir / f"{ticker}_states.csv"
    drivers_path = proc_dir / f"{ticker}_drivers.csv"
    if not states_path.exists() or not drivers_path.exists():
        LOGGER.debug("Missing data for %s under %s", ticker, proc_dir)
        return None
    states_df = _read_csv(states_path)
    drivers_df = _read_csv(drivers_path)
    states = states_df.to_numpy(dtype=float)
    drivers = drivers_df.to_numpy(dtype=float)
    if states.shape[0] < 3 or drivers.shape[0] < 2:
        LOGGER.debug("Skipping %s: not enough history (states=%d, drivers=%d)", ticker, states.shape[0], drivers.shape[0])
        return None
    split = _split_transitions(drivers.shape[0])
    if split is None:
        return None
    return TickerFrame(
        ticker=ticker,
        states=states,
        state_index=[str(idx) for idx in states_df.index],
        drivers=drivers,
        driver_index=[str(idx) for idx in drivers_df.index],
        split=split,
    )


def load_driver_dataset(proc_dir: Path, tickers: Sequence[str], min_transitions: int = 2) -> List[TickerFrame]:
    frames: List[TickerFrame] = []
    for tk in tickers:
        frame = load_ticker_frame(proc_dir, tk)
        if frame is None:
            continue
        if frame.drivers.shape[0] < min_transitions:
            continue
        frames.append(frame)
    LOGGER.info("Loaded %d/%d tickers with usable driver histories.", len(frames), len(tickers))
    return frames


def build_driver_batch(frames: Sequence[TickerFrame], split_name: str, lag: int = 1) -> DriverBatch:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    refs: List[Tuple[str, int]] = []
    for frame in frames:
        indices = getattr(frame.split, split_name)
        for idx in indices:
            prev_idx = idx - lag
            if prev_idx < 0:
                continue
            X_list.append(frame.drivers[prev_idx])
            y_list.append(frame.drivers[idx])
            refs.append((frame.ticker, idx))
    if not X_list:
        return DriverBatch(
            X=np.zeros((0, NUM_DRIVERS), dtype=float),
            y=np.zeros((0, NUM_DRIVERS), dtype=float),
            refs=[],
        )
    return DriverBatch(
        X=np.vstack(X_list),
        y=np.vstack(y_list),
        refs=refs,
    )


def collect_targets(frames: Sequence[TickerFrame], split_name: str) -> Dict[str, Dict[int, np.ndarray]]:
    targets: Dict[str, Dict[int, np.ndarray]] = {}
    for frame in frames:
        indices = getattr(frame.split, split_name)
        if len(indices) == 0:
            continue
        bucket: Dict[int, np.ndarray] = {}
        for idx in indices:
            bucket[idx] = frame.drivers[idx]
        if bucket:
            targets[frame.ticker] = bucket
    return targets


def predict_sliding(frames: Sequence[TickerFrame], split_name: str, window: int = 2) -> Dict[str, Dict[int, np.ndarray]]:
    predictions: Dict[str, Dict[int, np.ndarray]] = {}
    for frame in frames:
        indices = getattr(frame.split, split_name)
        pred_bucket: Dict[int, np.ndarray] = {}
        for idx in indices:
            start = max(0, idx - window)
            history = frame.drivers[start:idx]
            if history.size == 0:
                continue
            pred_bucket[idx] = history.mean(axis=0)
        if pred_bucket:
            predictions[frame.ticker] = pred_bucket
    return predictions


class AR1Model:
    def __init__(self) -> None:
        self.coef = np.zeros(NUM_DRIVERS, dtype=float)
        self.intercept = np.zeros(NUM_DRIVERS, dtype=float)

    def fit(self, batch: DriverBatch) -> None:
        if batch.X.shape[0] == 0:
            raise ValueError("AR1Model cannot fit: empty training batch.")
        for j in range(NUM_DRIVERS):
            x_prev = batch.X[:, j]
            x_curr = batch.y[:, j]
            var = np.var(x_prev)
            if var < 1e-10:
                self.coef[j] = 0.0
                self.intercept[j] = float(np.mean(x_curr))
            else:
                cov = np.cov(x_prev, x_curr, bias=True)[0, 1]
                phi = cov / var
                mu_prev = float(np.mean(x_prev))
                mu_curr = float(np.mean(x_curr))
                self.coef[j] = float(phi)
                self.intercept[j] = float(mu_curr - phi * mu_prev)

    def predict_one(self, prev_driver: np.ndarray) -> np.ndarray:
        return self.intercept + self.coef * prev_driver

    def predict(self, frames: Sequence[TickerFrame], split_name: str, lag: int = 1) -> Dict[str, Dict[int, np.ndarray]]:
        preds: Dict[str, Dict[int, np.ndarray]] = {}
        for frame in frames:
            indices = getattr(frame.split, split_name)
            bucket: Dict[int, np.ndarray] = {}
            for idx in indices:
                prev_idx = idx - lag
                if prev_idx < 0:
                    continue
                bucket[idx] = self.predict_one(frame.drivers[prev_idx])
            if bucket:
                preds[frame.ticker] = bucket
        return preds


class DriverMLP:
    def __init__(self, hidden_units: Sequence[int] = (32, 32), epochs: int = 500, batch_size: int = 16, patience: int = 30, seed: int = 2024) -> None:
        import tensorflow as tf  # Local import to avoid TF dependency when unused

        self.tf = tf
        self.hidden_units = list(hidden_units)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.model: Optional[tf.keras.Model] = None
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.y_mean: Optional[np.ndarray] = None
        self.y_std: Optional[np.ndarray] = None

    def _build_model(self, input_dim: int) -> "DriverMLP.tf.keras.Model":
        tf = self.tf
        inputs = tf.keras.Input(shape=(input_dim,))
        x = inputs
        for units in self.hidden_units:
            x = tf.keras.layers.Dense(units, activation="relu")(x)
        outputs = tf.keras.layers.Dense(NUM_DRIVERS, activation=None)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
        )
        return model

    def _normalize(self, X: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mean is None or std is None:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std = np.where(std < 1e-6, 1.0, std)
        X_norm = (X - mean) / std
        return X_norm, mean, std

    def fit(self, train_batch: DriverBatch, val_batch: Optional[DriverBatch] = None) -> None:
        if train_batch.X.shape[0] == 0:
            raise ValueError("DriverMLP cannot fit: empty training batch.")
        self.tf.random.set_seed(self.seed)
        X_train, self.x_mean, self.x_std = self._normalize(train_batch.X, None, None)
        Y_train, self.y_mean, self.y_std = self._normalize(train_batch.y, None, None)
        X_val = Y_val = None
        if val_batch is not None and val_batch.X.shape[0]:
            X_val, _, _ = self._normalize(val_batch.X, self.x_mean, self.x_std)
            Y_val, _, _ = self._normalize(val_batch.y, self.y_mean, self.y_std)
        self.model = self._build_model(X_train.shape[1])
        callbacks: List["DriverMLP.tf.keras.callbacks.Callback"] = []
        if X_val is not None and Y_val is not None:
            callbacks.append(
                self.tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                )
            )
        self.model.fit(
            X_train,
            Y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_data=(X_val, Y_val) if X_val is not None and Y_val is not None else None,
            callbacks=callbacks,
        )

    def predict(self, frames: Sequence[TickerFrame], split_name: str, lag: int = 1) -> Dict[str, Dict[int, np.ndarray]]:
        if self.model is None or self.x_mean is None or self.x_std is None or self.y_mean is None or self.y_std is None:
            raise ValueError("DriverMLP must be fitted before prediction.")
        preds: Dict[str, Dict[int, np.ndarray]] = {}
        for frame in frames:
            indices = getattr(frame.split, split_name)
            bucket: Dict[int, np.ndarray] = {}
            inputs: List[np.ndarray] = []
            refs: List[int] = []
            for idx in indices:
                prev_idx = idx - lag
                if prev_idx < 0:
                    continue
                inputs.append(frame.drivers[prev_idx])
                refs.append(idx)
            if not inputs:
                continue
            X = np.vstack(inputs)
            X_norm = (X - self.x_mean) / self.x_std
            Y_norm = self.model.predict(X_norm, verbose=0)
            Y = Y_norm * self.y_std + self.y_mean
            for ref_idx, pred in zip(refs, Y):
                bucket[ref_idx] = pred
            if bucket:
                preds[frame.ticker] = bucket
        return preds


def evaluate_driver_predictions(
    preds: Mapping[str, Mapping[int, np.ndarray]],
    targets: Mapping[str, Mapping[int, np.ndarray]],
) -> Dict[str, float]:
    errors: List[np.ndarray] = []
    rel1: List[np.ndarray] = []
    rel2: List[np.ndarray] = []
    for ticker, bucket in preds.items():
        tgt_bucket = targets.get(ticker, {})
        for idx, pred in bucket.items():
            target = tgt_bucket.get(idx)
            if target is None:
                continue
            diff = pred - target
            errors.append(diff ** 2)
            denom = np.abs(target) + 1e-8
            rel1.append(np.abs(diff) / denom)
            rel2.append((diff ** 2) / (denom ** 2))
    if not errors:
        return {"mse": float("nan"), "mae": float("nan"), "rel_l1": float("nan"), "rel_l2": float("nan")}
    stacked = np.vstack(errors)
    mae_arr = np.sqrt(stacked)  # since stacked is squared error
    return {
        "mse": float(np.mean(stacked)),
        "mae": float(np.mean(mae_arr)),
        "rel_l1": float(np.mean(np.vstack(rel1))) if rel1 else float("nan"),
        "rel_l2": float(np.mean(np.vstack(rel2))) if rel2 else float("nan"),
    }


def evaluate_state_predictions(
    preds: Mapping[str, Mapping[int, np.ndarray]],
    frames: Sequence[TickerFrame],
) -> Dict[str, float]:
    simulator = AccountingSimulator()
    squared_errors: List[np.ndarray] = []
    rel1: List[np.ndarray] = []
    rel2: List[np.ndarray] = []
    for frame in frames:
        ticker_preds = preds.get(frame.ticker)
        if not ticker_preds:
            continue
        for idx, driver_pred in ticker_preds.items():
            if idx + 1 >= frame.states.shape[0]:
                continue
            prev_state = frame.states[idx]
            true_state = frame.states[idx + 1].copy()
            next_state = simulator._step(prev_state, driver_pred)
            diff = next_state - true_state
            squared_errors.append(diff ** 2)
            denom = np.abs(true_state) + 1e-8
            rel1.append(np.abs(diff) / denom)
            rel2.append((diff ** 2) / (denom ** 2))
    if not squared_errors:
        return {"mse": float("nan"), "mae": float("nan"), "rel_l1": float("nan"), "rel_l2": float("nan")}
    mat = np.vstack(squared_errors)
    mae_arr = np.sqrt(mat)
    return {
        "mse": float(np.mean(mat)),
        "mae": float(np.mean(mae_arr)),
        "rel_l1": float(np.mean(np.vstack(rel1))) if rel1 else float("nan"),
        "rel_l2": float(np.mean(np.vstack(rel2))) if rel2 else float("nan"),
    }


def save_experiment(
    out_dir: Path,
    config: Mapping[str, object],
    log_lines: Sequence[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "learner.json").write_text(json.dumps(config, indent=2))
    (out_dir / "log.txt").write_text("\n".join(log_lines))
