from __future__ import annotations

"""LLM-based driver forecasting and evaluation utilities."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.experiments.driver_workflow import (
    DriverBatch,
    TickerFrame,
    build_driver_batch,
    collect_targets,
    evaluate_driver_predictions,
    evaluate_state_predictions,
    load_driver_dataset,
)
from src.llm.apiyi import create_async_client, extract_chat_content, load_apiyi_config, request_chat_completion
from src.llm.json_utils import coerce_float, safe_json_loads
from src.llm.prompt_config import get_prompt_section, render_prompt
from src.llm.prompt_logger import append_prompt_log
from src.llm.reasoning_logger import append_reasoning_log
from src.model.dynamics_tf import DRIVER_ORDER
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


# Hard bounds prevent obviously invalid LLM outputs.
DRIVER_BOUNDS = {
    "growth": (-1.0, 1.0),
    "gross_margin": (0.0, 1.0),
    "sga_ratio": (0.0, 1.5),
    "dep_rate": (0.0, 1.0),
    "dso": (0.0, 365.0),
    "dio": (0.0, 365.0),
    "dpo": (0.0, 365.0),
    "capex_ratio": (0.0, 1.5),
    "tax_rate": (0.0, 1.0),
    "interest_rate": (0.0, 0.5),
    "payout_ratio": (0.0, 1.5),
    "net_debt_issuance_ratio": (-1.0, 1.0),
    "net_equity_issuance_ratio": (-1.0, 1.0),
}


@dataclass(frozen=True)
class LLMRunConfig:
    model: Optional[str] = None
    temperature: float = 0.0
    window: int = 3
    max_calls: Optional[int] = None
    cache_path: Optional[Path] = None
    prompt_log_path: Optional[Path] = None
    reasoning_path: Optional[Path] = None


def _history_rows(frame: TickerFrame, indices: Iterable[int]) -> List[Dict[str, float]]:
    # Convert driver rows into JSON-friendly objects keyed by period.
    history = []
    for i in indices:
        if i < 0 or i >= len(frame.driver_index):
            continue
        payload = {"period": frame.driver_index[i]}
        for name, val in zip(DRIVER_ORDER, frame.drivers[i]):
            payload[name] = float(val)
        history.append(payload)
    return history


def _build_prompt(
    ticker: str,
    train_history: Sequence[Mapping[str, float]],
    recent_history: Sequence[Mapping[str, float]],
    window: int,
) -> List[Dict[str, str]]:
    # Assemble a strict JSON-only prompt for structured LLM extraction.
    section = get_prompt_section("driver_forecast")
    driver_list = ", ".join(DRIVER_ORDER)
    train_history_json = json.dumps(train_history, indent=2)
    recent_history_json = json.dumps(recent_history, indent=2)
    return render_prompt(
        section,
        ticker=ticker,
        driver_list=driver_list,
        train_history_json=train_history_json,
        recent_history_json=recent_history_json,
        window=window,
    )


def _extract_driver_payload(payload: Mapping[str, object]) -> Tuple[Mapping[str, object], Optional[str]]:
    # Accept both nested {"drivers": {...}} and flat driver maps.
    if "drivers" in payload:
        drivers = payload.get("drivers")
        if not isinstance(drivers, Mapping):
            raise ValueError("LLM response 'drivers' must be a JSON object.")
        return drivers, payload.get("reasoning") if isinstance(payload.get("reasoning"), str) else None
    return payload, payload.get("reasoning") if isinstance(payload.get("reasoning"), str) else None


def _sanitize_prediction(pred: Mapping[str, object], fallback: Sequence[float]) -> np.ndarray:
    # Replace missing outputs with last known values and clamp to bounds.
    values: List[float] = []
    for name, last in zip(DRIVER_ORDER, fallback):
        raw = pred.get(name)
        val = coerce_float(raw)
        if val is None:
            val = float(last)
        lower, upper = DRIVER_BOUNDS.get(name, (-np.inf, np.inf))
        val = float(np.clip(val, lower, upper))
        values.append(val)
    return np.array(values, dtype=float)


async def _request_driver_prediction(
    ticker: str,
    train_history: Sequence[Mapping[str, float]],
    recent_history: Sequence[Mapping[str, float]],
    window: int,
    *,
    model: Optional[str],
    temperature: float,
    client=None,
    prompt_log_path: Optional[Path] = None,
) -> Tuple[Mapping[str, object], Optional[str]]:
    messages = _build_prompt(ticker, train_history, recent_history, window)
    response = await request_chat_completion(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=500,
        client=client,
    )
    content = extract_chat_content(response)
    if prompt_log_path is not None:
        append_prompt_log(prompt_log_path, title=f"driver_forecast:{ticker}", messages=messages, response=content)
    payload = safe_json_loads(content)
    if not isinstance(payload, Mapping):
        raise ValueError("LLM response must be a JSON object.")
    return _extract_driver_payload(payload)


def _load_cache(path: Path | None) -> Dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _save_cache(path: Path | None, cache: Mapping[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))


async def _predict_llm_drivers_async(
    frames: Sequence[TickerFrame],
    split_name: str,
    cfg: LLMRunConfig,
) -> Dict[str, Dict[int, np.ndarray]]:
    # Use a single async client for all calls to avoid event-loop shutdown warnings.
    cache = _load_cache(cfg.cache_path)
    preds: Dict[str, Dict[int, np.ndarray]] = {}
    total_calls = 0
    stop = False
    config = load_apiyi_config()
    client = create_async_client(config, max_retries=0)
    try:
        for frame in frames:
            indices = getattr(frame.split, split_name)
            train_history = _history_rows(frame, frame.split.train)
            bucket: Dict[int, np.ndarray] = {}
            for idx in indices:
                if cfg.max_calls is not None and total_calls >= cfg.max_calls:
                    stop = True
                    break
                key = f"{frame.ticker}:{idx}:w{cfg.window}"
                if key in cache:
                    cached = cache[key]
                    if isinstance(cached, Mapping):
                        pred_vec = np.array(cached.get("drivers", []), dtype=float)
                        reasoning = cached.get("reasoning") or "(cached prediction; no reasoning captured)"
                    else:
                        pred_vec = np.array(cached, dtype=float)
                        reasoning = "(cached prediction; no reasoning captured)"
                    if pred_vec.size == len(DRIVER_ORDER):
                        bucket[idx] = pred_vec
                        if cfg.reasoning_path is not None:
                            append_reasoning_log(
                                cfg.reasoning_path,
                                title=f"{split_name}:{frame.ticker}:{idx}",
                                reasoning=reasoning,
                            )
                        continue
                if idx <= 0:
                    continue
                recent_history = _history_rows(frame, range(max(0, idx - cfg.window), idx))
                fallback = frame.drivers[idx - 1]
                try:
                    payload, reasoning = await _request_driver_prediction(
                        frame.ticker,
                        train_history,
                        recent_history,
                        cfg.window,
                        model=cfg.model,
                        temperature=cfg.temperature,
                        client=client,
                        prompt_log_path=cfg.prompt_log_path,
                    )
                except Exception as exc:
                    # Fall back to the last observed driver if the call fails.
                    LOGGER.warning("LLM prediction failed for %s idx=%s: %s", frame.ticker, idx, exc)
                    payload = {name: float(val) for name, val in zip(DRIVER_ORDER, fallback)}
                    reasoning = None
                pred_vec = _sanitize_prediction(payload, fallback)
                bucket[idx] = pred_vec
                cache[key] = {"drivers": pred_vec.tolist(), "reasoning": reasoning}
                if cfg.reasoning_path is not None:
                    append_reasoning_log(
                        cfg.reasoning_path,
                        title=f"{split_name}:{frame.ticker}:{idx}",
                        reasoning=reasoning,
                    )
                total_calls += 1
            if bucket:
                preds[frame.ticker] = bucket
            if stop:
                break
    finally:
        await client.close()
    _save_cache(cfg.cache_path, cache)
    return preds


def predict_llm_drivers(
    frames: Sequence[TickerFrame],
    split_name: str,
    cfg: LLMRunConfig,
) -> Dict[str, Dict[int, np.ndarray]]:
    return asyncio.run(_predict_llm_drivers_async(frames, split_name, cfg))


def run_llm_driver_experiment(
    proc_dir: Path,
    tickers: Sequence[str],
    *,
    window: int = 3,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_calls: Optional[int] = None,
    cache_path: Optional[Path] = None,
    prompt_log_path: Optional[Path] = None,
    reasoning_path: Optional[Path] = None,
) -> Dict[str, object]:
    frames = load_driver_dataset(proc_dir, tickers)
    val_targets = collect_targets(frames, "val")
    test_targets = collect_targets(frames, "test")
    cfg = LLMRunConfig(
        model=model,
        temperature=temperature,
        window=window,
        max_calls=max_calls,
        cache_path=cache_path,
        prompt_log_path=prompt_log_path,
        reasoning_path=reasoning_path,
    )
    preds_val = predict_llm_drivers(frames, "val", cfg)
    preds_test = predict_llm_drivers(frames, "test", cfg)
    metrics = {
        "driver_val": evaluate_driver_predictions(preds_val, val_targets),
        "driver_test": evaluate_driver_predictions(preds_test, test_targets),
        "state_val": evaluate_state_predictions(preds_val, frames),
        "state_test": evaluate_state_predictions(preds_test, frames),
    }
    metadata = {
        "model": model,
        "temperature": temperature,
        "window": window,
        "max_calls": max_calls,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return {
        "metrics": metrics,
        "metadata": metadata,
        "preds_val": {k: {str(i): v.tolist() for i, v in bucket.items()} for k, bucket in preds_val.items()},
        "preds_test": {k: {str(i): v.tolist() for i, v in bucket.items()} for k, bucket in preds_test.items()},
    }


def build_mlp_baseline(frames: Sequence[TickerFrame]) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, np.ndarray]]]:
    from src.experiments.driver_workflow import DriverMLP

    train_batch = build_driver_batch(frames, "train")
    val_batch = build_driver_batch(frames, "val")
    model = DriverMLP()
    model.fit(train_batch, val_batch)
    preds_val = model.predict(frames, "val")
    preds_test = model.predict(frames, "test")
    return preds_val, preds_test
