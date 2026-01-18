from __future__ import annotations

"""Generate CFO/CEO action recommendations from forecast deltas."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from src.llm.apiyi import extract_chat_content, request_chat_completion
from src.llm.prompt_config import get_prompt_section, render_prompt
from src.llm.prompt_logger import append_prompt_log
from src.model.dynamics_tf import DRIVER_ORDER, STATE_ORDER


def _state_summary(prev_state: np.ndarray, next_state: np.ndarray) -> Dict[str, float]:
    summary = {}
    for name, prev, nxt in zip(STATE_ORDER, prev_state, next_state):
        summary[f"{name}_prev"] = float(prev)
        summary[f"{name}_next"] = float(nxt)
        summary[f"{name}_delta"] = float(nxt - prev)
    return summary


def _driver_changes(last_driver: np.ndarray, pred_driver: np.ndarray, top_k: int = 4) -> Dict[str, float]:
    deltas = pred_driver - last_driver
    ranked = sorted(
        zip(DRIVER_ORDER, deltas.tolist(), pred_driver.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return {name: float(delta) for name, delta, _ in ranked[:top_k]}


def build_cfo_prompt(
    ticker: str,
    prev_state: np.ndarray,
    next_state: np.ndarray,
    last_driver: np.ndarray,
    pred_driver: np.ndarray,
) -> Sequence[Mapping[str, str]]:
    summary = _state_summary(prev_state, next_state)
    drivers = _driver_changes(last_driver, pred_driver)
    # Provide both state deltas and driver shifts to ground recommendations.
    section = get_prompt_section("cfo_recommendation")
    state_json = json.dumps(summary, indent=2)
    driver_json = json.dumps(drivers, indent=2)
    return render_prompt(
        section,
        ticker=ticker,
        state_json=state_json,
        driver_json=driver_json,
    )


async def generate_cfo_recommendation(
    ticker: str,
    prev_state: np.ndarray,
    next_state: np.ndarray,
    last_driver: np.ndarray,
    pred_driver: np.ndarray,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    prompt_log_path: Path | None = None,
) -> Dict[str, str]:
    messages = build_cfo_prompt(ticker, prev_state, next_state, last_driver, pred_driver)
    response = await request_chat_completion(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=700,
    )
    content = extract_chat_content(response)
    if prompt_log_path is not None:
        append_prompt_log(prompt_log_path, title=f"cfo_recommendation:{ticker}", messages=messages, response=content)
    return {
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "content": content,
    }
