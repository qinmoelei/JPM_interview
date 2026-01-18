from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Mapping, Sequence

import numpy as np

from src.llm.apiyi import extract_chat_content, request_chat_completion
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
    content = (
        f"Company: {ticker}\n"
        "Balance sheet forecast summary (next period vs previous):\n"
        f"{json.dumps(summary, indent=2)}\n"
        "Largest driver changes (delta):\n"
        f"{json.dumps(drivers, indent=2)}\n\n"
        "Provide CFO recommendations. Requirements:\n"
        "- Start with 1-2 sentence core conclusion.\n"
        "- List at least 3 actions, each formatted as: Action -> Impact path -> Monitor metric.\n"
        "- Use concrete operational levers (working capital, capex, financing, pricing, etc.).\n"
    )
    return [
        {
            "role": "system",
            "content": "You are a seasoned CFO advisor. Be concise and action-oriented.",
        },
        {"role": "user", "content": content},
    ]


async def generate_cfo_recommendation(
    ticker: str,
    prev_state: np.ndarray,
    next_state: np.ndarray,
    last_driver: np.ndarray,
    pred_driver: np.ndarray,
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> Dict[str, str]:
    messages = build_cfo_prompt(ticker, prev_state, next_state, last_driver, pred_driver)
    response = await request_chat_completion(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=700,
    )
    return {
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "content": extract_chat_content(response),
    }
