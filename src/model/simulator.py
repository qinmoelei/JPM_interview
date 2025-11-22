from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.model.dynamics_tf import DRIVER_ORDER, STATE_ORDER, forward_step_np


@dataclass
class SimulationMetrics:
    mse: float
    mae: float
    rel_l1: float
    rel_l2: float
    max_balance_gap: float


class AccountingSimulator:
    """
    Deterministic accounting simulator for the 15-D state / 13-D driver spec.

    State (prev period, t-1):
      [S, C, SG, D, AR, INV, AP, PPE, CASH, DEBT, EQ, RE, TAX, INT, DIV]
    Driver (current period, t):
      [gS, gm, sga_ratio, dep_rate, dso, dio, dpo, capex_ratio, tax_rate, int_rate, payout_ratio, ndebt_ratio, nequity_ratio]
    """

    def roll(self, initial_state: np.ndarray, drivers: np.ndarray) -> np.ndarray:
        state = np.array(initial_state, dtype=float)
        outputs = []
        for drv in drivers:
            state = self._step(state, np.array(drv, dtype=float))
            outputs.append(state)
        return np.array(outputs)

    def _step(self, state: np.ndarray, driver: np.ndarray) -> np.ndarray:
        return forward_step_np(state, driver)


def evaluate_simulation(actual: np.ndarray, predicted: np.ndarray) -> SimulationMetrics:
    if actual.size == 0 or predicted.size == 0:
        return SimulationMetrics(
            mse=float("nan"),
            mae=float("nan"),
            rel_l1=float("nan"),
            rel_l2=float("nan"),
            max_balance_gap=float("nan"),
        )
    steps = min(actual.shape[0], predicted.shape[0])
    actual_slice = actual[:steps].copy()
    predicted_slice = predicted[:steps]
    err = predicted_slice - actual_slice
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    denom = np.abs(actual_slice) + 1e-8
    rel_l1 = float(np.mean(np.abs(err) / denom))
    rel_l2 = float(np.mean((err ** 2) / (denom ** 2)))
    assets = predicted_slice[:, 8] + predicted_slice[:, 4] + predicted_slice[:, 5] + predicted_slice[:, 7]
    liability_equity = predicted_slice[:, 6] + predicted_slice[:, 9] + predicted_slice[:, 10]
    max_gap = float(np.max(np.abs(assets - liability_equity)))
    return SimulationMetrics(mse=mse, mae=mae, rel_l1=rel_l1, rel_l2=rel_l2, max_balance_gap=max_gap)


def format_metrics(metrics: SimulationMetrics) -> Dict[str, float]:
    return {
        "mse": metrics.mse,
        "mae": metrics.mae,
        "rel_l1": metrics.rel_l1,
        "rel_l2": metrics.rel_l2,
        "max_balance_gap": metrics.max_balance_gap,
    }
