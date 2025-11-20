from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def _relu_scalar(value: float) -> float:
    return float(max(value, 0.0))


@dataclass
class SimulationMetrics:
    mae: float
    max_balance_gap: float


class AccountingSimulator:
    """Deterministic accounting simulator inspired by Vélez-Pareja cash budgets."""

    def roll(self, initial_state: np.ndarray, drivers: np.ndarray) -> np.ndarray:
        state = np.array(initial_state, dtype=float)
        outputs = []
        for drv in drivers:
            state = self._step(state, np.array(drv, dtype=float))
            outputs.append(state)
        return np.array(outputs)

    def _step(self, state: np.ndarray, driver: np.ndarray) -> np.ndarray:
        (S, C, SG, D, AR, INV, AP, PPE, CASH, DEBT, EQ, RE) = state
        (
            gS,
            gm,
            sga_ratio,
            dep_rate,
            dso,
            dio,
            dpo,
            capex_ratio,
            tax,
            int_rate,
            payout,
        ) = driver

        S_next = _relu_scalar(S) * (1.0 + gS)
        C_next = _relu_scalar(S_next) * max(1.0 - gm, 0.0)
        SG_next = _relu_scalar(S_next) * max(sga_ratio, 0.0)
        Dep_next = _relu_scalar(PPE) * max(dep_rate, 0.0)

        AR_next = _relu_scalar(S_next) * max(dso, 0.0) / 365.0
        INV_next = _relu_scalar(C_next) * max(dio, 0.0) / 365.0
        AP_next = _relu_scalar(C_next) * max(dpo, 0.0) / 365.0

        EBIT_next = S_next - C_next - SG_next - Dep_next
        INT_next = np.maximum(int_rate, 0.0) * np.maximum(DEBT, 0.0)
        EBT_next = EBIT_next - INT_next
        tax_rate = np.clip(tax, 0.0, 0.6)
        NOPAT_next = EBT_next * (1.0 - tax_rate)
        NI_next = NOPAT_next

        CAPEX_next = _relu_scalar(S_next) * max(capex_ratio, 0.0)
        PPE_next = PPE + CAPEX_next - Dep_next

        dAR = AR_next - AR
        dINV = INV_next - INV
        dAP = AP_next - AP
        dWC = dAR + dINV - dAP

        FCF_like = NI_next + Dep_next - dWC - CAPEX_next
        DIV_next = np.maximum(payout, 0.0) * np.maximum(NI_next, 0.0)
        surplus = FCF_like - DIV_next
        repay = min(max(surplus, 0.0), max(DEBT, 0.0))
        borrow = max(-surplus, 0.0)
        DEBT_next = DEBT - repay + borrow
        CASH_next = CASH + max(surplus - repay, 0.0)

        RE_next = RE + NI_next - DIV_next
        EQ_next = EQ + NI_next - DIV_next

        ASSETS_next = CASH_next + AR_next + INV_next + PPE_next
        L_plus_E_next = AP_next + DEBT_next + EQ_next
        EQ_resid = EQ_next + (ASSETS_next - L_plus_E_next)

        return np.array([
            S_next,
            C_next,
            SG_next,
            Dep_next,
            AR_next,
            INV_next,
            AP_next,
            PPE_next,
            CASH_next,
            DEBT_next,
            EQ_resid,
            RE_next,
        ])


def evaluate_simulation(actual: np.ndarray, predicted: np.ndarray) -> SimulationMetrics:
    if actual.size == 0 or predicted.size == 0:
        return SimulationMetrics(mae=float("nan"), max_balance_gap=float("nan"))
    steps = min(actual.shape[0], predicted.shape[0])
    actual_slice = actual[:steps]
    predicted_slice = predicted[:steps]
    mae = float(np.mean(np.abs(predicted_slice - actual_slice)))
    assets = predicted_slice[:, 8] + predicted_slice[:, 4] + predicted_slice[:, 5] + predicted_slice[:, 7]
    liability_equity = predicted_slice[:, 6] + predicted_slice[:, 9] + predicted_slice[:, 10]
    max_gap = float(np.max(np.abs(assets - liability_equity)))
    return SimulationMetrics(mae=mae, max_balance_gap=max_gap)


def format_metrics(metrics: SimulationMetrics) -> Dict[str, float]:
    return {"mae": metrics.mae, "max_balance_gap": metrics.max_balance_gap}
