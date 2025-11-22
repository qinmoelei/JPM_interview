from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

STATE_ORDER = [
    "sales",
    "cogs",
    "sga",
    "dep",
    "ar",
    "inventory",
    "ap",
    "ppe",
    "cash",
    "debt",
    "equity",
    "retained_earnings",
    "tax_expense",
    "interest_expense",
    "dividends",
]

DRIVER_ORDER = [
    "growth",
    "gross_margin",
    "sga_ratio",
    "dep_rate",
    "dso",
    "dio",
    "dpo",
    "capex_ratio",
    "tax_rate",
    "interest_rate",
    "payout_ratio",
    "net_debt_issuance_ratio",
    "net_equity_issuance_ratio",
]

EPS = 1e-8


def forward_step_tf(state: tf.Tensor, driver: tf.Tensor, eps: float = EPS) -> tf.Tensor:
    """Single-step forward evolution y_t = f(y_{t-1}, x_t) in TF."""
    (
        S_prev,
        C_prev,
        SG_prev,
        D_prev,
        AR_prev,
        INV_prev,
        AP_prev,
        PPE_prev,
        CASH_prev,
        DEBT_prev,
        EQ_prev,
        RE_prev,
        TAX_prev,
        INT_prev,
        DIV_prev,
    ) = tf.unstack(state, axis=-1)
    (
        gS,
        gm,
        sga_ratio,
        dep_rate,
        dso,
        dio,
        dpo,
        capex_ratio,
        tax_rate,
        int_rate,
        payout,
        ndebt_ratio,
        nequity_ratio,
    ) = tf.unstack(driver, axis=-1)

    sales_base = tf.maximum(tf.abs(S_prev), eps)
    S_next = S_prev + gS * sales_base

    sales_next_base = tf.maximum(tf.abs(S_next), eps)
    C_next = (1.0 - gm) * sales_next_base
    SG_next = sga_ratio * sales_next_base

    ppe_base = tf.maximum(tf.abs(PPE_prev), eps)
    D_next = dep_rate * ppe_base

    cost_base = tf.maximum(tf.abs(C_next), eps)
    AR_next = (dso / 365.0) * sales_next_base
    INV_next = (dio / 365.0) * cost_base
    AP_next = (dpo / 365.0) * cost_base

    EBIT = S_next - C_next - SG_next - D_next
    debt_base = tf.maximum(tf.abs(DEBT_prev), eps)
    INT_next = int_rate * debt_base
    EBT = EBIT - INT_next
    ebt_base = tf.maximum(tf.abs(EBT), eps)
    TAX_next = tax_rate * ebt_base
    NI_next = EBT - TAX_next
    DIV_next = payout * tf.maximum(tf.abs(NI_next), 0.0)

    CAPEX_next = capex_ratio * sales_next_base
    PPE_next = PPE_prev + CAPEX_next - D_next

    WC_prev = AR_prev + INV_prev - AP_prev
    WC_next = AR_next + INV_next - AP_next
    dWC = WC_next - WC_prev

    FCF = NI_next + D_next - dWC - CAPEX_next
    delta_debt = ndebt_ratio * sales_next_base
    delta_eq_ext = nequity_ratio * sales_next_base

    DEBT_next = DEBT_prev + delta_debt
    RE_next = RE_prev + NI_next - DIV_next
    EQ_next = EQ_prev + delta_eq_ext + NI_next - DIV_next
    CASH_next = CASH_prev + FCF + delta_debt + delta_eq_ext - DIV_next

    return tf.stack(
        [
            S_next,
            C_next,
            SG_next,
            D_next,
            AR_next,
            INV_next,
            AP_next,
            PPE_next,
            CASH_next,
            DEBT_next,
            EQ_next,
            RE_next,
            TAX_next,
            INT_next,
            DIV_next,
        ],
        axis=-1,
    )


def roll_tf(initial_state: np.ndarray, drivers: np.ndarray) -> np.ndarray:
    """Run the TF evolution over a driver sequence and return numpy states."""
    state = tf.convert_to_tensor(initial_state, dtype=tf.float64)
    outputs = []
    for drv in drivers:
        drv_tensor = tf.convert_to_tensor(drv, dtype=tf.float64)
        state = forward_step_tf(state, drv_tensor)
        outputs.append(state)
    if not outputs:
        return np.zeros((0, len(STATE_ORDER)), dtype=float)
    stacked = tf.stack(outputs, axis=0)
    return stacked.numpy()


def forward_step_np(state: np.ndarray, driver: np.ndarray, eps: float = EPS) -> np.ndarray:
    """NumPy wrapper that mirrors forward_step_tf for downstream code."""
    (
        S_prev,
        C_prev,
        SG_prev,
        D_prev,
        AR_prev,
        INV_prev,
        AP_prev,
        PPE_prev,
        CASH_prev,
        DEBT_prev,
        EQ_prev,
        RE_prev,
        TAX_prev,
        INT_prev,
        DIV_prev,
    ) = state
    (
        gS,
        gm,
        sga_ratio,
        dep_rate,
        dso,
        dio,
        dpo,
        capex_ratio,
        tax_rate,
        int_rate,
        payout,
        ndebt_ratio,
        nequity_ratio,
    ) = driver

    sales_base = max(abs(S_prev), eps)
    S_next = S_prev + gS * sales_base

    sales_next_base = max(abs(S_next), eps)
    C_next = (1.0 - gm) * sales_next_base
    SG_next = sga_ratio * sales_next_base

    ppe_base = max(abs(PPE_prev), eps)
    D_next = dep_rate * ppe_base

    cost_base = max(abs(C_next), eps)
    AR_next = (dso / 365.0) * sales_next_base
    INV_next = (dio / 365.0) * cost_base
    AP_next = (dpo / 365.0) * cost_base

    EBIT = S_next - C_next - SG_next - D_next
    debt_base = max(abs(DEBT_prev), eps)
    INT_next = int_rate * debt_base
    EBT = EBIT - INT_next
    ebt_base = max(abs(EBT), eps)
    TAX_next = tax_rate * ebt_base
    NI_next = EBT - TAX_next
    DIV_next = payout * max(abs(NI_next), 0.0)

    CAPEX_next = capex_ratio * sales_next_base
    PPE_next = PPE_prev + CAPEX_next - D_next

    WC_prev = AR_prev + INV_prev - AP_prev
    WC_next = AR_next + INV_next - AP_next
    dWC = WC_next - WC_prev

    FCF = NI_next + D_next - dWC - CAPEX_next
    delta_debt = ndebt_ratio * sales_next_base
    delta_eq_ext = nequity_ratio * sales_next_base

    DEBT_next = DEBT_prev + delta_debt
    RE_next = RE_prev + NI_next - DIV_next
    EQ_next = EQ_prev + delta_eq_ext + NI_next - DIV_next
    CASH_next = CASH_prev + FCF + delta_debt + delta_eq_ext - DIV_next

    return np.array(
        [
            S_next,
            C_next,
            SG_next,
            D_next,
            AR_next,
            INV_next,
            AP_next,
            PPE_next,
            CASH_next,
            DEBT_next,
            EQ_next,
            RE_next,
            TAX_next,
            INT_next,
            DIV_next,
        ],
        dtype=float,
    )


def inverse_perfect(y_prev: np.ndarray, y_curr: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Perfect driver inversion x_t* = g(y_{t-1}, y_t) following the plan."""
    (
        S_prev,
        C_prev,
        SG_prev,
        D_prev,
        AR_prev,
        INV_prev,
        AP_prev,
        PPE_prev,
        CASH_prev,
        DEBT_prev,
        EQ_prev,
        RE_prev,
        TAX_prev,
        INT_prev,
        DIV_prev,
    ) = y_prev
    (
        S_t,
        C_t,
        SG_t,
        D_t,
        AR_t,
        INV_t,
        AP_t,
        PPE_t,
        CASH_t,
        DEBT_t,
        EQ_t,
        RE_t,
        TAX_t,
        INT_t,
        DIV_t,
    ) = y_curr

    growth = (S_t - S_prev) / max(abs(S_prev), eps)
    gross_margin = 1.0 - C_t / max(abs(S_t), eps)
    sga_ratio = SG_t / max(abs(S_t), eps)
    dep_rate = D_t / max(abs(PPE_prev), eps)
    dso = 365.0 * AR_t / max(abs(S_t), eps)
    dio = 365.0 * INV_t / max(abs(C_t), eps)
    dpo = 365.0 * AP_t / max(abs(C_t), eps)

    capex = PPE_t - PPE_prev + D_t
    capex_ratio = capex / max(abs(S_t), eps)

    EBIT = S_t - C_t - SG_t - D_t
    EBT = EBIT - INT_t
    tax_rate = TAX_t / max(abs(EBT), eps)
    interest_rate = INT_t / max(abs(DEBT_prev), eps)
    net_income = EBT - TAX_t
    payout_ratio = DIV_t / max(abs(net_income), eps)

    delta_debt = DEBT_t - DEBT_prev
    ndebt_ratio = delta_debt / max(abs(S_t), eps)

    delta_eq_ext = EQ_t - EQ_prev - (net_income - DIV_t)
    nequity_ratio = delta_eq_ext / max(abs(S_t), eps)

    return np.array(
        [
            growth,
            gross_margin,
            sga_ratio,
            dep_rate,
            dso,
            dio,
            dpo,
            capex_ratio,
            tax_rate,
            interest_rate,
            payout_ratio,
            ndebt_ratio,
            nequity_ratio,
        ],
        dtype=float,
    )


def inverse_sequence(states: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Vectorized inverse_perfect over a full state trajectory."""
    if states.shape[0] < 2:
        return np.zeros((0, len(DRIVER_ORDER)), dtype=float)
    drivers = []
    for idx in range(1, states.shape[0]):
        drivers.append(inverse_perfect(states[idx - 1], states[idx], eps=eps))
    return np.vstack(drivers)
