import numpy as np
import tensorflow as tf
from src.model.cash_budget_layer import CashBudgetLayer

def test_identity_holds():
    layer = CashBudgetLayer()
    # Simple single-sample test
    states_prev = tf.constant([[10., 0., 100., 0., 50., 20., 30., 5., 10., 15.]], dtype=tf.float32)
    # drivers: Sales, COGS, Opex, Dep, Capex, rST, rLT, rINV, AmortLT, Cbar, EI, Div, tau, DSO, DPO, DIO
    drivers = tf.constant([[200., 120., 40., 10., 12., 0.05, 0.04, 0.02, 5., 15., 0., 0., 0.25, 30., 45., 60.]], dtype=tf.float32)
    s_t, _ = layer((states_prev, drivers))
    C, INV, K, BST, BLT, RE, PIC, AR, AP, INVST = tf.unstack(s_t[0])
    A = C + INV + K + AR + INVST
    L = BST + BLT + AP
    E = PIC + RE
    diff = tf.abs(A - (L + E)).numpy()
    assert diff < 1e-4, f"Identity violated: {diff}"
