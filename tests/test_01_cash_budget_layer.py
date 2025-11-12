import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow is not installed in the current environment")

from src.model.cash_budget_layer import CashBudgetLayer
from src.model.metrics import identity_violation


def test_identity_violation_matches_layer_output():
    layer = CashBudgetLayer()
    states_prev = tf.constant([[10., 0., 100., 0., 50., 20., 30., 5., 10., 15.]], dtype=tf.float32)
    drivers = tf.constant([[200., 120., 40., 10., 12., 0.05, 0.04, 0.02, 5., 15., 0., 0., 0.25, 30., 45., 60.]], dtype=tf.float32)
    states_t, _ = layer((states_prev, drivers))
    metric_value = float(identity_violation(states_t).numpy())
    assert metric_value >= 0.0

    C, INV, K, BST, BLT, RE, PIC, AR, AP, INVST = tf.unstack(states_t[0])
    diff = float(tf.abs(C + INV + K + AR + INVST - (BST + BLT + AP + PIC + RE)).numpy())
    assert pytest.approx(diff, rel=1e-6, abs=1e-6) == metric_value


def test_identity_violation_zero_for_balanced_state():
    balanced = tf.constant(
        [[[10., 5., 20., 10., 15., 7., 3., 3., 5., 2.]]],
        dtype=tf.float32,
    )
    assert float(identity_violation(balanced).numpy()) == pytest.approx(0.0)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
