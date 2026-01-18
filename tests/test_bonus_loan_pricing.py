import numpy as np

from src.llm.loan_pricing import conformal_interval, evaluate_interval


def test_conformal_interval_and_coverage():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    q = conformal_interval(y_true, y_pred, alpha=0.25)
    assert np.isclose(q, 0.2)
    metrics = evaluate_interval(y_true, y_pred, q)
    assert metrics["coverage"] == 1.0
    assert metrics["avg_width"] > 0
