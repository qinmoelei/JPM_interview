import numpy as np

from src.experiments.driver_workflow import evaluate_driver_predictions


def test_evaluate_driver_predictions_basic():
    preds = {"AAA": {0: np.array([1.0, 2.0]), 1: np.array([2.0, 4.0])}}
    targets = {"AAA": {0: np.array([1.0, 2.0]), 1: np.array([1.0, 3.0])}}
    metrics = evaluate_driver_predictions(preds, targets)

    # diff arrays: [0,0] and [1,1] → squared mean = (0+0+1+1)/4 = 0.5
    assert abs(metrics["mse"] - 0.5) < 1e-8
    assert abs(metrics["mae"] - 0.5) < 1e-8
    # rel_l1 for second row: (1/1,1/3); rel_l1 averages elementwise over all entries.
    assert abs(metrics["rel_l1"] - (1 + 1 / 3) / 2 / 2) < 1e-8
