from __future__ import annotations
import pandas as pd
import numpy as np

from src.datahandler.dataset import build_sequences


def test_build_sequences_returns_expected_shapes():
    idx = pd.date_range("2020-01-01", periods=7, freq="Y")
    features = pd.DataFrame({"driver": np.arange(len(idx))}, index=idx)
    targets = pd.DataFrame({"state": np.arange(len(idx)) * 10}, index=idx)

    X, Y = build_sequences(features, targets, window=3, horizon=2)

    assert X.shape == (3, 3, 1)  # 7 - 3 - 2 + 1 = 3 samples
    assert Y.shape == (3, 2, 1)
    assert np.array_equal(X[0, :, 0], np.array([0, 1, 2]))
    assert np.array_equal(Y[0, :, 0], np.array([30, 40]))


def test_build_sequences_returns_empty_when_not_enough_history():
    idx = pd.date_range("2020-01-01", periods=3, freq="Y")
    features = pd.DataFrame({"driver": [1, 2, 3]}, index=idx)
    targets = pd.DataFrame({"state": [1, 2, 3]}, index=idx)

    X, Y = build_sequences(features, targets, window=4, horizon=2)

    assert X.shape[0] == 0
    assert Y.shape[0] == 0
