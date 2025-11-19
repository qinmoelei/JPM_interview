from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple

def build_sequences(features: pd.DataFrame,
                    targets: pd.DataFrame,
                    window: int = 6,
                    horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build rolling window sequences X (features) and Y (targets) for supervised learning.

    Args:
        features: DataFrame indexed by period-end (datetime) with driver columns.
        targets: DataFrame indexed by period-end with target columns (e.g., next BS states).
        window: Number of historical steps per sample.
        horizon: Number of prediction steps ahead.

    Returns:
        Tuple of numpy arrays with shapes `[N, window, F]` and `[N, horizon, T]`.

    Example:
        >>> idx = pd.to_datetime(["2020", "2021", "2022", "2023"])
        >>> feats = pd.DataFrame({"capex": [1, 2, 3, 4]}, index=idx)
        >>> targs = pd.DataFrame({"cash": [10, 11, 12, 13]}, index=idx)
        >>> X, Y = build_sequences(feats, targs, window=2, horizon=1)
        >>> X.shape, Y.shape
        ((2, 2, 1), (2, 1, 1))
    """
    # Align indices and sort
    idx = features.index.intersection(targets.index).sort_values()
    F = features.loc[idx].to_numpy(dtype=float)
    T = targets.loc[idx].to_numpy(dtype=float)
    X_list, Y_list = [], []
    for i in range(len(idx) - window - horizon + 1):
        X_list.append(F[i:i+window, :])
        Y_list.append(T[i+window:i+window+horizon, :])
    if not X_list:
        return np.empty((0, window, F.shape[1])), np.empty((0, horizon, T.shape[1]))
    return np.stack(X_list, axis=0), np.stack(Y_list, axis=0)
