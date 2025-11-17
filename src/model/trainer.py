from __future__ import annotations
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, Optional

from .cash_budget_layer import CashBudgetLayer
from .losses import relative_identity_penalty, mse_loss
from .metrics import identity_violation

class VPForecaster(tf.keras.Model):
    """Driver nets + deterministic CashBudgetLayer"""

    def __init__(self, cov_dim: int, driver_dim: int = 16, state_dim: int = 10, hidden: int = 16):
        super().__init__()
        self.driver_dim = driver_dim
        hidden = max(1, min(hidden, driver_dim, state_dim, cov_dim))
        # Lightweight sequence model for drivers (LayerNorm + SimpleRNN)
        self.driver_net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, cov_dim)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.SimpleRNN(hidden, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden, activation="tanh")),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(driver_dim)),
        ])
        self.cb = CashBudgetLayer()

    def call(self, inputs, training=False):
        """Forward pass: infer driver adjustments then roll CashBudgetLayer."""
        states_seq, covariates_seq = inputs  # shapes: [B,T,state_dim], [B,T,driver_dim]
        drivers_seq = self.driver_net(covariates_seq, training=training)
        outputs_states = []
        outputs_is = []
        s_prev = states_seq[:, 0, :]  # seed with first observed state
        for t in range(1, states_seq.shape[1]):
            drivers_t = drivers_seq[:, t, :]
            s_prev, is_out = self.cb((s_prev, drivers_t))
            outputs_states.append(s_prev)
            outputs_is.append(is_out)
        states_pred = tf.stack(outputs_states, axis=1)  # [B, T-1, state_dim]
        is_pred = tf.stack(outputs_is, axis=1)  # [B, T-1, 8]
        return states_pred, is_pred


def training_step(model: VPForecaster,
                  batch: Dict[str, np.ndarray],
                  optimizer: tf.keras.optimizers.Optimizer,
                  w_identity: float = 5.0,
                  growth_weight: float = 1.0,
                  level_weight: float = 0.3,
                  state_weights: Optional[np.ndarray] = None,
                  eps: float = 1e-6):
    """One full-batch training step with growth/level/id penalties and masks."""
    states_seq = tf.convert_to_tensor(batch["states"], dtype=tf.float32)
    covs_seq = tf.convert_to_tensor(batch["covs"], dtype=tf.float32)
    target_seq = tf.convert_to_tensor(batch["target_states"], dtype=tf.float32)
    state_shift = tf.convert_to_tensor(batch["state_shift"], dtype=tf.float32)[:, None, :]
    state_scale = tf.convert_to_tensor(batch["state_scale"], dtype=tf.float32)[:, None, :]
    mask_tensor = tf.convert_to_tensor(batch["transition_mask"], dtype=tf.float32)
    per_state_weights = None
    if state_weights is not None:
        per_state_weights = tf.convert_to_tensor(state_weights, dtype=tf.float32)
        if per_state_weights.ndim == 1:
            per_state_weights = tf.reshape(per_state_weights, (1, 1, -1))

    with tf.GradientTape() as tape:
        pred_states, _ = model((states_seq, covs_seq), training=True)
        pred_len = pred_states.shape[1]
        true_curr = target_seq[:, -pred_len:, :]
        true_prev = target_seq[:, -(pred_len + 1):-1, :]
        true_curr_raw = true_curr * state_scale + state_shift
        true_prev_raw = true_prev * state_scale + state_shift
        pred_raw = pred_states * state_scale + state_shift
        mask_slice = mask_tensor[:, -pred_len:]
        mask_expanded = mask_slice[..., None]
        if per_state_weights is not None:
            weights_tensor = mask_expanded * per_state_weights
        else:
            weights_tensor = mask_expanded
        eps_tensor = tf.constant(eps, dtype=pred_raw.dtype)
        denom = tf.where(tf.abs(true_prev_raw) < eps_tensor, eps_tensor, true_prev_raw)
        g_true = true_curr_raw / denom - 1.0
        g_pred = pred_raw / denom - 1.0
        loss_growth = mse_loss(g_true, g_pred, weights=weights_tensor)
        loss_level = mse_loss(true_curr_raw, pred_raw, weights=weights_tensor)
        mask_bool = mask_slice > 0.0
        pred_for_identity = tf.boolean_mask(pred_raw, mask_bool)
        if tf.size(pred_for_identity) > 0:
            loss_id = relative_identity_penalty(pred_for_identity) * w_identity
        else:
            loss_id = tf.constant(0.0, dtype=pred_raw.dtype)
        loss = growth_weight * loss_growth + level_weight * loss_level + loss_id

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    denom_count = tf.maximum(tf.reduce_sum(mask_slice), 1.0) * tf.cast(tf.shape(pred_raw)[-1], pred_raw.dtype)
    abs_state_err = tf.abs(pred_raw - true_curr_raw)
    abs_growth_err = tf.abs(g_pred - g_true)
    mae_level = tf.reduce_sum(abs_state_err * mask_expanded) / denom_count
    mae_growth = tf.reduce_sum(abs_growth_err * mask_expanded) / denom_count
    if tf.size(pred_for_identity) > 0:
        iden = identity_violation(pred_for_identity)
    else:
        iden = tf.constant(0.0, dtype=pred_raw.dtype)

    return {
        "loss": float(loss.numpy()),
        "loss_growth": float(loss_growth.numpy()),
        "loss_level": float(loss_level.numpy()),
        "loss_id": float(loss_id.numpy()),
        "mae_states": float(mae_level.numpy()),
        "mae_growth": float(mae_growth.numpy()),
        "identity_violation": float(iden.numpy()),
    }
