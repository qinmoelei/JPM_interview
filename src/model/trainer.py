from __future__ import annotations
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple

from .cash_budget_layer import CashBudgetLayer
from .losses import identity_penalty, mse_loss
from .metrics import mape, identity_violation

class VPForecaster(tf.keras.Model):
    """Driver nets + deterministic CashBudgetLayer"""

    def __init__(self, cov_dim: int, driver_dim: int = 16, state_dim: int = 10, hidden: int = 16):
        super().__init__()
        self.driver_dim = driver_dim
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
        states_seq, covariates_seq = inputs  # shapes: [B,T,state_dim], [B,T,driver_dim]
        drivers_seq = self.driver_net(covariates_seq, training=training)
        outputs_states = []
        outputs_is = []
        s_prev = states_seq[:,0,:]  # seed with first observed state
        for t in range(1, states_seq.shape[1]):
            drivers_t = drivers_seq[:,t,:]
            s_prev, is_out = self.cb((s_prev, drivers_t))
            outputs_states.append(s_prev)
            outputs_is.append(is_out)
        states_pred = tf.stack(outputs_states, axis=1)  # [B, T-1, state_dim]
        is_pred     = tf.stack(outputs_is, axis=1)      # [B, T-1, 8]
        return states_pred, is_pred

def training_step(model: VPForecaster,
                  batch: Dict[str, np.ndarray],
                  optimizer: tf.keras.optimizers.Optimizer,
                  w_identity: float = 1000.0):
    states_seq = tf.convert_to_tensor(batch["states"], dtype=tf.float32)
    covs_seq   = tf.convert_to_tensor(batch["covs"], dtype=tf.float32)
    target_seq = tf.convert_to_tensor(batch["target_states"], dtype=tf.float32)
    with tf.GradientTape() as tape:
        pred_states, _ = model((states_seq, covs_seq), training=True)
        # Align targets to prediction horizon (T-1)
        loss_fit = mse_loss(target_seq[:, -pred_states.shape[1]:, :], pred_states)
        loss_id  = identity_violation(pred_states) * w_identity
        loss = loss_fit + loss_id
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    mae = tf.reduce_mean(tf.abs(pred_states - target_seq[:, -pred_states.shape[1]:, :]))
    iden = identity_violation(pred_states)
    return {
        "loss": float(loss.numpy()),
        "loss_fit": float(loss_fit.numpy()),
        "loss_id": float(loss_id.numpy()),
        "mae_states": float(mae.numpy()),
        "identity_violation": float(iden.numpy()),
    }
