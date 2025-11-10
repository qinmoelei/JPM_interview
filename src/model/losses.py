from __future__ import annotations
import tensorflow as tf

def identity_penalty(states_t: tf.Tensor, bs_targets: tf.Tensor, weight: float=1000.0) -> tf.Tensor:
    """Penalize |Assets - (Liabilities + Equity)| at each time step.
       If bs_targets includes Total Assets, Total Liab, Total Equity columns, use them;
       otherwise reconstruct from states_t minimal stack: A=C+INV+K+AR+INVST, L=BST+BLT+AP, E=PIC+RE.
    """
    C, INV, K, BST, BLT, RE, PIC, AR, AP, INVST = tf.unstack(states_t, axis=-1)
    A = C + INV + K + AR + INVST
    L = BST + BLT + AP
    E = PIC + RE
    diff = tf.abs(A - (L + E))
    return weight * tf.reduce_mean(diff)

def mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(y_true - y_pred))
