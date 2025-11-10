from __future__ import annotations
import tensorflow as tf

def mape(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float=1e-6) -> tf.Tensor:
    return 100.0 * tf.reduce_mean(tf.abs((y_true - y_pred) / (tf.abs(y_true) + eps)))

def identity_violation(states_t: tf.Tensor) -> tf.Tensor:
    C, INV, K, BST, BLT, RE, PIC, AR, AP, INVST = tf.unstack(states_t, axis=-1)
    A = C + INV + K + AR + INVST
    L = BST + BLT + AP
    E = PIC + RE
    return tf.reduce_mean(tf.abs(A - (L + E)))
