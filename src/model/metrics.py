from __future__ import annotations
import tensorflow as tf


def mape(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """Mean absolute percentage error (safe when y_true≈0 thanks to eps).

    Example:
        >>> float(mape(tf.constant([1.0, 2.0]), tf.constant([1.0, 1.0])))
        25.0
    """
    return 100.0 * tf.reduce_mean(tf.abs((y_true - y_pred) / (tf.abs(y_true) + eps)))


def identity_violation(states_t: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """Relative |Assets - (Liab+Equity)| averaged across samples.

    Example:
        >>> s = tf.ones((1, 12))
        >>> float(identity_violation(s))
        0.0
    """
    (C,
     INV,
     K,
     BST,
     BLT,
     RE,
     PIC,
     AR,
     AP,
     INVST,
     OTHER_A,
     OTHER_LE) = tf.unstack(states_t, axis=-1)
    assets = C + INV + K + AR + INVST + OTHER_A
    liab_equity = BST + BLT + AP + PIC + RE + OTHER_LE
    diff = tf.abs(assets - liab_equity)
    return tf.reduce_mean(diff / (tf.abs(assets) + eps))
