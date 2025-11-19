from __future__ import annotations
import tensorflow as tf

def relative_identity_penalty(states_t: tf.Tensor,
                              clip_value: float = 0.1,
                              eps: float = 1e-6) -> tf.Tensor:
    """Soft penalty on |A - (L+E)| / |A| with clipping to avoid dominance.

    Example:
        >>> s = tf.ones((2, 12))
        >>> float(relative_identity_penalty(s))
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
    rel = diff / (tf.abs(assets) + eps)
    rel = tf.clip_by_value(rel, 0.0, clip_value)
    return tf.reduce_mean(rel)

def mse_loss(y_true: tf.Tensor,
             y_pred: tf.Tensor,
             weights: tf.Tensor | None = None,
             eps: float = 1e-9) -> tf.Tensor:
    """Mean-squared error with optional weights (e.g., transition masks * state weights).

    Example:
        >>> mse_loss(tf.constant([1.0, 2.0]), tf.constant([1.0, 1.5])).numpy()
        0.125
    """
    err = tf.square(y_true - y_pred)
    if weights is not None:
        err = err * weights
        denom = tf.reduce_sum(weights)
        return tf.reduce_sum(err) / tf.maximum(denom, eps)
    return tf.reduce_mean(err)
