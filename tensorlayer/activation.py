import tensorflow as tf

def identity(x):
    """The identity activation function.

    Parameters
    ----------
    x : input(s)
    """
    return x

def ramp(x, v_min=0, v_max=1, name=None):
    """The ramp activation function.

    Parameters
    ----------
    x : a tensor outputs
        input(s)
    v_min : float
        if input(s) smaller than v_min, change inputs to v_min
    v_max : float
        if input(s) greater than v_max, change inputs to v_max
    name : a string or None
        An optional name to attach to this activation function.
    """
    return tf.clip_by_value(x, clip_value_min=v_min, clip_value_max=v_max, name=name)
