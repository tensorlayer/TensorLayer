import tensorflow as tf
import math



def xavier_init(shape, uniform=True):
    """
    Set the parameter initialization using the method described.
    This method is designed to keep the scale of the gradients roughly the same
    in all layers.

    Parameters
    ----------
    n_inputs : int
        The number of units of the previous layer
    n_units : int
        The number of units of the current layer
    uniform : True, False
        If true use a uniform distribution, otherwise use a normal.

    Returns
    -------
    An initializer for 2D matrix.

    Reference
    ----------
    Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep
    feedforward neural networks. Proceedings of the 13th International Conference
    on Artificial Intelligence and Statistics (AISTATS), 9, 249–256. http://doi.org/10.1.1.207.2059
    """
    n_inputs, n_outputs = shape[0], shape[1]
    if uniform:
        # 6 was used in the paper.
        init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
        # return tf.random_uniform_initializer(-init_range, init_range)
        return tf.random_uniform(shape=[n_inputs, n_outputs], minval=-init_range, maxval=init_range, dtype=tf.float32, seed=None, name=None)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
        # return tf.truncated_normal_initializer(stddev=stddev)
    return tf.truncated_normal(shape=[n_input, n_outputs], mean=0.0, stddev=stddev, dtype=tf.float32, seed=None, name=None)
