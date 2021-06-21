#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorlayer as tl

__all__ = [
    'Initializer', 'Zeros', 'Ones', 'Constant', 'RandomUniform', 'RandomNormal', 'TruncatedNormal',
    'deconv2d_bilinear_upsampling_initializer', 'HeNormal'
]


class Initializer(object):
    """Initializer base class: all initializers inherit from this class.
    """

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Parameters
        ----------
        shape : tuple of int.
            The shape of the tensor.
        dtype : Optional dtype of the tensor.
            If not provided will return tensor of `tl.float32`.

        Returns
        -------

        """
        raise NotImplementedError

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.

        Returns
        -------
            A JSON-serializable Python dict.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """Instantiates an initializer from a configuration dictionary.

        Parameters
        ----------
        config : A python dictionary.
            It will typically be the output of `get_config`.

        Returns
        -------
            An Initializer instance.
        """
        if 'dtype' in config:
            config.pop('dtype')
        return cls(**config)


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.

    Examples
    --------

    >>> import tensorlayer as tl
    >>> init = tl.initializers.zeros()
    >>> print(init(shape=(5, 10), dtype=tl.float32))

    """

    def __call__(self, shape, dtype=tl.float32):
        return tl.zeros(shape, dtype=dtype)


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.

    Examples
    --------

    >>> import tensorlayer as tl
    >>> init = tl.initializers.ones()
    >>> print(init(shape=(5, 10), dtype=tl.float32))

    """

    def __call__(self, shape, dtype=tl.float32):
        return tl.ones(shape, dtype=dtype)


class Constant(Initializer):
    """Initializer that generates tensors initialized to a constant value.

    Parameters
    ----------
    value : A python scalar or a numpy array.
        The assigned value.

    Examples
    --------

    >>> import tensorlayer as tl
    >>> init = tl.initializers.constant(value=10)
    >>> print(init(shape=(5, 10), dtype=tl.float32))

    """

    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=tl.float32):
        return tl.constant(self.value, shape=shape, dtype=dtype)

    def get_config(self):
        return {"value": self.value}


class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.

    Parameters
    ----------
    minval : A python scalar or a scalar tensor.
        Lower bound of the range of random values to generate.
    maxval : A python scalar or a scalar tensor.
        Upper bound of the range of random values to generate.
    seed : A Python integer.
        Used to seed the random generator.

    Examples
    --------

    >>> import tensorlayer as tl
    >>> init = tl.initializers.random_uniform(minval=-0.05, maxval=0.05)
    >>> print(init(shape=(5, 10), dtype=tl.float32))

    """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=tl.float32):
        return tl.random_uniform(shape, self.minval, self.maxval, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {"minval": self.minval, "maxval": self.maxval, "seed": self.seed}


class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.

    Parameters
    ----------
    mean : A python scalar or a scalar tensor.
        Mean of the random values to generate.
    stddev : A python scalar or a scalar tensor.
        Standard deviation of the random values to generate.
    seed : A Python integer.
        Used to seed the random generator.

    minval=-0.05, maxval=0.05

    Examples
    --------

    >>> import tensorlayer as tl
    >>> init = tl.initializers.random_normal(mean=0.0, stddev=0.05)
    >>> print(init(shape=(5, 10), dtype=tl.float32))

    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=tl.float32):
        return tl.random_normal(shape, self.mean, self.stddev, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.

    These values are similar to values from a `RandomNormal`
    except that values more than two standard deviations from the mean
    are discarded and re-drawn. This is the recommended initializer for
    neural network weights and filters.


    Parameters
    ----------
    mean : A python scalar or a scalar tensor.
        Mean of the random values to generate.
    stddev : A python scalar or a scalar tensor.
        Standard deviation of the andom values to generate.
    seed : A Python integer.
        Used to seed the random generator.

    Examples
    --------

    >>> import tensorlayer as tl
    >>> init = tl.initializers.truncated_normal(mean=0.0, stddev=0.05)
    >>> print(init(shape=(5, 10), dtype=tl.float32))

    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=tl.float32):
        return tl.truncated_normal(shape, self.mean, self.stddev, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


class HeNormal(Initializer):
    """He normal initializer.

    Parameters
    ----------
    seed : A Python integer.
        Used to seed the random generator.

    Examples
    --------

    >>> import tensorlayer as tl
    >>> init = tl.initializers.he_normal()
    >>> print(init(shape=(5, 10), dtype=tl.float32))

    """

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape, dtype=tl.float32):
        return tl.he_normal(seed=self.seed, shape=shape, dtype=dtype)

    def get_config(self):
        return {"seed", self.seed}


def deconv2d_bilinear_upsampling_initializer(shape):
    """Returns the initializer that can be passed to DeConv2dLayer for initializing the
    weights in correspondence to channel-wise bilinear up-sampling.
    Used in segmentation approaches such as [FCN](https://arxiv.org/abs/1605.06211)

    Parameters
    ----------
    shape : tuple of int
        The shape of the filters, [height, width, output_channels, in_channels].
        It must match the shape passed to DeConv2dLayer.

    Returns
    -------
    ``tf.constant_initializer``
        A constant initializer with weights set to correspond to per channel bilinear upsampling
        when passed as W_int in DeConv2dLayer

    """
    if shape[0] != shape[1]:
        raise Exception('deconv2d_bilinear_upsampling_initializer only supports symmetrical filter sizes')

    if shape[3] < shape[2]:
        raise Exception(
            'deconv2d_bilinear_upsampling_initializer behaviour is not defined for num_in_channels < num_out_channels '
        )

    filter_size = shape[0]
    num_out_channels = shape[2]
    num_in_channels = shape[3]

    # Create bilinear filter kernel as numpy array
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_out_channels, num_in_channels), dtype=np.float32)
    for i in range(num_out_channels):
        weights[:, :, i, i] = bilinear_kernel

    # assign numpy array to constant_initalizer and pass to get_variable
    return Constant(value=weights)
