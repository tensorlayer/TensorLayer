#! /usr/bin/python
# -*- coding: utf-8 -*-

from paddle.fluid.initializer import ConstantInitializer
from paddle.fluid.initializer import UniformInitializer
from paddle.fluid.initializer import NormalInitializer
from paddle.fluid.initializer import TruncatedNormalInitializer
from paddle.fluid.initializer import MSRAInitializer
import paddle

__all__ = [
    'Zeros', 'Ones', 'Constant', 'RandomUniform', 'RandomNormal', 'TruncatedNormal',
    'deconv2d_bilinear_upsampling_initializer', 'HeNormal'
]


class Zeros(ConstantInitializer):
    """Initializer that generates tensors initialized to 0.
    """

    def __init__(self):
        super(Zeros, self).__init__(value=0.0, force_cpu=False)


class Ones(object):
    """Initializer that generates tensors initialized to 1.
    """

    def __init__(self):
        # super(Ones, self).__init__(value=1.0, force_cpu=False)
        pass

    def __call__(self, shape, dtype):
        return paddle.ones(shape=shape, dtype=dtype)


class Constant(ConstantInitializer):
    """Initializer that generates tensors initialized to a constant value.

    Parameters
    ----------
    value : A python scalar or a numpy array.
        The assigned value.

    """

    def __init__(self, value=0.0):
        if value is None:
            raise ValueError("value must not be none.")
        super(Constant, self).__init__(value=value, force_cpu=False)
        self.value = value

    def get_config(self):
        return {"value": self.value}


class RandomUniform(UniformInitializer):
    """Initializer that generates tensors with a uniform distribution.

    Parameters
    ----------
    minval : A python scalar or a scalar tensor.
        Lower bound of the range of random values to generate.
    maxval : A python scalar or a scalar tensor.
        Upper bound of the range of random values to generate.
    seed : A Python integer.
        Used to seed the random generator.

    """

    def __init__(self, minval=-0.05, maxval=0.05, seed=0):
        assert minval is not None, 'low should not be None'
        assert maxval is not None, 'high should not be None'
        assert maxval >= minval, 'high should greater or equal than low'
        super(RandomUniform, self).__init__(
            low=minval, high=maxval, seed=seed, diag_num=0, diag_step=0, diag_val=1.0)
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def get_config(self):
        return {"minval": self.minval, "maxval": self.maxval, "seed": self.seed}


class RandomNormal(NormalInitializer):
    """Initializer that generates tensors with a normal distribution.

    Parameters
    ----------
    mean : A python scalar or a scalar tensor.
        Mean of the random values to generate.
    stddev : A python scalar or a scalar tensor.
        Standard deviation of the random values to generate.
    seed : A Python integer.
        Used to seed the random generator.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=0):
        assert mean is not None, 'mean should not be None'
        assert stddev is not None, 'std should not be None'
        super(RandomNormal, self).__init__(loc=mean, scale=stddev, seed=seed)
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


class TruncatedNormal(TruncatedNormalInitializer):
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
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=0):
        assert mean is not None, 'mean should not be None'
        assert stddev is not None, 'std should not be None'
        super(TruncatedNormal, self).__init__(loc=mean, scale=stddev, seed=seed)
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


class HeNormal(MSRAInitializer):
    """He normal initializer.

    Parameters
    ----------
    seed : A Python integer.
        Used to seed the random generator.

    """

    def __init__(self, seed=0):
        super(HeNormal, self).__init__(
            uniform=False, fan_in=None, seed=seed)
        self.seed = seed

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
    raise NotImplementedError
