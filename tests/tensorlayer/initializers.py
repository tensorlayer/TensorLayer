#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from tensorlayer.layers.core import LayersConfig

__all__ = ['deconv2d_bilinear_upsampling_initializer']


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

    Examples
    --------
    - Upsampling by a factor of 2, ie e.g 100->200
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> rescale_factor = 2
    >>> imsize = 128
    >>> num_channels = 3
    >>> filter_shape = (5, 5)
    >>> filter_size = (2 * rescale_factor - rescale_factor % 2) #Corresponding bilinear filter size
    >>> num_in_channels = 3
    >>> num_out_channels = 3
    >>> deconv_filter_shape = (filter_size, filter_size, num_out_channels, num_in_channels)
    >>> x = tf.placeholder(tf.float32, (1, imsize, imsize, num_channels))
    >>> net = tl.layers.InputLayer(x, name='input_layer')
    >>> bilinear_init = deconv2d_bilinear_upsampling_initializer(shape=filter_shape)
    >>> net = tl.layers.DeConv2dLayer(net,
    ...                    shape=filter_shape,
    ...                    output_shape=(1, imsize*rescale_factor, imsize*rescale_factor, num_out_channels),
    ...                    strides=(1, rescale_factor, rescale_factor, 1),
    ...                    W_init=bilinear_init,
    ...                    padding='SAME',
    ...                    act=None, name='g/h1/decon2d')

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
    weights = np.zeros((filter_size, filter_size, num_out_channels, num_in_channels))
    for i in range(num_out_channels):
        weights[:, :, i, i] = bilinear_kernel

    # assign numpy array to constant_initalizer and pass to get_variable
    return tf.constant_initializer(value=weights, dtype=LayersConfig.tf_dtype)
