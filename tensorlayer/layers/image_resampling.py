#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer

__all__ = [
    'UpSampling2d',
    'DownSampling2d',
]


class UpSampling2d(Layer):
    """The :class:`UpSampling2d` class is a up-sampling 2D layer.

    See `tf.image.resize_images <https://www.tensorflow.org/api_docs/python/tf/image/resize_images>`__.

    Parameters
    ----------
    scale : int/float or tuple of int/float
        (height, width) scale factor.
    method : str
        The resize method selected through the given string. Default 'bilinear'.
            - 'bilinear', Bilinear interpolation.
            - 'nearest', Nearest neighbor interpolation.
            - 'bicubic', Bicubic interpolation.
            - 'area', Area interpolation.
    antialias : boolean
        Whether to use an anti-aliasing filter when downsampling an image.
    data_format : str
        channels_last 'channel_last' (default) or channels_first.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> ni = tl.layers.Input([None, 50, 50, 32], name='input')
    >>> ni = tl.layers.UpSampling2d(scale=(2, 2))(ni)
    >>> output shape : [None, 100, 100, 32]

    """

    def __init__(
            self,
            scale,
            method='bilinear',
            antialias=False,
            data_format='channel_last',
            name=None,
    ):
        super(UpSampling2d, self).__init__(name)
        self.method = method
        self.antialias = antialias
        self.data_format = data_format

        logging.info(
            "UpSampling2d %s: scale: %s method: %s antialias: %s" % (self.name, scale, self.method, self.antialias)
        )

        self.build(None)
        self._built = True

        if isinstance(scale, (list, tuple)) and len(scale) != 2:
            raise ValueError("scale must be int or tuple/list of length 2")

        self.scale = (scale, scale) if isinstance(scale, int) else scale

    def __repr__(self):
        s = '{classname}(scale={scale}, method={method}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, scale=self.scale, method=self.method, name=self.name)

    def build(self, inputs_shape):
        if self.data_format != 'channel_last':
            raise Exception("UpSampling2d tf.image.resize_images only support channel_last")

    def forward(self, inputs):
        """

        Parameters
        ------------
        inputs : :class:`Tensor`
            Inputs tensors with 4-D Tensor of the shape (batch, height, width, channels)
        """
        output_size = [inputs.shape[1] * self.scale[0], inputs.shape[2] * self.scale[1]]
        outputs = tf.image.resize(inputs, size=output_size, method=self.method, antialias=self.antialias)
        return outputs


class DownSampling2d(Layer):
    """The :class:`DownSampling2d` class is down-sampling 2D layer.

    See `tf.image.resize_images <https://www.tensorflow.org/versions/master/api_docs/python/image/resizing#resize_images>`__.

    Parameters
    ----------
    scale : int/float or tuple of int/float
        (height, width) scale factor.
    method : str
        The resize method selected through the given string. Default 'bilinear'.
            - 'bilinear', Bilinear interpolation.
            - 'nearest', Nearest neighbor interpolation.
            - 'bicubic', Bicubic interpolation.
            - 'area', Area interpolation.
    antialias : boolean
        Whether to use an anti-aliasing filter when downsampling an image.
    data_format : str
        channels_last 'channel_last' (default) or channels_first.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> ni = tl.layers.Input([None, 50, 50, 32], name='input')
    >>> ni = tl.layers.DownSampling2d(scale=(2, 2))(ni)
    >>> output shape : [None, 25, 25, 32]

    """

    def __init__(
            self,
            scale,
            method='bilinear',
            antialias=False,
            data_format='channel_last',
            name=None,
    ):
        super(DownSampling2d, self).__init__(name)
        self.method = method
        self.antialias = antialias
        self.data_format = data_format

        logging.info(
            "DownSampling2d %s: scale: %s method: %s antialias: %s" % (self.name, scale, self.method, self.antialias)
        )

        self.build(None)
        self._built = True

        if isinstance(scale, (list, tuple)) and len(scale) != 2:
            raise ValueError("scale must be int or tuple/list of length 2")

        self.scale = (scale, scale) if isinstance(scale, int) else scale

    def __repr__(self):
        s = '{classname}(scale={scale}, method={method}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, scale=self.scale, method=self.method, name=self.name)

    def build(self, inputs_shape):
        if self.data_format != 'channel_last':
            raise Exception("DownSampling2d tf.image.resize_images only support channel_last")

    def forward(self, inputs):
        """

        Parameters
        ------------
        inputs : :class:`Tensor`
            Inputs tensors with 4-D Tensor of the shape (batch, height, width, channels)
        """
        output_size = [int(inputs.shape[1] * 1.0 / self.scale[0]), int(inputs.shape[2] * 1.0 / self.scale[1])]
        outputs = tf.image.resize(inputs, size=output_size, method=self.method, antialias=self.antialias)
        return outputs
