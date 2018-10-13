#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'UpSampling2d',
    'DownSampling2d',
]


class UpSampling2d(Layer):
    """The :class:`UpSampling2d` class is a up-sampling 2D layer, see `tf.image.resize_images
    <https://www.tensorflow.org/api_docs/python/tf/image/resize_images>`__.

    Parameters
    ----------
    size : tuple of int/float
        (height, width) scale factor or new size of height and width.
    is_scale : boolean
        If True (default), the `size` is a scale factor; otherwise, the `size` is the numbers of pixels of height and width.
    method : int
        The resize method selected through the index. Defaults index is 0 which is ResizeMethod.BILINEAR.
            - Index 0 is ResizeMethod.BILINEAR, Bilinear interpolation.
            - Index 1 is ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
            - Index 2 is ResizeMethod.BICUBIC, Bicubic interpolation.
            - Index 3 ResizeMethod.AREA, Area interpolation.
    align_corners : boolean
        If True, align the corners of the input and output. Default is False.
    name : str
        A unique layer name.

    """

    def __init__(
        self,
        size,
        is_scale=True,
        method=0,
        align_corners=False,
        name='upsample2d_layer',
    ):

        if not isinstance(size, (list, tuple)):
            raise AssertionError("`size` argument should be a `list` or a `tuple`")

        if len(size) != 2:
            raise AssertionError("`size` argument should be of length 2")

        self.size = size
        self.is_scale = is_scale
        self.method = method
        self.align_corners = align_corners
        self.name = name

        super(UpSampling2d, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("is_scale: %s" % self.is_scale)
        except AttributeError:
            pass

        try:
            additional_str.append("size: {}".format(self.size))
        except AttributeError:
            pass

        try:
            additional_str.append("method: %s" % self.method)
        except AttributeError:
            pass

        try:
            additional_str.append("align_corners: %s" % self.align_corners)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        if len(self._temp_data['inputs'].shape) == 3:
            x_pos, y_pos = (0, 1)

        elif len(self._temp_data['inputs'].shape) == 4:
            x_pos, y_pos = (1, 2)

        else:
            raise RuntimeError("The input shape: %s is not supported" % tf.shape(self._temp_data['inputs']))

        with tf.variable_scope(self.name):

            if self.is_scale:
                if all(isinstance(x, int) for x in self.size):
                    if None not in [
                        self._temp_data['inputs'].get_shape()[x_pos]._value,
                        self._temp_data['inputs'].get_shape()[y_pos]._value
                    ]:
                        size_h = self._temp_data['inputs'].get_shape()[x_pos] * self.size[0]
                        size_w = self._temp_data['inputs'].get_shape()[y_pos] * self.size[1]
                    else:
                        size_h = tf.shape(self._temp_data['inputs'])[x_pos] * self.size[0]
                        size_w = tf.shape(self._temp_data['inputs'])[y_pos] * self.size[1]

                    _size = [size_h, size_w]

                else:
                    raise ValueError("all elements of tuple `size` hyperparameter should of type `int`")

            else:
                _size = self.size

            self._temp_data['outputs'] = tf.image.resize_images(
                self._temp_data['inputs'], size=_size, method=self.method, align_corners=self.align_corners
            )
            self._temp_data['outputs'] = tf.cast(self._temp_data['outputs'], self._temp_data['inputs'].dtype)


class DownSampling2d(Layer):
    """The :class:`DownSampling2d` class is down-sampling 2D layer, see `tf.image.resize_images
    <https://www.tensorflow.org/versions/master/api_docs/python/image/resizing#resize_images>`__.

    Parameters
    size : tuple of int/float
        (height, width) scale factor or new size of height and width.
    is_scale : boolean
        If True (default), the `size` is the scale factor; otherwise, the `size` are numbers of pixels of height and width.
    method : int
        The resize method selected through the index. Defaults index is 0 which is ResizeMethod.BILINEAR.
            - Index 0 is ResizeMethod.BILINEAR, Bilinear interpolation.
            - Index 1 is ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
            - Index 2 is ResizeMethod.BICUBIC, Bicubic interpolation.
            - Index 3 ResizeMethod.AREA, Area interpolation.
    align_corners : boolean
        If True, exactly align all 4 corners of the input and output. Default is False.
    name : str
        A unique layer name.

    """

    def __init__(
        self,
        size,
        is_scale=True,
        method=0,
        align_corners=False,
        name='downsample2d_layer',
    ):

        if not isinstance(size, (list, tuple)):
            raise AssertionError("`size` argument should be a `list` or a `tuple`")

        if len(size) != 2:
            raise AssertionError("`size` argument should be of length 2")

        self.size = size
        self.is_scale = is_scale
        self.method = method
        self.align_corners = align_corners
        self.name = name

        super(DownSampling2d, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("is_scale: %s" % self.is_scale)
        except AttributeError:
            pass

        try:
            additional_str.append("size: {}".format(self.size))
        except AttributeError:
            pass

        try:
            additional_str.append("method: %s" % self.method)
        except AttributeError:
            pass

        try:
            additional_str.append("align_corners: %s" % self.align_corners)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        if len(self._temp_data['inputs'].shape) == 3:
            x_pos, y_pos = (0, 1)

        elif len(self._temp_data['inputs'].shape) == 4:
            x_pos, y_pos = (1, 2)

        else:
            raise RuntimeError("The input shape: %s is not supported" % tf.shape(self._temp_data['inputs']))

        with tf.variable_scope(self.name):

            if self.is_scale:
                if all(isinstance(x, int) for x in self.size):
                    if None not in [
                        self._temp_data['inputs'].get_shape()[x_pos]._value,
                        self._temp_data['inputs'].get_shape()[y_pos]._value
                    ]:

                        size_h = np.ceil(int(self._temp_data['inputs'].get_shape()[x_pos]) / float(self.size[0])
                                        ).astype(np.int32)
                        size_w = np.ceil(int(self._temp_data['inputs'].get_shape()[y_pos]) / float(self.size[1])
                                        ).astype(np.int32)

                    else:
                        size_h = np.ceil(tf.shape(self._temp_data['inputs'])[x_pos] / np.float32(self.size[0])
                                        ).astype(np.int32)
                        size_w = np.ceil(tf.shape(self._temp_data['inputs'])[y_pos] / np.float32(self.size[1])
                                        ).astype(np.int32)

                elif all(isinstance(x, float) for x in self.size):
                    if None not in [
                        self._temp_data['inputs'].get_shape()[x_pos]._value,
                        self._temp_data['inputs'].get_shape()[y_pos]._value
                    ]:
                        size_h = np.ceil(int(self._temp_data['inputs'].get_shape()[x_pos]) * self.size[0]
                                        ).astype(np.int32)
                        size_w = np.ceil(int(self._temp_data['inputs'].get_shape()[y_pos]) * self.size[1]
                                        ).astype(np.int32)
                    else:
                        size_h = np.ceil(tf.shape(self._temp_data['inputs'])[x_pos] * self.size[0]).astype(np.int32)
                        size_w = np.ceil(tf.shape(self._temp_data['inputs'])[y_pos] * self.size[1]).astype(np.int32)

                else:
                    raise ValueError("all elements of tuple `size` hyperparameter should be either `int` or `float`")

                _size = [size_h, size_w]

            else:
                _size = self.size

            self._temp_data['outputs'] = tf.image.resize_images(
                self._temp_data['inputs'], size=_size, method=self.method, align_corners=self.align_corners
            )
            self._temp_data['outputs'] = tf.cast(self._temp_data['outputs'], self._temp_data['inputs'].dtype)
