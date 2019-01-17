#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'UpSampling2d',
    'DownSampling2d',
]


class UpSampling2d(Layer):
    """The :class:`UpSampling2d` class is a up-sampling 2D layer.

    See `tf.image.resize_images <https://www.tensorflow.org/api_docs/python/tf/image/resize_images>`__.

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
    data_format : str
        channels_last 'channel_last' (default) or channels_first.
    name : None or str
        A unique layer name.
    """

    def __init__(
            self,
            size,
            is_scale=True,
            method=0,
            align_corners=False,
            data_format='channel_last',
            name=None,  #'upsample2d',
    ):
        # super(UpSampling2d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.size = size
        self.is_scale = scale
        self.method = method
        self.align_corners = align_corners
        self.data_format = data_format

        logging.info(
            "UpSampling2d %s: is_scale: %s size: %s method: %d align_corners: %s" %
            (self.name, self.is_scale, self.size, self.method, self.align_corners)
        )

        if not isinstance(self.size, (list, tuple)) and len(self.size) == 2:
            raise AssertionError()

    def build(self, inputs_shape):
        if self.data_format != 'channel_last':
            raise Exception("UpSampling2d tf.image.resize_images only support channel_last")

        # if len(self.inputs.get_shape()) == 3:
        if len(inputs_shape) == 3:
            if self.is_scale:
                # input_shape = inputs.shape.as_list()
                if input_shape[0] is not None:
                    size_h = self.size[0] * input_shape[0]
                else:
                    size_h = self.size[0] * tf.shape(input=inputs)[0]
                if input_shape[1] is not None:
                    size_w = self.size[1] * input_shape[1]
                else:
                    size_w = self.size[1] * tf.shape(input=inputs)[1]
                self.size = [size_h, size_w]

        # elif len(self.inputs.get_shape()) == 4:
        elif len(inputs_shape) == 4:
            if self.is_scale:
                # input_shape = inputs.shape.as_list()
                if input_shape[1] is not None:
                    size_h = self.size[0] * input_shape[1]
                else:
                    size_h = self.size[0] * tf.shape(input=inputs)[1]
                if input_shape[2] is not None:
                    size_w = self.size[1] * input_shape[2]
                else:
                    size_w = self.size[1] * tf.shape(input=inputs)[2]
                self.size = [size_h, size_w]

        else:
            raise Exception("Donot support shape %s" % str(inputs.shape.as_list()))

    def forward(self, inputs):
        """

        Parameters
        ------------
        prev_layer : :class:`Layer`
            Previous layer with 4-D Tensor of the shape (batch, height, width, channels) or 3-D Tensor of the shape (height, width, channels).
        """
        outputs = tf.image.resize(inputs, size=self.size, method=self.method, align_corners=self.align_corners)
        return outputs


class DownSampling2d(Layer):
    """The :class:`DownSampling2d` class is down-sampling 2D layer.

    See `tf.image.resize_images <https://www.tensorflow.org/versions/master/api_docs/python/image/resizing#resize_images>`__.

    Parameters
    ----------
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
    data_format : str
        channels_last 'channel_last' (default) or channels_first.
    name : None or str
        A unique layer name.
    """

    def __init__(
            self,
            size,
            is_scale=True,
            method=0,
            align_corners=False,
            data_format='channel_last',
            name='downsample2d',
    ):
        # super(DownSampling2d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.size = size
        self.is_scale = scale
        self.method = method
        self.align_corners = align_corners
        self.data_format = data_format

        logging.info(
            "DownSampling2d %s: is_scale: %s size: %s method: %d, align_corners: %s" %
            (self.name, self.is_scale, self.size, self.method, self.align_corners)
        )

        if not isinstance(self.size, (list, tuple)) and len(self.size) == 2:
            raise AssertionError()

    def build(self, inputs_shape):
        if self.data_format != 'channel_last':
            raise Exception("DownSampling2d tf.image.resize_images only support channel_last")

        if len(inputs_shape) == 3:
            # if inputs.shape.ndims == 3:
            if self.is_scale:
                # input_shape = inputs.shape.as_list()
                if input_shape[1] is not None:
                    size_h = self.size[0] * input_shape[0]
                else:
                    size_h = self.size[0] * tf.shape(input=inputs)[0]
                if input_shape[1] is not None:
                    size_w = self.size[1] * input_shape[1]
                else:
                    size_w = self.size[1] * tf.shape(input=inputs)[1]
                self.size = [size_h, size_w]

        elif len(inputs_shape) == 4:
            # elif inputs.shape.ndims == 4:
            if self.is_scale:
                # input_shape = inputs.shape.as_list()
                if input_shape[1] is not None:
                    size_h = self.size[0] * input_shape[1]
                else:
                    size_h = self.size[0] * tf.shape(input=inputs)[1]
                if input_shape[2] is not None:
                    size_w = self.size[1] * input_shape[2]
                else:
                    size_w = self.size[1] * tf.shape(input=inputs)[2]
                self.size = [size_h, size_w]

        else:
            raise Exception("Donot support shape %s" % str(inputs.shape.as_list()))

    def forward(self, inputs):
        """

        Parameters
        ------------
        prev_layer : :class:`Layer`
            Previous layer with 4-D Tensor of the shape (batch, height, width, channels) or 3-D Tensor of the shape (height, width, channels).
        """
        outputs = tf.image.resize(inputs, size=self.size, method=self.method, align_corners=self.align_corners)
        return outputs
