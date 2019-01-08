#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'PadLayer',
    'ZeroPad1d',
    'ZeroPad2d',
    'ZeroPad3d',
]


class PadLayer(Layer):
    """The :class:`PadLayer` class is a padding layer for any mode and dimension.
    Please see `tf.pad <https://www.tensorflow.org/api_docs/python/tf/pad>`__ for usage.

    Parameters
    ----------
    padding : list of lists of 2 ints, or a Tensor of type int32.
        The int32 values to pad.
    mode : str
        "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).
    name : None or str
        A unique layer name.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> net = tl.layers.Input(images, name='in')
    >>> net = tl.layers.PadLayer(net, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT", name='inpad')

    """

    def __init__(
            self,
            padding=None,
            mode='CONSTANT',
            name=None,  #'pad_layer',
    ):
        # super(PadLayer, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.padding = padding
        self.mode = mode

        logging.info("PadLayer   %s: padding: %s mode: %s" % (self.name, list(self.padding), self.mode))

        if self.padding is None:
            raise Exception(
                "padding should be a Tensor of type int32. see https://www.tensorflow.org/api_docs/python/tf/pad"
            )

    def build(self, inputs):
        pass

    def forward(self, inputs):

        outputs = tf.pad(inputs, paddings=self.padding, mode=self.mode, name=self.name)
        return outputs


class ZeroPad1d(Layer):
    """
    The :class:`ZeroPad1d` class is a 1D padding layer for signal [batch, length, channel].

    Parameters
    ----------
    padding : int, or tuple of 2 ints
            - If int, zeros to add at the beginning and end of the padding dimension (axis 1).
            - If tuple of 2 ints, zeros to add at the beginning and at the end of the padding dimension.
    name : None or str
        A unique layer name.

    """

    def __init__(
            self,
            padding,
            name=None,  #'zeropad1d',
    ):
        # super(ZeroPad1d, self).__init__(prev_layer=prev_layer, name=name)
        self.padding = padding
        logging.info("ZeroPad1d   %s: padding: %s" % (self.name, str(padding)))

        if not isinstance(self.padding, (int, tuple, dict)):
            raise AssertionError()

    def build(self, inputs):
        self.layer = tf.keras.layers.ZeroPadding1D(padding=self.padding, name=self.name)

    def forward(self, inputs):
        outputs = self.layer(inputs)


class ZeroPad2d(Layer):
    """
    The :class:`ZeroPad2d` class is a 2D padding layer for image [batch, height, width, channel].

    Parameters
    ----------
    padding : int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int, the same symmetric padding is applied to width and height.
            - If tuple of 2 ints, interpreted as two different symmetric padding values for height and width as ``(symmetric_height_pad, symmetric_width_pad)``.
            - If tuple of 2 tuples of 2 ints, interpreted as ``((top_pad, bottom_pad), (left_pad, right_pad))``.
    name : None or str
        A unique layer name.

    """

    def __init__(
            self,
            padding,
            name=None,  #'zeropad2d',
    ):
        # super(ZeroPad2d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)

        self.padding = padding
        logging.info("ZeroPad2d   %s: padding: %s" % (self.name, str(self.padding)))

        if not isinstance(self.padding, (int, tuple)):
            raise AssertionError("Padding should be of type `int` or `tuple`")

    def build(self, inputs):
        self.layer = tf.keras.layers.ZeroPadding2D(padding=self.padding, name=self.name)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        return outputs


class ZeroPad3d(Layer):
    """
    The :class:`ZeroPad3d` class is a 3D padding layer for volume [batch, depth, height, width, channel].

    Parameters
    ----------
    padding : int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int, the same symmetric padding is applied to width and height.
            - If tuple of 2 ints, interpreted as two different symmetric padding values for height and width as ``(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)``.
            - If tuple of 2 tuples of 2 ints, interpreted as ``((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))``.
    name : None or str
        A unique layer name.

    """

    def __init__(
            self,
            padding,
            name=None,  #'zeropad3d',
    ):
        # super(ZeroPad3d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.padding = padding

        logging.info("ZeroPad3d   %s: padding: %s" % (self.name, str(self.padding)))

        if not isinstance(self.padding, (int, tuple)):
            raise AssertionError()

    def build(self, inputs):
        self.layer = tf.keras.layers.ZeroPadding3D(padding=self.padding, name=self.name)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        return outputs
