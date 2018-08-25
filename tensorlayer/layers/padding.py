#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import auto_parse_inputs
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

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
    prev_layer : :class:`Layer`
        The previous layer.
    padding : list of lists of 2 ints, or a Tensor of type int32.
        The int32 values to pad.
    mode : str
        "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).
    name : str
        A unique layer name.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> net = tl.layers.InputLayer(images, name='in')
    >>> net = tl.layers.PadLayer(net, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT", name='inpad')

    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self,
        prev_layer=None,
        padding=None,
        mode='CONSTANT',
        name='pad_layer',
    ):

        if padding is None:
            raise Exception(
                "padding should be a Tensor of type int32. see https://www.tensorflow.org/api_docs/python/tf/pad"
            )

        if not isinstance(padding, (int, tuple, list)):
            raise AssertionError()

        self.prev_layer = prev_layer
        self.padding = padding
        self.mode = mode
        self.name = name

        super(PadLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("padding: %s" % str(self.padding))
        except AttributeError:
            pass

        try:
            additional_str.append("mode: %s" % self.mode)
        except AttributeError:
            pass

        try:
            additional_str.append("out_shape: %s" % self._temp_data['outputs'].get_shape())
        except AttributeError:
            pass

        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):

        self._temp_data['outputs'] = tf.pad(
            self._temp_data['inputs'], paddings=self.padding, mode=self.mode, name=self.name
        )

        self._add_layers(self._temp_data['outputs'])


class ZeroPad1d(Layer):
    """
    The :class:`ZeroPad1d` class is a 1D padding layer for signal [batch, length, channel].

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer.
    padding : int, or tuple of 2 ints
            - If int, zeros to add at the beginning and end of the padding dimension (axis 1).
            - If tuple of 2 ints, zeros to add at the beginning and at the end of the padding dimension.
    name : str
        A unique layer name.

    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self,
        prev_layer=None,
        padding=None,
        name='zeropad1d',
    ):

        if not isinstance(padding, (int, tuple, list)):
            raise AssertionError()

        if isinstance(padding, (tuple, list)) and len(padding) != 2:
            raise AssertionError()

        self.prev_layer = prev_layer
        self.padding = padding
        self.name = name

        super(ZeroPad1d, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("padding: %s" % str(self.padding))
        except AttributeError:
            pass

        try:
            additional_str.append("out_shape: %s" % self._temp_data['outputs'].shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):

        # TODO: Stop using Keras
        self._temp_data['outputs'] = tf.keras.layers.ZeroPadding1D(
            padding=self.padding, name=self.name
        )(self._temp_data['inputs'])

        self._add_layers(self._temp_data['outputs'])


class ZeroPad2d(Layer):
    """
    The :class:`ZeroPad2d` class is a 2D padding layer for image [batch, height, width, channel].

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer.
    padding : int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int, the same symmetric padding is applied to width and height.
            - If tuple of 2 ints, interpreted as two different symmetric padding values for height and width as ``(symmetric_height_pad, symmetric_width_pad)``.
            - If tuple of 2 tuples of 2 ints, interpreted as ``((top_pad, bottom_pad), (left_pad, right_pad))``.
    name : str
        A unique layer name.

    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self,
        prev_layer=None,
        padding=None,
        name='zeropad2d',
    ):

        if not isinstance(padding, (int, tuple, list)):
            raise AssertionError()

        if isinstance(padding, (tuple, list)) and len(padding) != 2:
            raise AssertionError()

        self.prev_layer = prev_layer
        self.padding = padding
        self.name = name

        super(ZeroPad2d, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("padding: %s" % str(self.padding))
        except AttributeError:
            pass

        try:
            additional_str.append("out_shape: %s" % self._temp_data['outputs'].shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):

        # TODO: Stop using Keras
        self._temp_data['outputs'] = tf.keras.layers.ZeroPadding2D(
            padding=self.padding, name=self.name
        )(self._temp_data['inputs'])

        self._add_layers(self._temp_data['outputs'])


class ZeroPad3d(Layer):
    """
    The :class:`ZeroPad3d` class is a 3D padding layer for volume [batch, depth, height, width, channel].

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer.
    padding : int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int, the same symmetric padding is applied to width and height.
            - If tuple of 2 ints, interpreted as two different symmetric padding values for height and width as ``(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)``.
            - If tuple of 2 tuples of 2 ints, interpreted as ``((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))``.
    name : str
        A unique layer name.

    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self,
        prev_layer=None,
        padding=None,
        name='zeropad3d',
    ):

        if not isinstance(padding, (int, tuple, list)):
            raise AssertionError()

        if isinstance(padding, (tuple, list)) and len(padding) != 3:
            raise AssertionError()

        self.prev_layer = prev_layer
        self.padding = padding
        self.name = name

        super(ZeroPad3d, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("padding: %s" % str(self.padding))
        except AttributeError:
            pass

        try:
            additional_str.append("out_shape: %s" % self._temp_data['outputs'].shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):

        # TODO: Stop using Keras
        self._temp_data['outputs'] = tf.keras.layers.ZeroPadding3D(
            padding=self.padding, name=self.name
        )(self._temp_data['inputs'])

        self._add_layers(self._temp_data['outputs'])
