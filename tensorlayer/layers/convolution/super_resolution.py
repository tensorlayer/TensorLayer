#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'SubpixelConv1d',
    'SubpixelConv2d',
]


class SubpixelConv1d(Layer):
    """It is a 1D sub-pixel up-sampling layer.

    Calls a TensorFlow function that directly implements this functionality.
    We assume input has dim (batch, width, r)

    Parameters
    ------------
    scale : int
        The up-scaling ratio, a wrong setting will lead to Dimension size error.
    act : activation function
        The activation function of this layer.
    name : str
        A unique layer name.

    Examples
    ----------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> t_signal = tf.placeholder('float32', [10, 100, 4], name='x')
    >>> n = tl.layers.InputLayer(t_signal, name='in')
    >>> n = tl.layers.SubpixelConv1d(n, scale=2, name='s')
    >>> print(n.outputs.shape)
    (10, 200, 2)

    References
    -----------
    `Audio Super Resolution Implementation <https://github.com/kuleshov/audio-super-res/blob/master/src/models/layers/subpixel.py>`__.

    """

    def __init__(self, scale=2, act=None, name='subpixel_conv1d'):

        self.scale = scale
        self.act = act
        self.name = name

        super(SubpixelConv1d, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("scale: %d" % self.scale)
        except AttributeError:
            pass

        return self._str(additional_str)

    def compile(self):

        with tf.variable_scope(self.name):

            self._temp_data['outputs'] = tf.transpose(self._temp_data['inputs'], [2, 1, 0])  # (r, w, b)

            self._temp_data['outputs'] = tf.batch_to_space_nd(self._temp_data['outputs'],
                                                              [self.scale], [[0, 0]])  # (1, r*w, b)

            self._temp_data['outputs'] = tf.transpose(self._temp_data['outputs'], [2, 1, 0])

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])


class SubpixelConv2d(Layer):
    """It is a 2D sub-pixel up-sampling layer, usually be used
    for Super-Resolution applications, see `SRGAN <https://github.com/tensorlayer/srgan/>`__ for example.

    Parameters
    ------------
    scale : int
        The up-scaling ratio, a wrong setting will lead to dimension size error.
    n_out_channels : int or None
        The number of output channels.
        - If None, automatically set n_out_channels == the number of input channels / (scale x scale).
        - The number of input channels == (scale x scale) x The number of output channels.
    act : activation function
        The activation function of this layer.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> # examples here just want to tell you how to set the n_out_channels.
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = np.random.rand(2, 16, 16, 4)
    >>> X = tf.placeholder("float32", shape=(2, 16, 16, 4), name="X")
    >>> net = tl.layers.InputLayer(X, name='input')
    >>> net = tl.layers.SubpixelConv2d(net, scale=2, n_out_channels=1, name='subpixel_conv2d')
    >>> sess = tf.Session()
    >>> y = sess.run(net.outputs, feed_dict={X: x})
    >>> print(x.shape, y.shape)
    (2, 16, 16, 4) (2, 32, 32, 1)

    >>> x = np.random.rand(2, 16, 16, 4*10)
    >>> X = tf.placeholder("float32", shape=(2, 16, 16, 4*10), name="X")
    >>> net = tl.layers.InputLayer(X, name='input2')
    >>> net = tl.layers.SubpixelConv2d(net, scale=2, n_out_channels=10, name='subpixel_conv2d2')
    >>> y = sess.run(net.outputs, feed_dict={X: x})
    >>> print(x.shape, y.shape)
    (2, 16, 16, 40) (2, 32, 32, 10)

    >>> x = np.random.rand(2, 16, 16, 25*10)
    >>> X = tf.placeholder("float32", shape=(2, 16, 16, 25*10), name="X")
    >>> net = tl.layers.InputLayer(X, name='input3')
    >>> net = tl.layers.SubpixelConv2d(net, scale=5, n_out_channels=None, name='subpixel_conv2d3')
    >>> y = sess.run(net.outputs, feed_dict={X: x})
    >>> print(x.shape, y.shape)
    (2, 16, 16, 250) (2, 80, 80, 10)

    References
    ------------
    - `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/pdf/1609.05158.pdf>`__

    """

    # github/Tetrachrome/subpixel  https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py
    def __init__(self, scale=2, n_out_channels=None, act=None, name='subpixel_conv2d'):

        self.scale = scale
        self.n_out_channels = n_out_channels
        self.act = act
        self.name = name

        super(SubpixelConv2d, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("scale: %d" % self.scale)
        except AttributeError:
            pass

        try:
            additional_str.append("n_out_channels: %d" % self.n_out_channels)
        except AttributeError:
            pass

        return self._str(additional_str)

    def compile(self):

        if self.n_out_channels is None:

            if int(self._temp_data['inputs'].get_shape()[-1]) / (self.scale**2) % 1 != 0:
                raise RuntimeError(
                    "%s: The number of input channels == (scale x scale) x The number of output channels" %
                    self.__class__.__name__
                )

            self.n_out_channels = int(int(self._temp_data['inputs'].get_shape()[-1]) / (self.scale**2))

        if self.n_out_channels < 1 or int(self._temp_data['inputs'].get_shape()[-1]
                                         ) != (self.scale**2) * self.n_out_channels:
            _err_log = "%s: The number of input channels == (scale x scale) x n_out_channels" % (
                self.__class__.__name__
            )
            raise Exception(_err_log)

        with tf.variable_scope(self.name):

            # bsize, a, b, c = self._temp_data['outputs'].get_shape().as_list()
            # bsize = tf.shape(self._temp_data['outputs'])[0]  # Handling Dimension(None) type for undefined batch dim
            # x_s = tf.split(self._temp_data['outputs'], self.scale, 3)  # b*h*w*r*r
            # x_r = tf.concat(x_s, 2)  # b*h*(r*w)*r
            # self._temp_data['outputs'] = tf.reshape(x_r, (bsize, self.scale*a, self.scale*b, self.n_out_channels))  # b*(r*h)*(r*w)*c

            self._temp_data['outputs'] = tf.depth_to_space(self._temp_data['inputs'], self.scale)

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
