#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import private_method

__all__ = [
    'SubpixelConv1d',
    'SubpixelConv2d',
]


class SubpixelConv2d(Layer):
    """It is a 2D sub-pixel up-sampling layer, usually be used
    for Super-Resolution applications, see `SRGAN <https://github.com/tensorlayer/srgan/>`__ for example.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        Previous layer,
    scale : int
        The up-scaling ratio, a wrong setting will lead to dimension size error.
    n_out_channel : int or None
        The number of output channels.
        - If None, automatically set n_out_channel == the number of input channels / (scale x scale).
        - The number of input channels == (scale x scale) x The number of output channels.
    act : activation function
        The activation function of this layer.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> # examples here just want to tell you how to set the n_out_channel.
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = np.random.rand(2, 16, 16, 4)
    >>> X = tf.placeholder("float32", shape=(2, 16, 16, 4), name="X")
    >>> net = tl.layers.InputLayer(X, name='input')
    >>> net = tl.layers.SubpixelConv2d(net, scale=2, n_out_channel=1, name='subpixel_conv2d')
    >>> sess = tf.Session()
    >>> y = sess.run(net.outputs, feed_dict={X: x})
    >>> print(x.shape, y.shape)
    (2, 16, 16, 4) (2, 32, 32, 1)

    >>> x = np.random.rand(2, 16, 16, 4*10)
    >>> X = tf.placeholder("float32", shape=(2, 16, 16, 4*10), name="X")
    >>> net = tl.layers.InputLayer(X, name='input2')
    >>> net = tl.layers.SubpixelConv2d(net, scale=2, n_out_channel=10, name='subpixel_conv2d2')
    >>> y = sess.run(net.outputs, feed_dict={X: x})
    >>> print(x.shape, y.shape)
    (2, 16, 16, 40) (2, 32, 32, 10)

    >>> x = np.random.rand(2, 16, 16, 25*10)
    >>> X = tf.placeholder("float32", shape=(2, 16, 16, 25*10), name="X")
    >>> net = tl.layers.InputLayer(X, name='input3')
    >>> net = tl.layers.SubpixelConv2d(net, scale=5, n_out_channel=None, name='subpixel_conv2d3')
    >>> y = sess.run(net.outputs, feed_dict={X: x})
    >>> print(x.shape, y.shape)
    (2, 16, 16, 250) (2, 80, 80, 10)

    References
    ------------
    - `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/pdf/1609.05158.pdf>`__

    """
    # github/Tetrachrome/subpixel  https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py
    @deprecated_alias(net='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, scale=2, n_out_channel=None, act=None, name='subpixel_conv2d'):

        super(SubpixelConv2d, self).__init__(prev_layer=prev_layer, act=act, name=name)

        if n_out_channel is None:

            if int(self.inputs.get_shape()[-1]) / (scale**2) % 1 != 0:
                raise Exception(
                    "SubpixelConv2d: The number of input channels == (scale x scale) x The number of output channels"
                )

            n_out_channel = int(int(self.inputs.get_shape()[-1]) / (scale**2))

        logging.info(
            "SubpixelConv2d  %s: scale: %d n_out_channel: %s act: %s" %
            (self.name, scale, n_out_channel, self.act.__name__ if self.act is not None else 'No Activation')
        )

        with tf.variable_scope(name):
            self.outputs = self._apply_activation(self._PS(self.inputs, r=scale, n_out_channels=n_out_channel))

        self._add_layers(self.outputs)

    @private_method
    def _PS(self, X, r, n_out_channels):

        _err_log = "SubpixelConv2d: The number of input channels == (scale x scale) x The number of output channels"

        if n_out_channels >= 1:
            if int(X.get_shape()[-1]) != (r**2) * n_out_channels:
                raise Exception(_err_log)
            # bsize, a, b, c = X.get_shape().as_list()
            # bsize = tf.shape(X)[0] # Handling Dimension(None) type for undefined batch dim
            # Xs=tf.split(X,r,3) #b*h*w*r*r
            # Xr=tf.concat(Xs,2) #b*h*(r*w)*r
            # X=tf.reshape(Xr,(bsize,r*a,r*b,n_out_channel)) # b*(r*h)*(r*w)*c

            X = tf.depth_to_space(X, r)
        else:
            raise RuntimeError(_err_log)

        return X


class SubpixelConv1d(Layer):
    """It is a 1D sub-pixel up-sampling layer.

    Calls a TensorFlow function that directly implements this functionality.
    We assume input has dim (batch, width, r)

    Parameters
    ------------
    net : :class:`Layer`
        Previous layer with output shape of (batch, width, r).
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

    @deprecated_alias(net='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, scale=2, act=None, name='subpixel_conv1d'):

        super(SubpixelConv1d, self).__init__(prev_layer=prev_layer, act=act, name=name)

        logging.info(
            "SubpixelConv1d  %s: scale: %d act: %s" %
            (self.name, scale, self.act.__name__ if self.act is not None else 'No Activation')
        )

        with tf.name_scope(name):
            self.outputs = self._apply_activation(self._PS(self.inputs, r=scale))

        self._add_layers(self.outputs)

    @private_method
    def _PS(self, I, r):
        X = tf.transpose(I, [2, 1, 0])  # (r, w, b)
        X = tf.batch_to_space_nd(X, [r], [[0, 0]])  # (1, r*w, b)
        X = tf.transpose(X, [2, 1, 0])
        return X
