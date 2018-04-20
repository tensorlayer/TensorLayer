# -*- coding: utf-8 -*-

import tensorflow as tf

from .. import _logging as logging
from .core import *

from ..deprecation import deprecated_alias

__all__ = [
    'SubpixelConv1d',
    'SubpixelConv2d',
]


class SubpixelConv2d(Layer):
    """It is a 2D sub-pixel up-sampling layer, usually be used
    for Super-Resolution applications, see `SRGAN <https://github.com/zsdonghao/SRGAN/>`__ for example.

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
    >>> x = np.random.rand(2, 16, 16, 4)
    >>> X = tf.placeholder("float32", shape=(2, 16, 16, 4), name="X")
    >>> net = InputLayer(X, name='input')
    >>> net = SubpixelConv2d(net, scale=2, n_out_channel=1, name='subpixel_conv2d')
    >>> y = sess.run(net.outputs, feed_dict={X: x})
    >>> print(x.shape, y.shape)
    ... (2, 16, 16, 4) (2, 32, 32, 1)
    >>>
    >>> x = np.random.rand(2, 16, 16, 4*10)
    >>> X = tf.placeholder("float32", shape=(2, 16, 16, 4*10), name="X")
    >>> net = InputLayer(X, name='input2')
    >>> net = SubpixelConv2d(net, scale=2, n_out_channel=10, name='subpixel_conv2d2')
    >>> y = sess.run(net.outputs, feed_dict={X: x})
    >>> print(x.shape, y.shape)
    ... (2, 16, 16, 40) (2, 32, 32, 10)
    >>>
    >>> x = np.random.rand(2, 16, 16, 25*10)
    >>> X = tf.placeholder("float32", shape=(2, 16, 16, 25*10), name="X")
    >>> net = InputLayer(X, name='input3')
    >>> net = SubpixelConv2d(net, scale=5, n_out_channel=None, name='subpixel_conv2d3')
    >>> y = sess.run(net.outputs, feed_dict={X: x})
    >>> print(x.shape, y.shape)
    ... (2, 16, 16, 250) (2, 80, 80, 10)

    References
    ------------
    - `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/pdf/1609.05158.pdf>`__

    """
    # github/Tetrachrome/subpixel  https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py
    @deprecated_alias(net='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, scale=2, n_out_channel=None, act=tf.identity, name='subpixel_conv2d'):
        _err_log = "SubpixelConv2d: The number of input channels == (scale x scale) x The number of output channels"

        super(SubpixelConv2d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "SubpixelConv2d  %s: scale: %d n_out_channel: %s act: %s" % (name, scale, n_out_channel, act.__name__)
        )

        def _PS(X, r, n_out_channels):
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
                logging.info(_err_log)
            return X

        self.inputs = prev_layer.outputs
        if n_out_channel is None:
            if int(self.inputs.get_shape()[-1]) / (scale**2) % 1 != 0:
                raise Exception(_err_log)
            n_out_channel = int(int(self.inputs.get_shape()[-1]) / (scale**2))

        with tf.variable_scope(name):
            self.outputs = act(_PS(self.inputs, r=scale, n_out_channels=n_out_channel))

        self.all_layers.append(self.outputs)


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
    >>> t_signal = tf.placeholder('float32', [10, 100, 4], name='x')
    >>> n = InputLayer(t_signal, name='in')
    >>> n = SubpixelConv1d(n, scale=2, name='s')
    >>> print(n.outputs.shape)
    ... (10, 200, 2)

    References
    -----------
    `Audio Super Resolution Implementation <https://github.com/kuleshov/audio-super-res/blob/master/src/models/layers/subpixel.py>`__.

    """

    @deprecated_alias(net='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, scale=2, act=tf.identity, name='subpixel_conv1d'):

        def _PS(I, r):
            X = tf.transpose(I, [2, 1, 0])  # (r, w, b)
            X = tf.batch_to_space_nd(X, [r], [[0, 0]])  # (1, r*w, b)
            X = tf.transpose(X, [2, 1, 0])
            return X

        super(SubpixelConv1d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info("SubpixelConv1d  %s: scale: %d act: %s" % (name, scale, act.__name__))

        self.inputs = prev_layer.outputs
        with tf.name_scope(name):
            self.outputs = act(_PS(self.inputs, r=scale))

        self.all_layers.append(self.outputs)


# Alias
# SubpixelConv2d = subpixel_conv2d
# SubpixelConv1d = subpixel_conv1d
