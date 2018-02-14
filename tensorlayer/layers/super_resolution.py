# -*- coding: utf-8 -*-
from .core import *


def subpixel_conv2d(net, scale=2, n_out_channel=None, act=tf.identity, name='subpixel_conv2d'):
    """It is a sub-pixel 2d upsampling layer, usually be used
    for Super-Resolution applications, see `example code <https://github.com/zsdonghao/SRGAN/>`_.

    Parameters
    ------------
    net : TensorLayer layer.
    scale : int, upscaling ratio, a wrong setting will lead to Dimension size error.
    n_out_channel : int or None, the number of output channels.
        Note that, the number of input channels == (scale x scale) x The number of output channels.
        If None, automatically set n_out_channel == the number of input channels / (scale x scale).
    act : activation function.
    name : string.
        An optional name to attach to this layer.

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
    - `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/pdf/1609.05158.pdf>`_
    """
    # github/Tetrachrome/subpixel  https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py

    _err_log = "SubpixelConv2d: The number of input channels == (scale x scale) x The number of output channels"

    scope_name = tf.get_variable_scope().name
    if scope_name:
        whole_name = scope_name + '/' + name
    else:
        whole_name = name

    def _PS(X, r, n_out_channels):
        if n_out_channels >= 1:
            assert int(X.get_shape()[-1]) == (r**2) * n_out_channels, _err_log
            '''
            bsize, a, b, c = X.get_shape().as_list()
            bsize = tf.shape(X)[0] # Handling Dimension(None) type for undefined batch dim
            Xs=tf.split(X,r,3) #b*h*w*r*r
            Xr=tf.concat(Xs,2) #b*h*(r*w)*r
            X=tf.reshape(Xr,(bsize,r*a,r*b,n_out_channel)) # b*(r*h)*(r*w)*c
            '''
            X = tf.depth_to_space(X, r)
        else:
            print(_err_log)
        return X

    inputs = net.outputs

    if n_out_channel is None:
        assert int(inputs.get_shape()[-1]) / (scale**2) % 1 == 0, _err_log
        n_out_channel = int(int(inputs.get_shape()[-1]) / (scale**2))

    print("  [TL] SubpixelConv2d  %s: scale: %d n_out_channel: %s act: %s" % (name, scale, n_out_channel, act.__name__))

    net_new = Layer(inputs, name=whole_name)
    # with tf.name_scope(name):
    with tf.variable_scope(name) as vs:
        net_new.outputs = act(_PS(inputs, r=scale, n_out_channels=n_out_channel))

    net_new.all_layers = list(net.all_layers)
    net_new.all_params = list(net.all_params)
    net_new.all_drop = dict(net.all_drop)
    net_new.all_layers.extend([net_new.outputs])
    return net_new


def subpixel_conv2d_deprecated(net, scale=2, n_out_channel=None, act=tf.identity, name='subpixel_conv2d'):
    """It is a sub-pixel 2d upsampling layer, usually be used
    for Super-Resolution applications, `example code <https://github.com/zsdonghao/SRGAN/>`_.

    Parameters
    ------------
    net : TensorLayer layer.
    scale : int, upscaling ratio, a wrong setting will lead to Dimension size error.
    n_out_channel : int or None, the number of output channels.
        Note that, the number of input channels == (scale x scale) x The number of output channels.
        If None, automatically set n_out_channel == the number of input channels / (scale x scale).
    act : activation function.
    name : string.
        An optional name to attach to this layer.

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
    - `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/pdf/1609.05158.pdf>`_
    """
    # github/Tetrachrome/subpixel  https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py

    _err_log = "SubpixelConv2d: The number of input channels == (scale x scale) x The number of output channels"

    scope_name = tf.get_variable_scope().name
    if scope_name:
        name = scope_name + '/' + name

    def _PS(X, r, n_out_channel):
        if n_out_channel > 1:
            assert int(X.get_shape()[-1]) == (r**2) * n_out_channel, _err_log
            X = tf.transpose(X, [0, 2, 1, 3])
            X = tf.depth_to_space(X, r)
            X = tf.transpose(X, [0, 2, 1, 3])
        else:
            print(_err_log)
        return X

    inputs = net.outputs

    if n_out_channel is None:
        assert int(inputs.get_shape()[-1]) / (scale**2) % 1 == 0, _err_log
        n_out_channel = int(int(inputs.get_shape()[-1]) / (scale**2))

    print("  [TL] SubpixelConv2d  %s: scale: %d n_out_channel: %s act: %s" % (name, scale, n_out_channel, act.__name__))

    net_new = Layer(inputs, name=name)
    # with tf.name_scope(name):
    with tf.variable_scope(name) as vs:
        net_new.outputs = act(_PS(inputs, r=scale, n_out_channel=n_out_channel))

    net_new.all_layers = list(net.all_layers)
    net_new.all_params = list(net.all_params)
    net_new.all_drop = dict(net.all_drop)
    net_new.all_layers.extend([net_new.outputs])
    return net_new


def subpixel_conv1d(net, scale=2, act=tf.identity, name='subpixel_conv1d'):
    """One-dimensional subpixel upsampling layer.
    Calls a tensorflow function that directly implements this functionality.
    We assume input has dim (batch, width, r)

    Parameters
    ------------
    net : TensorLayer layer.
    scale : int, upscaling ratio, a wrong setting will lead to Dimension size error.
    act : activation function.
    name : string.
        An optional name to attach to this layer.

    Examples
    ----------
    >>> t_signal = tf.placeholder('float32', [10, 100, 4], name='x')
    >>> n = InputLayer(t_signal, name='in')
    >>> n = SubpixelConv1d(n, scale=2, name='s')
    >>> print(n.outputs.shape)
    ... (10, 200, 2)

    References
    -----------
    - `Audio Super Resolution Implementation <https://github.com/kuleshov/audio-super-res/blob/master/src/models/layers/subpixel.py>`_.
    """

    def _PS(I, r):
        X = tf.transpose(I, [2, 1, 0])  # (r, w, b)
        X = tf.batch_to_space_nd(X, [r], [[0, 0]])  # (1, r*w, b)
        X = tf.transpose(X, [2, 1, 0])
        return X

    print("  [TL] SubpixelConv1d  %s: scale: %d act: %s" % (name, scale, act.__name__))

    inputs = net.outputs
    net_new = Layer(inputs, name=name)
    with tf.name_scope(name):
        net_new.outputs = act(_PS(inputs, r=scale))

    net_new.all_layers = list(net.all_layers)
    net_new.all_params = list(net.all_params)
    net_new.all_drop = dict(net.all_drop)
    net_new.all_layers.extend([net_new.outputs])
    return net_new


# Alias
SubpixelConv2d = subpixel_conv2d
SubpixelConv1d = subpixel_conv1d
