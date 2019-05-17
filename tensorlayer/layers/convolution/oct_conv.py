#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer

# from tensorlayer.layers.core import LayersConfig



__all__ = [
    'OctConv2dIn',
    'OctConv2d',
    'OctConv2dOut',
    'OctConv2dHighOut',
    'OctConv2dLowOut',
    'OctConv2dConcat',
]

class OctConv2dIn(Layer):
    """
    The :class:`OctConv2dIn` class is a preprocessing layer for 2D image [batch, height, width, channel],
     see `Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave
     Convolution <https://arxiv.org/abs/1904.05049>`__.
    Parameters
    ----------
    name : None or str
        A unique layer name.
    Notes
    -----
    - The height and width of input must be a multiple of the 2.
    - Use this layer before any other octconv layers.
    - The output will be a list which contains 2 tensor.
    Examples
    --------
    With TensorLayer
    >>> net = tl.layers.Input([8, 28, 28, 16], name='input')
    >>> octconv2d = tl.layers.OctConv2dIn(name='octconv2din_1')(net)
    >>> print(octconv2d)
    >>> output shape : [(8, 28, 28, 16),(8, 14, 14, 16)]
    """

    def __init__(
            self,
            name=None,  # 'cnn2d_layer',
    ):
        super().__init__(name)
        self.build(None)
        self._built = True

        logging.info(
            "OctConv2dIn %s: " % (
                self.name,
            )
        )

    def __repr__(self):
        s = ('{classname}(')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs):
        pass

    def forward(self, inputs):
        high_out=tf.identity(inputs,name=(self.name+'_high_out'))
        low_out  = tf.nn.avg_pool2d(inputs, (2,2), strides=(2,2),padding='SAME',name=self.name+'_low_out')
        outputs=[high_out,low_out]
        return outputs


class OctConv2d(Layer):
    """
    The :class:`OctConv2d` class is a 2D CNN layer for OctConv2d layer output, see
    `Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with
    Octave Convolution <https://arxiv.org/abs/1904.05049>`__. Use this layer to process tensor list.
    Parameters
    ----------
    filter : int
        The sum of the number of filters.
    alpha : :float
        The percentage of high_res output.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    act : activation function
        The activation function of this layer.
    name : None or str
        A unique layer name.
    Notes
    -----
    - The input should be a list with shape [high_res_tensor , low_res_tensor],
    the height and width of high_res should be twice of the low_res_tensor.
    - If you do not which tensor is larger, use OctConv2dConcat layer.
    - The output will be a list which contains 2 tensor.
    - You should not use the output directly.
    Examples
    --------
    With TensorLayer
    >>> net = tl.layers.Input([8, 28, 28, 32], name='input')
    >>> octconv2d = tl.layers.OctConv2dIn(name='octconv2din_1')(net)
    >>> print(octconv2d)
    >>> output shape : [(8, 28, 28, 32),(8, 14, 14, 32)]
    >>> octconv2d = tl.layers.OctConv2d(32,0.5,act=tf.nn.relu, name='octconv2d_1')(octconv2d)
    >>> print(octconv2d)
    >>> output shape : [(8, 28, 28, 16),(8, 14, 14, 16)]
    """

    def __init__(
            self,
            filter=32,
            alpha=0.5,
            filter_size=(2, 2),
            strides=(1,1),
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            act=None,
            in_channels=None,
            name=None  # 'cnn2d_layer',
    ):
        super().__init__(name)
        self.filter = filter
        self.alpha = alpha
        if (self.alpha >= 1) or (self.alpha <= 0):
            raise ValueError(
                "The alpha must be in (0,1)")
        self.high_out = int(self.alpha * self.filter)
        self.low_out = self.filter - self.high_out
        if (self.high_out == 0) or (self.low_out == 0):
            raise ValueError(
                "The output channel must be greater than 0.")
        self.filter_size = filter_size
        self.strides = strides
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.in_channels = in_channels
        if self.in_channels:
            self.build(None)
            self._built = True


        logging.info(
            "OctConv2d %s: filter_size: %s strides: %s high_out: %s low_out: %s act: %s" % (
                self.name, str(filter_size), str(strides), str(self.high_out), str(self.low_out),
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(in_channels={in_channels}, out_channels={filter} kernel_size={filter_size}'
             ', strides={strides}')
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'

        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if not self.in_channels:
            high_ch=inputs_shape[0][-1]
            low_ch=inputs_shape[1][-1]
        else:
            high_ch=self.in_channels[0]
            low_ch=self.in_channels[1]
        self.high_high_filter_shape = (
            self.filter_size[0], self.filter_size[1], high_ch, self.high_out
        )
        self.high_low_filter_shape = (
            self.filter_size[0], self.filter_size[1], high_ch, self.low_out
        )
        self.low_low_filter_shape = (
            self.filter_size[0], self.filter_size[1], low_ch, self.low_out
        )
        self.low_high_filter_shape = (
            self.filter_size[0], self.filter_size[1], low_ch, self.high_out
        )
        self.high_high__W = self._get_weights(
            "high_high_filters", shape=self.high_high_filter_shape, init=self.W_init
        )
        self.high_low__W = self._get_weights(
            "high_low_filters", shape=self.high_low_filter_shape, init=self.W_init
        )
        self.low_low_W = self._get_weights(
            "low_low_filters", shape=self.low_low_filter_shape, init=self.W_init
        )
        self.low_high_W = self._get_weights(
            "low_high_filters", shape=self.low_high_filter_shape, init=self.W_init
        )
        if self.b_init:
            self.high_b = self._get_weights(
                "high_biases", shape=(self.high_out), init=self.b_init
            )
            self.low_b = self._get_weights(
                "low_biases", shape=(self.low_out), init=self.b_init
            )

    def forward(self, inputs):
        high_input = inputs[0]
        low_input=inputs[1]
        high_to_high = tf.nn.conv2d(high_input, self.high_high__W,
                                    strides=self.strides, padding="SAME")
        high_to_low =tf.nn.avg_pool2d(high_input, (2,2), strides=(2,2),padding='SAME')
        high_to_low=tf.nn.conv2d(high_to_low, self.high_low__W,
                               strides=self.strides, padding="SAME")
        low_to_low = tf.nn.conv2d(low_input, self.low_low_W,
                                    strides=self.strides, padding="SAME")
        low_to_high = tf.nn.conv2d(low_input, self.low_high_W,
                                    strides=self.strides, padding="SAME")
        low_to_high=tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(low_to_high)
        high_out=high_to_high+low_to_high
        low_out=low_to_low+high_to_low
        if self.b_init:
            high_out = tf.nn.bias_add(high_out, self.high_b, data_format="NHWC")
            low_out = tf.nn.bias_add(low_out, self.low_b, data_format="NHWC")
        if self.act:
            high_out = self.act(high_out,name=self.name+'_high_out')
            low_out= self.act(low_out,name=self.name+'_low_out')
        else:
            high_out=tf.identity(high_out,name=self.name+'_high_out')
            low_out=tf.identity(low_out,name=self.name+'_low_out')
        outputs=[high_out,low_out]
        return outputs



class OctConv2dOut(Layer):
    """
    The :class:`OctConv2dOut` class is a 2D CNN layer for OctConv2d layer output to get only a tensor, see
    `Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
    <https://arxiv.org/abs/1904.05049>`__.
    Parameters
    ----------
    filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    act : activation function
        The activation function of this layer.
    name : None or str
        A unique layer name.
    Notes
    -----
    - Use this layer to get only a tensor for other normal layer.
    Examples
    --------
    With TensorLayer
    >>> net = tl.layers.Input([8, 28, 28, 32], name='input')
    >>> octconv2d = tl.layers.OctConv2dIn(name='octconv2din_1')(net)
    >>> print(octconv2d)
    >>> output shape : [(8, 28, 28, 32),(8, 14, 14, 32)]
    >>> octconv2d = tl.layers.OctConv2d(32,0.5,act=tf.nn.relu, name='octconv2d_1')(octconv2d)
    >>> print(octconv2d)
    >>> output shape : [(8, 28, 28, 16),(8, 14, 14, 16)]
    >>> octconv2d = tl.layers.OctConv2dOut(32,act=tf.nn.relu, name='octconv2dout_1')(octconv2d)
    >>> print(octconv2d)
    >>> output shape : (8, 14, 14, 32)
    """

    def __init__(
            self,
            n_filter=32,
            filter_size=(2, 2),
            strides=(1,1),
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            act=None,
            in_channels=None,
            name=None  # 'cnn2d_layer',
    ):
        super().__init__(name)

        self.high_out = n_filter
        self.low_out = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.in_channels = in_channels
        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "OctConv2dOut %s: filter_size: %s strides: %s out_channels: %s  act: %s" % (
                self.name, str(filter_size), str(strides), str(self.low_out),
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(in_channels={in_channels}, out_channels={low_out}, kernel_size={filter_size}'
             ', strides={strides}')
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if not self.in_channels:
            high_ch=inputs_shape[0][-1]
            low_ch=inputs_shape[1][-1]
        else:
            high_ch=self.in_channels[0]
            low_ch=self.in_channels[1]
        self.high_low_filter_shape = (
            self.filter_size[0], self.filter_size[1], high_ch, self.high_out
        )
        self.low_low_filter_shape = (
            self.filter_size[0], self.filter_size[1], low_ch, self.low_out
        )
        self.high_low__W = self._get_weights(
            "high_low_filters", shape=self.high_low_filter_shape, init=self.W_init
        )
        self.low_low_W = self._get_weights(
            "low_low_filters", shape=self.low_low_filter_shape, init=self.W_init
        )
        if self.b_init:
            self.low_b = self._get_weights(
                "low_biases", shape=(self.low_out), init=self.b_init
            )

    def forward(self, inputs):
        high_input = inputs[0]
        low_input=inputs[1]
        high_to_low =tf.nn.avg_pool2d(high_input, (2,2), strides=(2,2),padding='SAME')
        high_to_low=tf.nn.conv2d(high_to_low, self.high_low__W,
                               strides=self.strides, padding="SAME")
        low_to_low = tf.nn.conv2d(low_input, self.low_low_W,
                                    strides=self.strides, padding="SAME")
        low_out=low_to_low+high_to_low
        if self.b_init:
            low_out = tf.nn.bias_add(low_out, self.low_b, data_format="NHWC")
        if self.act:
            outputs= self.act(low_out,name=self.name+'_low_out')
        else:
            outputs=tf.identity(low_out,name=self.name+'_low_out')
        return outputs




class OctConv2dHighOut(Layer):
    """
    The :class:`OctConv2dHighOut` class is a slice layer for Octconv tensor list, see
    `Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with
    Octave Convolution <https://arxiv.org/abs/1904.05049>`__.
    Parameters
    ----------
    name : None or str
        A unique layer name.
    Notes
    -----
    - Use this layer to get high resolution tensor.
    - If you want to do some customized normalization ops, use this layer with
    OctConv2dLowOut and OctConv2dConcat layers to implement your idea.
    Examples
    --------
    With TensorLayer
    >>> net = tl.layers.Input([8, 28, 28, 32], name='input')
    >>> octconv2d = tl.layers.OctConv2dIn(name='octconv2din_1')(net)
    >>> print(octconv2d)
    >>> output shape : [(8, 28, 28, 32),(8, 14, 14, 32)]
    >>> octconv2d = tl.layers.OctConv2dHighOut(name='octconv2dho_1')(octconv2d)
    >>> print(octconv2d)
    >>> output shape : (8, 28, 28, 32)
    """

    def __init__(
            self,
            name=None,  # 'cnn2d_layer',
    ):
        super().__init__(name)
        self.build(None)
        self._built = True

        logging.info(
            "OctConv2dHighOut %s: " % (
                self.name,
            )
        )

    def __repr__(self):

        s = ('{classname}(')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs):
        pass

    def forward(self, inputs):
        outputs=tf.identity(inputs[0],self.name)
        return outputs


class OctConv2dLowOut(Layer):
    """
    The :class:`OctConv2dLowOut` class is a slice layer for Octconv tensor list, see
    `Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with
    Octave Convolution <https://arxiv.org/abs/1904.05049>`__.
    Parameters
    ----------
    name : None or str
        A unique layer name.
    Notes
    -----
    - Use this layer to get low resolution tensor.
    - If you want to do some customized normalization ops, use this layer with
    OctConv2dHighOut and OctConv2dConcat layers to implement your idea.
    Examples
    --------
    With TensorLayer
    >>> net = tl.layers.Input([8, 28, 28, 32], name='input')
    >>> octconv2d = tl.layers.OctConv2dIn(name='octconv2din_1')(net)
    >>> print(octconv2d)
    >>> output shape : [(8, 28, 28, 32),(8, 14, 14, 32)]
    >>> octconv2d = tl.layers.OctConv2dLowOut(name='octconv2dlo_1')(octconv2d)
    >>> print(octconv2d)
    >>> output shape : (8, 14, 14, 32)
    """

    def __init__(
            self,
            name=None,  # 'cnn2d_layer',
    ):
        super().__init__(name)
        self.build(None)
        self._built = True

        logging.info(
            "OctConv2dHighOut %s: " % (
                self.name,
            )
        )

    def __repr__(self):

        s = ('{classname}(')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs):
        pass

    def forward(self, inputs):
        outputs=tf.identity(inputs[1],self.name)
        return outputs

class OctConv2dConcat(Layer):
    """
    The :class:`OctConv2dConcat` class is a concat layer for two 2D image batches, see
    `Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with
    Octave Convolution <https://arxiv.org/abs/1904.05049>`__.
    Parameters
    ----------
    name : None or str
        A unique layer name.
    Notes
    -----
    - Use this layer to concat two tensor.
    - The height and width of one tensor should be twice of the other tensor.
    Examples
    --------
    With TensorLayer
    >>> net = tl.layers.Input([8, 28, 28, 32], name='input')
    >>> octconv2d = tl.layers.OctConv2dIn(name='octconv2din_1')
    >>> print(octconv2d)
    >>> output shape : [(8, 28, 28, 32),(8, 14, 14, 32)]
    >>> octconv2dl = tl.layers.OctConv2dLowOut(name='octconv2dlo_1')(octconv2d)
    >>> octconv2dh = tl.layers.OctConv2dHighOut(name='octconv2dho_1')(octconv2d)
    >>> octconv2 = tl.layers.OctConv2dConcat(name='octconv2dcat_1')([octconv2dh,octconv2dl])
    >>> print(octconv2d)
    >>> output shape : [(8, 28, 28, 32),(8, 14, 14, 32)]
    """

    def __init__(
            self,
            name=None,  # 'cnn2d_layer',
    ):
        super().__init__(name)
        self.build(None)
        self._built = True

        logging.info(
            "OctConv2dConcat %s: " % (
                self.name,
            )
        )

    def __repr__(self):

        s = ('{classname}(')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs):
        pass

    def forward(self, inputs):
        if inputs[0].shape[1]>inputs[1].shape[1]:
            outputs=[inputs[0],inputs[1]]
        else:
            outputs = [inputs[1], inputs[0]]
        return outputs