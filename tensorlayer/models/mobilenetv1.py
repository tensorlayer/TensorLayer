#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
MobileNet for ImageNet.
"""

import os

import tensorflow as tf

from tensorlayer import tl_logging as logging

from tensorlayer.layers import Layer
from tensorlayer.layers import BatchNormLayer
from tensorlayer.layers import Conv2d
from tensorlayer.layers import DepthwiseConv2d
from tensorlayer.layers import FlattenLayer
from tensorlayer.layers import GlobalMeanPool2d
from tensorlayer.layers import InputLayer
from tensorlayer.layers import ReshapeLayer

from tensorlayer.files import maybe_download_and_extract, assign_params, load_npz

__all__ = [
    'MobileNetV1',
]


class MobileNetV1(Layer):
    """Pre-trained MobileNetV1 model.

    Parameters
    ------------
    x : placeholder
        shape [None, 224, 224, 3], value range [0, 1].
    end_with : str
        The end point of the model [conv, depth1, depth2 ... depth13, globalmeanpool, out]. Default ``out`` i.e. the whole model.
    is_train : boolean
        Whether the model is used for training i.e. enable dropout.
    reuse : boolean
        Whether to reuse the model.

    Examples
    ---------
    Classify ImageNet classes, see `tutorial_models_mobilenetv1.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_mobilenetv1.py>`__

    >>> x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> # get the whole model
    >>> net = tl.models.MobileNetV1(x)
    >>> # restore pre-trained parameters
    >>> sess = tf.InteractiveSession()
    >>> net.restore_params(sess)
    >>> # use for inferencing
    >>> probs = tf.nn.softmax(net.outputs)

    Extract features and Train a classifier with 100 classes

    >>> x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> # get model without the last layer
    >>> cnn = tl.models.MobileNetV1(x, end_with='reshape')
    >>> # add one more layer
    >>> net = Conv2d(cnn, 100, (1, 1), (1, 1), name='out')
    >>> net = FlattenLayer(net, name='flatten')
    >>> # initialize all parameters
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>> # restore pre-trained parameters
    >>> cnn.restore_params(sess)
    >>> # train your own classifier (only update the last layer)
    >>> train_params = tl.layers.get_variables_with_name('out')

    Reuse model

    >>> x1 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> x2 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> # get model without the last layer
    >>> net1 = tl.models.MobileNetV1(x1, end_with='reshape')
    >>> # reuse the parameters with different input
    >>> net2 = tl.models.MobileNetV1(x2, end_with='reshape', reuse=True)
    >>> # restore pre-trained parameters (as they share parameters, we don’t need to restore net2)
    >>> sess = tf.InteractiveSession()
    >>> net1.restore_params(sess)

    """

    def __init__(self, x, end_with='out', is_train=False, reuse=None):

        self.net = self.mobilenetv1(x, end_with, is_train, reuse)

        self.outputs = self.net.outputs

        self.all_params = list(self.net.all_params)
        self.all_layers = list(self.net.all_layers)
        self.all_drop = dict(self.net.all_drop)

        self.print_layers = self.net.print_layers
        self.print_params = self.net.print_params

    # @classmethod
    def mobilenetv1(self, x, end_with='out', is_train=False, reuse=None):
        with tf.variable_scope("mobilenetv1", reuse=reuse):
            n = InputLayer(x)
            n = self.conv_block(n, 32, strides=(2, 2), is_train=is_train, name="conv")
            if end_with in n.outputs.name: return n
            n = self.depthwise_conv_block(n, 64, is_train=is_train, name="depth1")
            if end_with in n.outputs.name: return n

            n = self.depthwise_conv_block(n, 128, strides=(2, 2), is_train=is_train, name="depth2")
            if end_with in n.outputs.name: return n
            n = self.depthwise_conv_block(n, 128, is_train=is_train, name="depth3")
            if end_with in n.outputs.name: return n

            n = self.depthwise_conv_block(n, 256, strides=(2, 2), is_train=is_train, name="depth4")
            if end_with in n.outputs.name: return n
            n = self.depthwise_conv_block(n, 256, is_train=is_train, name="depth5")
            if end_with in n.outputs.name: return n

            n = self.depthwise_conv_block(n, 512, strides=(2, 2), is_train=is_train, name="depth6")
            if end_with in n.outputs.name: return n
            n = self.depthwise_conv_block(n, 512, is_train=is_train, name="depth7")
            if end_with in n.outputs.name: return n
            n = self.depthwise_conv_block(n, 512, is_train=is_train, name="depth8")
            if end_with in n.outputs.name: return n
            n = self.depthwise_conv_block(n, 512, is_train=is_train, name="depth9")
            if end_with in n.outputs.name: return n
            n = self.depthwise_conv_block(n, 512, is_train=is_train, name="depth10")
            if end_with in n.outputs.name: return n
            n = self.depthwise_conv_block(n, 512, is_train=is_train, name="depth11")
            if end_with in n.outputs.name: return n

            n = self.depthwise_conv_block(n, 1024, strides=(2, 2), is_train=is_train, name="depth12")
            if end_with in n.outputs.name: return n
            n = self.depthwise_conv_block(n, 1024, is_train=is_train, name="depth13")
            if end_with in n.outputs.name: return n

            n = GlobalMeanPool2d(n, name='globalmeanpool')
            if end_with in n.outputs.name: return n
            # n = DropoutLayer(n, 1-1e-3, True, is_train, name='drop')
            # n = DenseLayer(n, 1000, name='output')   # equal
            n = ReshapeLayer(n, [-1, 1, 1, 1024], name='reshape')
            if end_with in n.outputs.name: return n
            n = Conv2d(n, 1000, (1, 1), (1, 1), name='out')
            n = FlattenLayer(n, name='flatten')
            if end_with == 'out': return n

            raise Exception("end_with : conv, depth1, depth2 ... depth13, globalmeanpool, out")

    @classmethod
    def conv_block(cls, n, n_filter, filter_size=(3, 3), strides=(1, 1), is_train=False, name='conv_block'):
        # ref: https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py
        with tf.variable_scope(name):
            n = Conv2d(n, n_filter, filter_size, strides, b_init=None, name='conv')
            n = BatchNormLayer(n, act=tf.nn.relu6, is_train=is_train, name='batchnorm')
        return n

    @classmethod
    def depthwise_conv_block(cls, n, n_filter, strides=(1, 1), is_train=False, name="depth_block"):
        with tf.variable_scope(name):
            n = DepthwiseConv2d(n, (3, 3), strides, b_init=None, name='depthwise')
            n = BatchNormLayer(n, act=tf.nn.relu6, is_train=is_train, name='batchnorm1')
            n = Conv2d(n, n_filter, (1, 1), (1, 1), b_init=None, name='conv')
            n = BatchNormLayer(n, act=tf.nn.relu6, is_train=is_train, name='batchnorm2')
        return n

    def restore_params(self, sess, path='models'):
        logging.info("Restore pre-trained parameters")
        maybe_download_and_extract(
            'mobilenet.npz', path, 'https://github.com/tensorlayer/pretrained-models/raw/master/models/',
            expected_bytes=25600116
        )  # ls -al
        params = load_npz(name=os.path.join(path, 'mobilenet.npz'))
        assign_params(sess, params[:len(self.net.all_params)], self.net)
        del params
