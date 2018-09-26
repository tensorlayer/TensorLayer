#! /usr/bin/python
# -*- coding: utf-8 -*-
"""SqueezeNet for ImageNet."""

import os

import tensorflow as tf

from tensorlayer import logging

from tensorlayer.layers import Layer
from tensorlayer.layers import Conv2d
from tensorlayer.layers import InputLayer
from tensorlayer.layers import MaxPool2d
from tensorlayer.layers import ConcatLayer
from tensorlayer.layers import DropoutLayer
from tensorlayer.layers import GlobalMeanPool2d

from tensorlayer.files import maybe_download_and_extract
from tensorlayer.files import assign_params
from tensorlayer.files import load_npz

__all__ = [
    'SqueezeNetV1',
]


class SqueezeNetV1(Layer):
    """Pre-trained SqueezeNetV1 model.

    Parameters
    ------------
    x : placeholder
        shape [None, 224, 224, 3], value range [0, 255].
    end_with : str
        The end point of the model [input, fire2, fire3 ... fire9, output]. Default ``output`` i.e. the whole model.
    is_train : boolean
        Whether the model is used for training i.e. enable dropout.
    reuse : boolean
        Whether to reuse the model.

    Examples
    ---------
    Classify ImageNet classes, see `tutorial_models_squeezenetv1.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_squeezenetv1.py>`__

    >>> x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> # get the whole model
    >>> net = tl.models.SqueezeNetV1(x)
    >>> # restore pre-trained parameters
    >>> sess = tf.InteractiveSession()
    >>> net.restore_params(sess)
    >>> # use for inferencing
    >>> probs = tf.nn.softmax(net.outputs)

    Extract features and Train a classifier with 100 classes

    >>> x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> # get model without the last layer
    >>> cnn = tl.models.SqueezeNetV1(x, end_with='fire9')
    >>> # add one more layer
    >>> net = Conv2d(cnn, 100, (1, 1), (1, 1), padding='VALID', name='output')
    >>> net = GlobalMeanPool2d(net)
    >>> # initialize all parameters
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>> # restore pre-trained parameters
    >>> cnn.restore_params(sess)
    >>> # train your own classifier (only update the last layer)
    >>> train_params = tl.layers.get_variables_with_name('output')

    Reuse model

    >>> x1 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> x2 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> # get model without the last layer
    >>> net1 = tl.models.SqueezeNetV1(x1, end_with='fire9')
    >>> # reuse the parameters with different input
    >>> net2 = tl.models.SqueezeNetV1(x2, end_with='fire9', reuse=True)
    >>> # restore pre-trained parameters (as they share parameters, we don’t need to restore net2)
    >>> sess = tf.InteractiveSession()
    >>> net1.restore_params(sess)

    """

    def __init__(self, x, end_with='output', is_train=False, reuse=None):

        self.net = self.squeezenetv1(x, end_with, is_train, reuse)

        self.outputs = self.net.outputs

        self.all_params = list(self.net.all_params)
        self.all_layers = list(self.net.all_layers)
        self.all_drop = dict(self.net.all_drop)
        self.print_layers = self.net.print_layers
        self.print_params = self.net.print_params

    @classmethod
    def squeezenetv1(cls, x, end_with='output', is_train=False, reuse=None):
        with tf.variable_scope("squeezenetv1", reuse=reuse):
            with tf.variable_scope("input"):
                n = InputLayer(x)
                # n = Conv2d(n, 96, (7,7),(2,2),tf.nn.relu,'SAME',name='conv1')
                n = Conv2d(n, 64, (3, 3), (2, 2), tf.nn.relu, 'SAME', name='conv1')
                n = MaxPool2d(n, (3, 3), (2, 2), 'VALID', name='max')
            if end_with in n.outputs.name:
                return n

            with tf.variable_scope("fire2"):
                n = Conv2d(n, 16, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')
                n1 = Conv2d(n, 64, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')
                n2 = Conv2d(n, 64, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')
                n = ConcatLayer([n1, n2], -1, name='concat')
            if end_with in n.outputs.name:
                return n

            with tf.variable_scope("fire3"):
                n = Conv2d(n, 16, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')
                n1 = Conv2d(n, 64, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')
                n2 = Conv2d(n, 64, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')
                n = ConcatLayer([n1, n2], -1, name='concat')
                n = MaxPool2d(n, (3, 3), (2, 2), 'VALID', name='max')
            if end_with in n.outputs.name:
                return n

            with tf.variable_scope("fire4"):
                n = Conv2d(n, 32, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')
                n1 = Conv2d(n, 128, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')
                n2 = Conv2d(n, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')
                n = ConcatLayer([n1, n2], -1, name='concat')
            if end_with in n.outputs.name:
                return n

            with tf.variable_scope("fire5"):
                n = Conv2d(n, 32, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')
                n1 = Conv2d(n, 128, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')
                n2 = Conv2d(n, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')
                n = ConcatLayer([n1, n2], -1, name='concat')
                n = MaxPool2d(n, (3, 3), (2, 2), 'VALID', name='max')
            if end_with in n.outputs.name:
                return n

            with tf.variable_scope("fire6"):
                n = Conv2d(n, 48, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')
                n1 = Conv2d(n, 192, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')
                n2 = Conv2d(n, 192, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')
                n = ConcatLayer([n1, n2], -1, name='concat')
            if end_with in n.outputs.name:
                return n

            with tf.variable_scope("fire7"):
                n = Conv2d(n, 48, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')
                n1 = Conv2d(n, 192, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')
                n2 = Conv2d(n, 192, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')
                n = ConcatLayer([n1, n2], -1, name='concat')
            if end_with in n.outputs.name:
                return n

            with tf.variable_scope("fire8"):
                n = Conv2d(n, 64, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')
                n1 = Conv2d(n, 256, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')
                n2 = Conv2d(n, 256, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')
                n = ConcatLayer([n1, n2], -1, name='concat')
            if end_with in n.outputs.name:
                return n

            with tf.variable_scope("fire9"):
                n = Conv2d(n, 64, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')
                n1 = Conv2d(n, 256, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')
                n2 = Conv2d(n, 256, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')
                n = ConcatLayer([n1, n2], -1, name='concat')
            if end_with in n.outputs.name:
                return n

            with tf.variable_scope("output"):
                n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=is_train, name='drop1')
                n = Conv2d(n, 1000, (1, 1), (1, 1), padding='VALID', name='conv10')  # 13, 13, 1000
                n = GlobalMeanPool2d(n)
            if end_with in n.outputs.name:
                return n

            raise Exception("end_with : input, fire2, fire3 ... fire9, output")

    def restore_params(self, sess, path='models'):
        logging.info("Restore pre-trained parameters")
        maybe_download_and_extract(
            'squeezenet.npz', path, 'https://github.com/tensorlayer/pretrained-models/raw/master/models/',
            expected_bytes=7405613
        )  # ls -al
        params = load_npz(name=os.path.join(path, 'squeezenet.npz'))
        assign_params(sess, params[:len(self.net.all_params)], self.net)
        del params
