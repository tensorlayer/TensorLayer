# -*- coding: utf-8 -*-
"""
VGG-19 for ImageNet.

Introduction
----------------
VGG is a convolutional neural network model proposed by K. Simonyan and A. Zisserman
from the University of Oxford in the paper “Very Deep Convolutional Networks for
Large-Scale Image Recognition”  . The model achieves 92.7% top-5 test accuracy in ImageNet,
which is a dataset of over 14 million images belonging to 1000 classes.

Download Pre-trained Model
----------------------------
- Model weights in this example - vgg19.npz : https://github.com/machrisaa/tensorflow-vgg
- Caffe VGG 19 model : https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
- Tool to convert the Caffe models to TensorFlow's : https://github.com/ethereon/caffe-tensorflow

Note
------
- For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.
- When feeding other images to the model be sure to properly resize or crop them
beforehand. Distorted images might end up being misclassified. One way of safely
feeding images of multiple sizes is by doing center cropping.
"""

import os
import numpy as np
import tensorflow as tf

from tensorlayer import logging

from tensorlayer.layers import Conv2d
from tensorlayer.layers import DenseLayer
from tensorlayer.layers import FlattenLayer
from tensorlayer.layers import InputLayer
from tensorlayer.layers import MaxPool2d

from tensorlayer.files import maybe_download_and_extract
from tensorlayer.files import assign_params

__all__ = [
    'VGG19',
]


class VGG19Base(object):
    """The VGG19 model."""

    @staticmethod
    def vgg19_simple_api(net_in, end_with):
        with tf.name_scope('preprocess'):
            # Notice that we include a preprocessing layer that takes the RGB image
            # with pixels values in the range of 0-255 and subtracts the mean image
            # values (calculated over the entire ImageNet training set).

            # # Rescale the input tensor with pixels values in the range of 0-255
            # net_in.outputs = net_in.outputs * 255.0
            # red, green, blue = tf.split(net_in.outputs, 3, 3)
            VGG_MEAN = [103.939, 116.779, 123.68]
            #
            # bgr = tf.concat([
            #     blue - VGG_MEAN[0],
            #     green - VGG_MEAN[1],
            #     red - VGG_MEAN[2],
            # ], axis=3)
            # # mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            # # net_in.outputs = net_in.outputs - mean
            # net_in.outputs = bgr
            rgb = net_in.outputs
            rgb_scaled = rgb * 255.0
            # Convert RGB to BGR
            red, green, blue = tf.split(rgb_scaled, 3, 3)

            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)

            net_in.outputs = bgr

        layers = [
            # conv1
            lambda net: Conv2d(
                net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1'
            ),
            lambda net: Conv2d(
                net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2'
            ),
            lambda net: MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1'),

            # conv2
            lambda net: Conv2d(
                net, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1'
            ),
            lambda net: Conv2d(
                net, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2'
            ),
            lambda net: MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2'),

            # conv3
            lambda net: Conv2d(
                net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1'
            ),
            lambda net: Conv2d(
                net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2'
            ),
            lambda net: Conv2d(
                net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3'
            ),
            lambda net: Conv2d(
                net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4'
            ),
            lambda net: MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3'),

            # conv4
            lambda net: Conv2d(
                net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1'
            ),
            lambda net: Conv2d(
                net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2'
            ),
            lambda net: Conv2d(
                net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3'
            ),
            lambda net: Conv2d(
                net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4'
            ),
            lambda net: MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4'),

            # conv5
            lambda net: Conv2d(
                net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1'
            ),
            lambda net: Conv2d(
                net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2'
            ),
            lambda net: Conv2d(
                net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3'
            ),
            lambda net: Conv2d(
                net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4'
            ),
            lambda net: MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5'),
            lambda net: FlattenLayer(net, name='flatten'),
            lambda net: DenseLayer(net, n_units=4096, act=tf.nn.relu, name='fc1_relu'),
            lambda net: DenseLayer(net, n_units=4096, act=tf.nn.relu, name='fc2_relu'),
            lambda net: DenseLayer(net, n_units=1000, act=tf.identity, name='fc3_relu'),
        ]
        net = net_in
        for l in layers:
            net = l(net)
            # if end_with in net.name:
            if net.name.endswith(end_with):
                return net

        raise Exception("unknown layer name (end_with): {}".format(end_with))

    def restore_params(self, sess):
        # For no pre-trained model link

        # logging.info("Restore pre-trained parameters")
        # vgg19_npy_path = os.path.join('models', 'vgg19.npy')
        # if not os.path.isfile(vgg19_npy_path):
        #     print("Please download vgg19.npy from : https://github.com/machrisaa/tensorflow-vgg")
        #     exit()
        # npz = np.load(vgg19_npy_path, encoding='latin1').item()

        # For existing pre-trained model link
        logging.info("Restore pre-trained parameters")
        maybe_download_and_extract(
            'vgg19.npy', 'models',
            'https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/',
            expected_bytes=574670860
        )
        vgg19_npy_path = os.path.join('models', 'vgg19.npy')
        npz = np.load(vgg19_npy_path, encoding='latin1').item()

        params = []
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
            if len(self.all_params) == len(params):
                break

        print("Restoring model from npz file")
        assign_params(sess, params, self.net)
        del params


class VGG19(VGG19Base):
    """Pre-trained VGG-19 model.

    Parameters
    ------------
    x : placeholder
        shape [None, 224, 224, 3], value range [0, 1].
    end_with : str
        The end point of the model. Default ``fc3_relu`` i.e. the whole model.
    reuse : boolean
        Whether to reuse the model.

    Examples
    ---------
    Classify ImageNet classes with VGG19, see `tutorial_models_vgg19.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_vgg19.py>`__

    >>> x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> # get the whole model
    >>> vgg = tl.models.VGG19(x)
    >>> # restore pre-trained VGG parameters
    >>> sess = tf.InteractiveSession()
    >>> vgg.restore_params(sess)
    >>> # use for inferencing
    >>> probs = tf.nn.softmax(vgg.outputs)

    Extract features with VGG19 and Train a classifier with 100 classes

    >>> x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> # get VGG without the last layer
    >>> vgg = tl.models.VGG19(x, end_with='fc2_relu')
    >>> # add one more layer
    >>> net = tl.layers.DenseLayer(vgg, 100, name='out')
    >>> # initialize all parameters
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>> # restore pre-trained VGG parameters
    >>> vgg.restore_params(sess)
    >>> # train your own classifier (only update the last layer)
    >>> train_params = tl.layers.get_variables_with_name('out')

    Reuse model

    >>> x1 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> x2 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    >>> # get VGG without the last layer
    >>> vgg1 = tl.models.VGG19(x1, end_with='fc2_relu')
    >>> # reuse the parameters of vgg1 with different input
    >>> vgg2 = tl.models.VGG19(x2, end_with='fc2_relu', reuse=True)
    >>> # restore pre-trained VGG parameters (as they share parameters, we don’t need to restore vgg2)
    >>> sess = tf.InteractiveSession()
    >>> vgg1.restore_params(sess)

    """

    def __init__(self, x, end_with='fc3_relu', reuse=None):
        with tf.variable_scope("vgg19", reuse=reuse):
            scope_name = tf.get_variable_scope().name
            self.name = scope_name + '/vgg19' if scope_name else '/vgg19'

            net = InputLayer(x, name='input')
            self.net = VGG19Base.vgg19_simple_api(net, end_with)
            self.outputs = self.net.outputs
            self.all_params = self.net.all_params
            self.all_layers = self.net.all_layers
            self.all_drop = self.net.all_drop
            self.print_layers = self.net.print_layers
            self.print_params = self.net.print_params
