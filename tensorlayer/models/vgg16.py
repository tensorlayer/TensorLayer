# -*- coding: utf-8 -*-
"""
VGG-16 for ImageNet.

Introduction
----------------
VGG is a convolutional neural network model proposed by K. Simonyan and A. Zisserman
from the University of Oxford in the paper “Very Deep Convolutional Networks for
Large-Scale Image Recognition”  . The model achieves 92.7% top-5 test accuracy in ImageNet,
which is a dataset of over 14 million images belonging to 1000 classes.

Download Pre-trained Model
----------------------------
- Model weights in this example - vgg16_weights.npz : http://www.cs.toronto.edu/~frossard/post/vgg16/
- Caffe VGG 16 model : https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
- Tool to convert the Caffe models to TensorFlow's : https://github.com/ethereon/caffe-tensorflow

Note
------
- For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.
- When feeding other images to the model be sure to properly resize or crop them
beforehand. Distorted images might end up being misclassified. One way of safely
feeding images of multiple sizes is by doing center cropping.
"""

import tensorflow as tf

from ..layers import (Conv2d, Conv2dLayer, DenseLayer, FlattenLayer, InputLayer, MaxPool2d, PoolLayer)


class Vgg16Base(object):
    """The vgg16 model."""

    @staticmethod
    def conv_layers(net_in):
        with tf.name_scope('preprocess'):
            # Notice that we include a preprocessing layer that takes the RGB image
            # with pixels values in the range of 0-255 and subtracts the mean image
            # values (calculated over the entire ImageNet training set).
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            net_in.outputs = net_in.outputs - mean

        # conv1
        network = Conv2dLayer(
            net_in,
            act=tf.nn.relu,
            shape=[3, 3, 3, 64],  # 64 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv1_1')
        network = Conv2dLayer(
            network,
            act=tf.nn.relu,
            shape=[3, 3, 64, 64],  # 64 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv1_2')
        network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='pool1')

        # conv2
        network = Conv2dLayer(
            network,
            act=tf.nn.relu,
            shape=[3, 3, 64, 128],  # 128 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv2_1')
        network = Conv2dLayer(
            network,
            act=tf.nn.relu,
            shape=[3, 3, 128, 128],  # 128 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv2_2')
        network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='pool2')

        # conv3
        network = Conv2dLayer(
            network,
            act=tf.nn.relu,
            shape=[3, 3, 128, 256],  # 256 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv3_1')
        network = Conv2dLayer(
            network,
            act=tf.nn.relu,
            shape=[3, 3, 256, 256],  # 256 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv3_2')
        network = Conv2dLayer(
            network,
            act=tf.nn.relu,
            shape=[3, 3, 256, 256],  # 256 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv3_3')
        network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='pool3')

        # conv4
        network = Conv2dLayer(
            network,
            act=tf.nn.relu,
            shape=[3, 3, 256, 512],  # 512 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv4_1')
        network = Conv2dLayer(
            network,
            act=tf.nn.relu,
            shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv4_2')
        network = Conv2dLayer(
            network,
            act=tf.nn.relu,
            shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv4_3')
        network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='pool4')

        # conv5
        network = Conv2dLayer(
            network,
            act=tf.nn.relu,
            shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv5_1')
        network = Conv2dLayer(
            network,
            act=tf.nn.relu,
            shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv5_2')
        network = Conv2dLayer(
            network,
            act=tf.nn.relu,
            shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv5_3')
        network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool, name='pool5')
        return network

    @staticmethod
    def conv_layers_simple_api(net_in):
        with tf.name_scope('preprocess'):
            # Notice that we include a preprocessing layer that takes the RGB image
            # with pixels values in the range of 0-255 and subtracts the mean image
            # values (calculated over the entire ImageNet training set).
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            net_in.outputs = net_in.outputs - mean

        # conv1
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')

        # conv2
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')

        # conv3
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')

        # conv4
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')

        # conv5
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')
        return network

    @staticmethod
    def fc_layers(net):
        network = FlattenLayer(net, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1_relu')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2_relu')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc3_relu')
        return network


class Vgg16Simple(Vgg16Base):
    def __call__(self, x):
        net_in = InputLayer(x, name='input')
        net_cnn = Vgg16Base.conv_layers_simple_api(net_in)  # simplified CNN APIs
        network = Vgg16Base.fc_layers(net_cnn)
        return network


class Vgg16Professional(Vgg16Base):
    def __call__(self, x):
        net_in = InputLayer(x, name='input')
        net_cnn = Vgg16Base.conv_layers(net_in)
        network = Vgg16Base.fc_layers(net_cnn)
        return network
