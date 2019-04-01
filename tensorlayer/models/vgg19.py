#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
VGG-19 for ImageNet.

Introduction
----------------
VGG is a convolutional neural network model proposed by K. Simonyan and A. Zisserman
from the University of Oxford in the paper "Very Deep Convolutional Networks for
Large-Scale Image Recognition"  . The model achieves 92.7% top-5 test accuracy in ImageNet,
which is a dataset of over 14 million images belonging to 1000 classes.

Download Pre-trained Model
----------------------------
- Model weights in this example - vgg19.npy : https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/

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

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.files import assign_weights, maybe_download_and_extract
from tensorlayer.layers import (Conv2d, Dense, Flatten, Input, LayerList, MaxPool2d)
from tensorlayer.models import Model

__all__ = [
    'VGG19',
]


class VGG19(Model):
    """Pre-trained VGG-19 model.

    Parameters
    ------------
    end_with : str
        The end point of the model. Default ``fc3_relu`` i.e. the whole model.
    name : None or str
        A unique layer name.

    Examples
    ---------
    Classify ImageNet classes with VGG19, see `tutorial_models_vgg19.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_vgg19.py>`__
    With TensorLayer

    >>> # get the whole model
    >>> vgg = tl.models.VGG19()
    >>> # restore pre-trained VGG parameters
    >>> vgg.restore_weights()
    >>> # use for inferencing
    >>> probs = tf.nn.softmax(vgg.outputs)

    Extract features with VGG19 and Train a classifier with 100 classes

    >>> # get VGG without the last layer
    >>> vgg = tl.models.VGG19(end_with='fc2_relu')
    >>> # add one more layer
    >>> net = tl.layers.Dense(n_units=100, name='out')(vgg)
    >>> # restore pre-trained VGG parameters
    >>> vgg.restore_weights()
    >>> # train your own classifier (only update the last layer)
    >>> train_params = tl.layers.get_variables_with_name('out')

    Reuse model

    >>> # in dynamic mode, we can directly use the same model
    >>> # in static mode
    >>> vgg_layer = tl.models.VGG19.as_layer()
    >>> ni_1 = tl.layers.Input([None, 224, 244, 3])
    >>> ni_2 = tl.layers.Input([None, 224, 244, 3])
    >>> a_1 = vgg_layer(ni_1)
    >>> a_2 = vgg_layer(ni_2)
    >>> M = Model(inputs=[ni_1, ni_2], outputs=[a_1, a_2])

    """

    def __init__(self, end_with='outputs', name=None):
        super(VGG19, self).__init__(name=name)
        self.end_with = end_with

        self.layer_names = [
            'conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
            'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
            'pool5', 'flatten', 'fc1_relu', 'fc2_relu', 'outputs'
        ]
        self.layers = LayerList(
            [
                # conv1
                Conv2d(
                    n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=3,
                    name='conv1_1'
                ),
                Conv2d(
                    n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=64,
                    name='conv1_2'
                ),
                MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1'),

                # conv2
                Conv2d(
                    n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=64,
                    name='conv2_1'
                ),
                Conv2d(
                    n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=128,
                    name='conv2_2'
                ),
                MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2'),

                # conv3
                Conv2d(
                    n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=128,
                    name='conv3_1'
                ),
                Conv2d(
                    n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=256,
                    name='conv3_2'
                ),
                Conv2d(
                    n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=256,
                    name='conv3_3'
                ),
                Conv2d(
                    n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=256,
                    name='conv3_4'
                ),
                MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3'),

                # conv4
                Conv2d(
                    n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=256,
                    name='conv4_1'
                ),
                Conv2d(
                    n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=512,
                    name='conv4_2'
                ),
                Conv2d(
                    n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=512,
                    name='conv4_3'
                ),
                Conv2d(
                    n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=512,
                    name='conv4_4'
                ),
                MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4'),

                # conv5
                Conv2d(
                    n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=512,
                    name='conv5_1'
                ),
                Conv2d(
                    n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=512,
                    name='conv5_2'
                ),
                Conv2d(
                    n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=512,
                    name='conv5_3'
                ),
                Conv2d(
                    n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', in_channels=512,
                    name='conv5_4'
                ),
                MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5'),
                Flatten(name='flatten'),
                Dense(n_units=4096, act=tf.nn.relu, in_channels=512 * 7 * 7, name='fc1_relu'),
                Dense(n_units=4096, act=tf.nn.relu, in_channels=4096, name='fc2_relu'),
                Dense(n_units=1000, in_channels=4096, name='outputs'),
            ][:self.layer_names.index(self.end_with) + 1]
        )

    def forward(self, inputs):
        """
        inputs : tensor
            Shape [None, 224, 224, 3], value range [0, 255] - mean, mean = [123.68, 116.779, 103.939].
        """

        out = self.layers(inputs)
        return out

    def restore_params(self, **kwargs):
        raise Exception("please change restore_params --> restore_weights")

    def restore_weights(self, sess=None):
        logging.info("Restore pre-trained weights")
        ## download weights
        maybe_download_and_extract(
            'vgg19.npy', 'models',
            'https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/',
            expected_bytes=574670860
        )
        vgg19_npy_path = os.path.join('models', 'vgg19.npy')
        npz = np.load(vgg19_npy_path, encoding='latin1').item()

        weights = []
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            weights.extend([W, b])
            if len(self.all_params) == len(weights):
                break
        ## assign weight values
        print(self.weights)
        assign_weights(sess, weights, self)
        del weights
