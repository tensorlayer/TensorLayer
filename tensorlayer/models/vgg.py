#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
VGG for ImageNet.

Introduction
----------------
VGG is a convolutional neural network model proposed by K. Simonyan and A. Zisserman
from the University of Oxford in the paper "Very Deep Convolutional Networks for
Large-Scale Image Recognition"  . The model achieves 92.7% top-5 test accuracy in ImageNet,
which is a dataset of over 14 million images belonging to 1000 classes.

Download Pre-trained Model
----------------------------
- Model weights in this example - vgg16_weights.npz : http://www.cs.toronto.edu/~frossard/post/vgg16/
- Model weights in this example - vgg19.npy : https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/
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

import os

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.files import assign_weights, maybe_download_and_extract
from tensorlayer.layers import (BatchNorm, Conv2d, Dense, Flatten, Input, LayerList, MaxPool2d, Lambda)
from tensorlayer.models import Model

__all__ = [
    'VGG',
    'vgg16',
    'vgg19',
    'VGG16',
    'VGG19',
    #    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    #    'vgg19_bn', 'vgg19',
]

layer_names = [
    ['conv1_1', 'conv1_2'], 'pool1', ['conv2_1', 'conv2_2'], 'pool2',
    ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4'], 'pool3', ['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4'], 'pool4',
    ['conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'], 'pool5', 'flatten', 'fc1_relu', 'fc2_relu', 'outputs'
]

cfg = {
    'A': [[64], 'M', [128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'B': [[64, 64], 'M', [128, 128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'D': [
        [64, 64], 'M', [128, 128], 'M', [256, 256, 256], 'M', [512, 512, 512], 'M', [512, 512, 512], 'M', 'F', 'fc1',
        'fc2', 'O'
    ],
    'E': [
        [64, 64], 'M', [128, 128], 'M', [256, 256, 256, 256], 'M', [512, 512, 512, 512], 'M', [512, 512, 512, 512], 'M',
        'F', 'fc1', 'fc2', 'O'
    ],
}

mapped_cfg = {
    'vgg11': 'A',
    'vgg11_bn': 'A',
    'vgg13': 'B',
    'vgg13_bn': 'B',
    'vgg16': 'D',
    'vgg16_bn': 'D',
    'vgg19': 'E',
    'vgg19_bn': 'E'
}

model_urls = {
    'vgg16': 'http://www.cs.toronto.edu/~frossard/vgg16/',
    'vgg19': 'https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/'
}

model_saved_name = {'vgg16': 'vgg16_weights.npz', 'vgg19': 'vgg19.npy'}


class VGG(Model):

    def __init__(self, layer_type, batch_norm=False, end_with='outputs', name=None):
        super(VGG, self).__init__(name=name)
        self.end_with = end_with

        config = cfg[mapped_cfg[layer_type]]
        self.layers = make_layers(config, batch_norm, end_with)

    def forward(self, inputs):
        """
        inputs : tensor
            Shape [None, 224, 224, 3], value range [0, 1].
        """

        inputs = inputs * 255 - np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape([1, 1, 1, 3])

        out = self.layers(inputs)
        return out


def make_layers(config, batch_norm=False, end_with='outputs'):
    layer_list = []
    is_end = False
    for layer_group_idx, layer_group in enumerate(config):
        if isinstance(layer_group, list):
            for idx, layer in enumerate(layer_group):
                layer_name = layer_names[layer_group_idx][idx]
                n_filter = layer
                if idx == 0:
                    if layer_group_idx > 0:
                        in_channels = config[layer_group_idx - 2][-1]
                    else:
                        in_channels = 3
                else:
                    in_channels = layer
                layer_list.append(
                    Conv2d(
                        n_filter=n_filter, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',
                        in_channels=in_channels, name=layer_name
                    )
                )
                if batch_norm:
                    layer_list.append(BatchNorm())
                if layer_name == end_with:
                    is_end = True
                    break
        else:
            layer_name = layer_names[layer_group_idx]
            if layer_group == 'M':
                layer_list.append(MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name=layer_name))
            elif layer_group == 'O':
                layer_list.append(Dense(n_units=1000, in_channels=4096, name=layer_name))
            elif layer_group == 'F':
                layer_list.append(Flatten(name='flatten'))
            elif layer_group == 'fc1':
                layer_list.append(Dense(n_units=4096, act=tf.nn.relu, in_channels=512 * 7 * 7, name=layer_name))
            elif layer_group == 'fc2':
                layer_list.append(Dense(n_units=4096, act=tf.nn.relu, in_channels=4096, name=layer_name))
            if layer_name == end_with:
                is_end = True
        if is_end:
            break
    return LayerList(layer_list)


def restore_model(model, layer_type):
    logging.info("Restore pre-trained weights")
    # download weights
    maybe_download_and_extract(model_saved_name[layer_type], 'models', model_urls[layer_type])
    weights = []
    if layer_type == 'vgg16':
        npz = np.load(os.path.join('models', model_saved_name[layer_type]))
        # get weight list
        for val in sorted(npz.items()):
            logging.info("  Loading weights %s in %s" % (str(val[1].shape), val[0]))
            weights.append(val[1])
            if len(model.all_weights) == len(weights):
                break
    elif layer_type == 'vgg19':
        npz = np.load(os.path.join('models', model_saved_name[layer_type]), encoding='latin1').item()
        # get weight list
        for val in sorted(npz.items()):
            logging.info("  Loading %s in %s" % (str(val[1][0].shape), val[0]))
            logging.info("  Loading %s in %s" % (str(val[1][1].shape), val[0]))
            weights.extend(val[1])
            if len(model.all_weights) == len(weights):
                break
    # assign weight values
    assign_weights(weights, model)
    del weights


def VGG_static(layer_type, batch_norm=False, end_with='outputs', name=None):
    ni = Input([None, 224, 224, 3])
    n = Lambda(
        lambda x: x * 255 - np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape([1, 1, 1, 3]), name='scale'
    )(ni)

    config = cfg[mapped_cfg[layer_type]]
    layers = make_layers(config, batch_norm, end_with)

    nn = layers(n)

    M = Model(inputs=ni, outputs=nn, name=name)
    return M


def vgg16(pretrained=False, end_with='outputs', mode='dynamic', name=None):
    """Pre-trained VGG16 model.

    Parameters
    ------------
    pretrained : boolean
        Whether to load pretrained weights. Default False.
    end_with : str
        The end point of the model. Default ``fc3_relu`` i.e. the whole model.
    mode : str.
        Model building mode, 'dynamic' or 'static'. Default 'dynamic'.
    name : None or str
        A unique layer name.

    Examples
    ---------
    Classify ImageNet classes with VGG16, see `tutorial_models_vgg.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_vgg.py>`__
    With TensorLayer

    >>> # get the whole model, without pre-trained VGG parameters
    >>> vgg = tl.models.vgg16()
    >>> # get the whole model, restore pre-trained VGG parameters
    >>> vgg = tl.models.vgg16(pretrained=True)
    >>> # use for inferencing
    >>> output = vgg(img, is_train=False)
    >>> probs = tf.nn.softmax(output)[0].numpy()

    Extract features with VGG16 and Train a classifier with 100 classes

    >>> # get VGG without the last layer
    >>> cnn = tl.models.vgg16(end_with='fc2_relu', mode='static').as_layer()
    >>> # add one more layer and build a new model
    >>> ni = Input([None, 224, 224, 3], name="inputs")
    >>> nn = cnn(ni)
    >>> nn = tl.layers.Dense(n_units=100, name='out')(nn)
    >>> model = tl.models.Model(inputs=ni, outputs=nn)
    >>> # train your own classifier (only update the last layer)
    >>> train_params = model.get_layer('out').trainable_weights

    Reuse model

    >>> # in dynamic model, we can directly use the same model
    >>> # in static model
    >>> vgg_layer = tl.models.vgg16().as_layer()
    >>> ni_1 = tl.layers.Input([None, 224, 244, 3])
    >>> ni_2 = tl.layers.Input([None, 224, 244, 3])
    >>> a_1 = vgg_layer(ni_1)
    >>> a_2 = vgg_layer(ni_2)
    >>> M = Model(inputs=[ni_1, ni_2], outputs=[a_1, a_2])

    """
    if mode == 'dynamic':
        model = VGG(layer_type='vgg16', batch_norm=False, end_with=end_with, name=name)
    elif mode == 'static':
        model = VGG_static(layer_type='vgg16', batch_norm=False, end_with=end_with, name=name)
    else:
        raise Exception("No such mode %s" % mode)
    if pretrained:
        restore_model(model, layer_type='vgg16')
    return model


def vgg19(pretrained=False, end_with='outputs', mode='dynamic', name=None):
    """Pre-trained VGG19 model.

    Parameters
    ------------
    pretrained : boolean
        Whether to load pretrained weights. Default False.
    end_with : str
        The end point of the model. Default ``fc3_relu`` i.e. the whole model.
    mode : str.
        Model building mode, 'dynamic' or 'static'. Default 'dynamic'.
    name : None or str
        A unique layer name.

    Examples
    ---------
    Classify ImageNet classes with VGG19, see `tutorial_models_vgg.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_vgg.py>`__
    With TensorLayer

    >>> # get the whole model, without pre-trained VGG parameters
    >>> vgg = tl.models.vgg19()
    >>> # get the whole model, restore pre-trained VGG parameters
    >>> vgg = tl.models.vgg19(pretrained=True)
    >>> # use for inferencing
    >>> output = vgg(img, is_train=False)
    >>> probs = tf.nn.softmax(output)[0].numpy()

    Extract features with VGG19 and Train a classifier with 100 classes

    >>> # get VGG without the last layer
    >>> cnn = tl.models.vgg19(end_with='fc2_relu', mode='static').as_layer()
    >>> # add one more layer and build a new model
    >>> ni = Input([None, 224, 224, 3], name="inputs")
    >>> nn = cnn(ni)
    >>> nn = tl.layers.Dense(n_units=100, name='out')(nn)
    >>> model = tl.models.Model(inputs=ni, outputs=nn)
    >>> # train your own classifier (only update the last layer)
    >>> train_params = model.get_layer('out').trainable_weights

    Reuse model

    >>> # in dynamic model, we can directly use the same model
    >>> # in static model
    >>> vgg_layer = tl.models.vgg19().as_layer()
    >>> ni_1 = tl.layers.Input([None, 224, 244, 3])
    >>> ni_2 = tl.layers.Input([None, 224, 244, 3])
    >>> a_1 = vgg_layer(ni_1)
    >>> a_2 = vgg_layer(ni_2)
    >>> M = Model(inputs=[ni_1, ni_2], outputs=[a_1, a_2])

    """
    if mode == 'dynamic':
        model = VGG(layer_type='vgg19', batch_norm=False, end_with=end_with, name=name)
    elif mode == 'static':
        model = VGG_static(layer_type='vgg19', batch_norm=False, end_with=end_with, name=name)
    else:
        raise Exception("No such mode %s" % mode)
    if pretrained:
        restore_model(model, layer_type='vgg19')
    return model


VGG16 = vgg16
VGG19 = vgg19

# models without pretrained parameters
# def vgg11(pretrained=False, end_with='outputs'):
#     model = VGG(layer_type='vgg11', batch_norm=False, end_with=end_with)
#     if pretrained:
#         model.restore_weights()
#     return model
#
#
# def vgg11_bn(pretrained=False, end_with='outputs'):
#     model = VGG(layer_type='vgg11_bn', batch_norm=True, end_with=end_with)
#     if pretrained:
#         model.restore_weights()
#     return model
#
#
# def vgg13(pretrained=False, end_with='outputs'):
#     model = VGG(layer_type='vgg13', batch_norm=False, end_with=end_with)
#     if pretrained:
#         model.restore_weights()
#     return model
#
#
# def vgg13_bn(pretrained=False, end_with='outputs'):
#     model = VGG(layer_type='vgg13_bn', batch_norm=True, end_with=end_with)
#     if pretrained:
#         model.restore_weights()
#     return model
#
#
# def vgg16_bn(pretrained=False, end_with='outputs'):
#     model = VGG(layer_type='vgg16_bn', batch_norm=True, end_with=end_with)
#     if pretrained:
#         model.restore_weights()
#     return model
#
#
# def vgg19_bn(pretrained=False, end_with='outputs'):
#     model = VGG(layer_type='vgg19_bn', batch_norm=True, end_with=end_with)
#     if pretrained:
#         model.restore_weights()
#     return model
