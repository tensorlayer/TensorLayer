#! /usr/bin/python
# -*- coding: utf-8 -*-
"""ResNet for ImageNet.

# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)

"""

import os

import tensorflow as tf
from tensorlayer import logging
from tensorlayer.files import (assign_weights, load_npz, maybe_download_and_extract)
from tensorlayer.layers import (BatchNorm, Conv2d, Elementwise, GlobalMeanPool2d, MaxPool2d, Input, Dense)
from tensorlayer.models import Model

__all__ = [
    'ResNet50',
]


def identity_block(input, kernel_size, n_filters, stage, block):
    """The identity block where there is no conv layer at shortcut.

    Parameters
    ----------
    input : tf tensor
        Input tensor from above layer.
    kernel_size : int
        The kernel size of middle conv layer at main path.
    n_filters : list of integers
        The numbers of filters for 3 conv layer at main path.
    stage : int
        Current stage label.
    block : str
        Current block label.

    Returns
    -------
        Output tensor of this block.

    """
    filters1, filters2, filters3 = n_filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2d(filters1, (1, 1), W_init=tf.initializers.he_normal(), name=conv_name_base + '2a')(input)
    x = BatchNorm(name=bn_name_base + '2a', act='relu')(x)

    ks = (kernel_size, kernel_size)
    x = Conv2d(filters2, ks, padding='SAME', W_init=tf.initializers.he_normal(), name=conv_name_base + '2b')(x)
    x = BatchNorm(name=bn_name_base + '2b', act='relu')(x)

    x = Conv2d(filters3, (1, 1), W_init=tf.initializers.he_normal(), name=conv_name_base + '2c')(x)
    x = BatchNorm(name=bn_name_base + '2c')(x)

    x = Elementwise(tf.add, act='relu')([x, input])
    return x


def conv_block(input, kernel_size, n_filters, stage, block, strides=(2, 2)):
    """The conv block where there is a conv layer at shortcut.

    Parameters
    ----------
    input : tf tensor
        Input tensor from above layer.
    kernel_size : int
        The kernel size of middle conv layer at main path.
    n_filters : list of integers
        The numbers of filters for 3 conv layer at main path.
    stage : int
        Current stage label.
    block : str
        Current block label.
    strides : tuple
        Strides for the first conv layer in the block.

    Returns
    -------
        Output tensor of this block.

    """
    filters1, filters2, filters3 = n_filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2d(filters1, (1, 1), strides=strides, W_init=tf.initializers.he_normal(), name=conv_name_base + '2a')(input)
    x = BatchNorm(name=bn_name_base + '2a', act='relu')(x)

    ks = (kernel_size, kernel_size)
    x = Conv2d(filters2, ks, padding='SAME', W_init=tf.initializers.he_normal(), name=conv_name_base + '2b')(x)
    x = BatchNorm(name=bn_name_base + '2b', act='relu')(x)

    x = Conv2d(filters3, (1, 1), W_init=tf.initializers.he_normal(), name=conv_name_base + '2c')(x)
    x = BatchNorm(name=bn_name_base + '2c')(x)

    shortcut = Conv2d(filters3, (1, 1), strides=strides, W_init=tf.initializers.he_normal(),
                      name=conv_name_base + '1')(input)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut)

    x = Elementwise(tf.add, act='relu')([x, shortcut])
    return x


block_names = ['2a', '2b', '2c', '3a', '3b', '3c', '3d', '4a', '4b', '4c', '4d', '4e', '4f', '5a', '5b', '5c'
              ] + ['avg_pool', 'fc1000']
block_filters = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]


def ResNet50(pretrained=False, end_with='fc1000', n_classes=1000, name=None):
    """Pre-trained MobileNetV1 model (static mode). Input shape [?, 224, 224, 3].
    To use pretrained model, input should be in BGR format and subtracted from ImageNet mean [103.939, 116.779, 123.68].

    Parameters
    ----------
    pretrained : boolean
        Whether to load pretrained weights. Default False.
    end_with : str
        The end point of the model [conv, depth1, depth2 ... depth13, globalmeanpool, out].
        Default ``out`` i.e. the whole model.
    n_classes : int
        Number of classes in final prediction.
    name : None or str
        Name for this model.

    Examples
    ---------
    Classify ImageNet classes, see `tutorial_models_resnet50.py`

    >>> # get the whole model with pretrained weights
    >>> resnet = tl.models.ResNet50(pretrained=True)
    >>> # use for inferencing
    >>> output = resnet(img1, is_train=False)
    >>> prob = tf.nn.softmax(output)[0].numpy()

    Extract the features before fc layer
    >>> resnet = tl.models.ResNet50(pretrained=True, end_with='5c')
    >>> output = resnet(img1, is_train=False)

    Returns
    -------
        ResNet50 model.

    """
    ni = Input([None, 224, 224, 3], name="input")
    n = Conv2d(64, (7, 7), strides=(2, 2), padding='SAME', W_init=tf.initializers.he_normal(), name='conv1')(ni)
    n = BatchNorm(name='bn_conv1', act='relu')(n)
    n = MaxPool2d((3, 3), strides=(2, 2), name='max_pool1')(n)

    for i, block_name in enumerate(block_names):
        if len(block_name) == 2:
            stage = int(block_name[0])
            block = block_name[1]
            if block == 'a':
                strides = (1, 1) if stage == 2 else (2, 2)
                n = conv_block(n, 3, block_filters[stage - 2], stage=stage, block=block, strides=strides)
            else:
                n = identity_block(n, 3, block_filters[stage - 2], stage=stage, block=block)
        elif block_name == 'avg_pool':
            n = GlobalMeanPool2d(name='avg_pool')(n)
        elif block_name == 'fc1000':
            n = Dense(n_classes, name='fc1000')(n)

        if block_name == end_with:
            break

    network = Model(inputs=ni, outputs=n, name=name)

    if pretrained:
        restore_params(network)

    return network


def restore_params(network, path='models'):
    logging.info("Restore pre-trained parameters")
    maybe_download_and_extract(
        'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
        path,
        'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/',
    )  # ls -al
    try:
        import h5py
    except Exception:
        raise ImportError('h5py not imported')

    f = h5py.File(os.path.join(path, 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'), 'r')

    for layer in network.all_layers:
        if len(layer.all_weights) == 0:
            continue
        w_names = list(f[layer.name])
        params = [f[layer.name][n][:] for n in w_names]
        # if 'bn' in layer.name:
        #     params = [x.reshape(1, 1, 1, -1) for x in params]
        assign_weights(params, layer)
        del params

    f.close()
