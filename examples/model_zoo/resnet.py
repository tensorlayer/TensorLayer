#! /usr/bin/python
# -*- coding: utf-8 -*-
"""ResNet for ImageNet.

# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)

"""

import os

import tensorlayer as tl

from tensorlayer import logging
from tensorlayer.files import (assign_weights, maybe_download_and_extract)
from tensorlayer.layers import (BatchNorm, Conv2d, Dense, Elementwise, GlobalMeanPool2d, Input, MaxPool2d)
from tensorlayer.layers import Module, SequentialLayer

__all__ = [
    'ResNet50',
]

block_names = ['2a', '2b', '2c', '3a', '3b', '3c', '3d', '4a', '4b', '4c', '4d', '4e', '4f', '5a', '5b', '5c'
              ] + ['avg_pool', 'fc1000']
block_filters = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
in_channels_conv = [64, 256, 512, 1024]
in_channels_identity = [256, 512, 1024, 2048]
henorm = tl.initializers.he_normal()

class identity_block(Module):
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
    def __init__(self, kernel_size, n_filters, stage, block):
        super(identity_block, self).__init__()
        filters1, filters2, filters3 = n_filters
        _in_channels = in_channels_identity[stage-2]
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv1 = Conv2d(filters1, (1, 1), W_init=henorm, name=conv_name_base + '2a', in_channels=_in_channels)
        self.bn1 = BatchNorm(name=bn_name_base + '2a', act='relu', num_features=filters1)

        ks = (kernel_size, kernel_size)
        self.conv2 = Conv2d(filters2, ks, padding='SAME', W_init=henorm, name=conv_name_base + '2b', in_channels=filters1)
        self.bn2 = BatchNorm(name=bn_name_base + '2b', act='relu', num_features=filters2)

        self.conv3 = Conv2d(filters3, (1, 1), W_init=henorm, name=conv_name_base + '2c', in_channels=filters2)
        self.bn3 = BatchNorm(name=bn_name_base + '2c', num_features=filters3)

        self.add = Elementwise(tl.add, act='relu')

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        result = self.add([output, inputs])
        return result


class conv_block(Module):
    def __init__(self, kernel_size, n_filters, stage, block, strides=(2, 2)):
        super(conv_block, self).__init__()
        filters1, filters2, filters3 = n_filters
        _in_channels = in_channels_conv[stage-2]
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        self.conv1 = Conv2d(filters1, (1, 1), strides=strides, W_init=henorm, name=conv_name_base + '2a', in_channels=_in_channels)
        self.bn1 = BatchNorm(name=bn_name_base + '2a', act='relu', num_features=filters1)

        ks = (kernel_size, kernel_size)
        self.conv2 = Conv2d(filters2, ks, padding='SAME', W_init=henorm, name=conv_name_base + '2b', in_channels=filters1)
        self.bn2 = BatchNorm(name=bn_name_base + '2b', act='relu', num_features=filters2)

        self.conv3 = Conv2d(filters3, (1, 1), W_init=henorm, name=conv_name_base + '2c', in_channels=filters2)
        self.bn3 = BatchNorm(name=bn_name_base + '2c', num_features=filters3)

        self.shortcut_conv = Conv2d(filters3, (1, 1), strides=strides, W_init=henorm, name=conv_name_base + '1', in_channels=_in_channels)
        self.shortcut_bn = BatchNorm(name=bn_name_base + '1', num_features=filters3)

        self.add = Elementwise(tl.add, act='relu')

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.conv3(output)
        output = self.bn3(output)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut)

        result = self.add([output, shortcut])
        return result


class ResNet50_model(Module):
    def __init__(self, end_with='fc1000', n_classes=1000):
        super(ResNet50_model, self).__init__()
        self.end_with = end_with
        self.n_classes = n_classes
        self.conv1 = Conv2d(64, (7, 7), in_channels=3, strides=(2, 2), padding='SAME', W_init=henorm, name='conv1')
        self.bn_conv1 = BatchNorm(name='bn_conv1', act="relu", num_features=64)
        self.max_pool1 = MaxPool2d((3, 3), strides=(2, 2), name='max_pool1')
        self.res_layer = self.make_layer()

    def forward(self, inputs):
        z = self.conv1(inputs)
        z = self.bn_conv1(z)
        z = self.max_pool1(z)
        z = self.res_layer(z)
        return z

    def make_layer(self):
        layer_list = []
        for i, block_name in enumerate(block_names):
            if len(block_name) == 2:
                stage = int(block_name[0])
                block = block_name[1]
                if block == 'a':
                    strides = (1, 1) if stage == 2 else (2, 2)
                    layer_list.append(conv_block(3, block_filters[stage - 2], stage=stage, block=block, strides=strides))
                else:
                    layer_list.append(identity_block(3, block_filters[stage - 2], stage=stage, block=block))
            elif block_name == 'avg_pool':
                layer_list.append(GlobalMeanPool2d(name='avg_pool'))
            elif block_name == 'fc1000':
                layer_list.append(Dense(self.n_classes, name='fc1000', in_channels=2048))

            if block_name == self.end_with:
                break
        return SequentialLayer(layer_list)


def ResNet50(pretrained=False, end_with='fc1000', n_classes=1000):
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
    TODO Modify the usage example according to the model storage location
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

    network = ResNet50_model(end_with=end_with, n_classes=n_classes)

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
