#! /usr/bin/python
# -*- coding: utf-8 -*-
"""YOLOv4 for MS-COCO.

# Reference:
- [tensorflow-yolov4-tflite](
    https://github.com/hunglc007/tensorflow-yolov4-tflite)

"""

import tensorflow as tf
import numpy as np
import tensorlayer as tl
from tensorlayer.activation import mish
from tensorlayer.layers import Conv2d, MaxPool2d, BatchNorm2d, ZeroPad2d, UpSampling2d, Concat, Input, Elementwise
from tensorlayer.models import Model
from tensorlayer import logging

INPUT_SIZE = 416
weights_url = {'link': 'https://pan.baidu.com/s/1MC1dmEwpxsdgHO1MZ8fYRQ', 'password': 'idsz'}


def upsample(input_layer):
    return UpSampling2d(scale=2)(input_layer)


def convolutional(
    input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky', name=None
):
    if downsample:
        input_layer = ZeroPad2d(((1, 0), (1, 0)))(input_layer)
        padding = 'VALID'
        strides = 2
    else:
        strides = 1
        padding = 'SAME'

    if bn:
        b_init = None
    else:
        b_init = tl.initializers.constant(value=0.0)

    conv = Conv2d(
        n_filter=filters_shape[-1], filter_size=(filters_shape[0], filters_shape[1]), strides=(strides, strides),
        padding=padding, b_init=b_init, name=name
    )(input_layer)

    if bn:
        if activate ==True:
            if activate_type == 'leaky':
                conv = BatchNorm2d(act='lrelu0.1')(conv)
            elif activate_type == 'mish':
                conv = BatchNorm2d(act=mish)(conv)
        else:
            conv = BatchNorm2d()(conv)
    return conv


def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2), activate_type=activate_type)

    residual_output = Elementwise(tf.add)([short_cut, conv])
    return residual_output


def cspdarknet53(input_data=None):

    input_data = convolutional(input_data, (3, 3, 3, 32), activate_type='mish')
    input_data = convolutional(input_data, (3, 3, 32, 64), downsample=True, activate_type='mish')

    route = input_data
    route = convolutional(route, (1, 1, 64, 64), activate_type='mish', name='conv_rote_block_1')
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type='mish')

    for i in range(1):
        input_data = residual_block(input_data, 64, 32, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type='mish')

    input_data = Concat()([input_data, route])
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type='mish')
    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type='mish')
    route = input_data
    route = convolutional(route, (1, 1, 128, 64), activate_type='mish', name='conv_rote_block_2')
    input_data = convolutional(input_data, (1, 1, 128, 64), activate_type='mish')
    for i in range(2):
        input_data = residual_block(input_data, 64, 64, 64, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 64, 64), activate_type='mish')
    input_data = Concat()([input_data, route])

    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type='mish')
    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type='mish')
    route = input_data
    route = convolutional(route, (1, 1, 256, 128), activate_type='mish', name='conv_rote_block_3')
    input_data = convolutional(input_data, (1, 1, 256, 128), activate_type='mish')
    for i in range(8):
        input_data = residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 128, 128), activate_type='mish')
    input_data = Concat()([input_data, route])

    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type='mish')
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type='mish')
    route = input_data
    route = convolutional(route, (1, 1, 512, 256), activate_type='mish', name='conv_rote_block_4')
    input_data = convolutional(input_data, (1, 1, 512, 256), activate_type='mish')
    for i in range(8):
        input_data = residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 256, 256), activate_type='mish')
    input_data = Concat()([input_data, route])

    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type='mish')
    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type='mish')
    route = input_data
    route = convolutional(route, (1, 1, 1024, 512), activate_type='mish', name='conv_rote_block_5')
    input_data = convolutional(input_data, (1, 1, 1024, 512), activate_type='mish')
    for i in range(4):
        input_data = residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = convolutional(input_data, (1, 1, 512, 512), activate_type='mish')
    input_data = Concat()([input_data, route])

    input_data = convolutional(input_data, (1, 1, 1024, 1024), activate_type='mish')
    input_data = convolutional(input_data, (1, 1, 1024, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    maxpool1 = MaxPool2d(filter_size=(13, 13), strides=(1, 1))(input_data)
    maxpool2 = MaxPool2d(filter_size=(9, 9), strides=(1, 1))(input_data)
    maxpool3 = MaxPool2d(filter_size=(5, 5), strides=(1, 1))(input_data)
    input_data = Concat()([maxpool1, maxpool2, maxpool3, input_data])

    input_data = convolutional(input_data, (1, 1, 2048, 512))
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    input_data = convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data


def YOLOv4(NUM_CLASS, pretrained=False):
    """Pre-trained YOLOv4 model.

    Parameters
    ------------
    NUM_CLASS : int
        Number of classes in final prediction.
    pretrained : boolean
        Whether to load pretrained weights. Default False.

    Examples
    ---------
    Object Detection with YOLOv4, see `computer_vision.py
    <https://github.com/tensorlayer/tensorlayer/blob/master/tensorlayer/app/computer_vision.py>`__
    With TensorLayer

    >>> # get the whole model, without pre-trained YOLOv4 parameters
    >>> yolov4 = tl.app.YOLOv4(NUM_CLASS=80, pretrained=False)
    >>> # get the whole model, restore pre-trained YOLOv4 parameters
    >>> yolov4 = tl.app.YOLOv4(NUM_CLASS=80, pretrained=True)
    >>> # use for inferencing
    >>> output = yolov4(img, is_train=False)

    """

    input_layer = Input([None, INPUT_SIZE, INPUT_SIZE, 3])
    route_1, route_2, conv = cspdarknet53(input_layer)

    route = conv
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = upsample(conv)
    route_2 = convolutional(route_2, (1, 1, 512, 256), name='conv_yolo_1')
    conv = Concat()([route_2, conv])

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    route_1 = convolutional(route_1, (1, 1, 256, 128), name='conv_yolo_2')
    conv = Concat()([route_1, conv])

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = convolutional(conv, (3, 3, 128, 256), name='conv_route_1')
    conv_sbbox = convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(route_1, (3, 3, 128, 256), downsample=True, name='conv_route_2')
    conv = Concat()([conv, route_2])

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = convolutional(conv, (3, 3, 256, 512), name='conv_route_3')
    conv_mbbox = convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = convolutional(route_2, (3, 3, 256, 512), downsample=True, name='conv_route_4')
    conv = Concat()([conv, route])

    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))

    conv = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    network = Model(input_layer, [conv_sbbox, conv_mbbox, conv_lbbox])

    if pretrained:
        restore_params(network, model_path='model/yolov4_model.npz')

    return network


def restore_params(network, model_path='models.npz'):
    logging.info("Restore pre-trained weights")

    try:
        npz = np.load(model_path, allow_pickle=True)
    except:
        print("Download the model file, placed in the /model ")
        print("Weights download: ", weights_url['link'], "password:", weights_url['password'])

    txt_path = 'model/yolov4_weights_config.txt'
    f = open(txt_path, "r")
    line = f.readlines()
    for i in range(len(line)):
        network.all_weights[i].assign(npz[line[i].strip()])
        logging.info("  Loading weights %s in %s" % (network.all_weights[i].shape, network.all_weights[i].name))
