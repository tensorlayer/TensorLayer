#! /usr/bin/python
# -*- coding: utf-8 -*-
"""YOLOv4 for MS-COCO.

# Reference:
- [tensorflow-yolov4-tflite](
    https://github.com/hunglc007/tensorflow-yolov4-tflite)

"""

import numpy as np
import tensorlayer as tl
from tensorlayer.layers.activation import Mish
from tensorlayer.layers import Conv2d, MaxPool2d, BatchNorm2d, ZeroPad2d, UpSampling2d, Concat, Elementwise
from tensorlayer.layers import Module, SequentialLayer
from tensorlayer import logging

INPUT_SIZE = 416
weights_url = {'link': 'https://pan.baidu.com/s/1MC1dmEwpxsdgHO1MZ8fYRQ', 'password': 'idsz'}


class Convolutional(Module):
    """
    Create Convolution layer
    Because it is only a stack of reference layers, there is no build, so self._built=True
    """
    def __init__(self, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky',name=None):
        super(Convolutional, self).__init__()
        self.act = activate
        self.act_type = activate_type
        self.downsample = downsample
        self.bn = bn
        self._built = True
        if downsample:
            padding = 'VALID'
            strides = 2
        else:
            strides = 1
            padding = 'SAME'

        if bn:
            b_init = None
        else:
            b_init = tl.initializers.constant(value=0.0)

        self.zeropad = ZeroPad2d(((1, 0), (1, 0)))
        self.conv = Conv2d(n_filter=filters_shape[-1], in_channels=filters_shape[2], filter_size=(filters_shape[0], filters_shape[1]),
                           strides=(strides, strides),padding=padding, b_init=b_init, name=name)

        if bn:
            if activate == True:
                if activate_type == 'leaky':
                    self.batchnorm2d = BatchNorm2d(act='leaky_relu0.1', num_features=filters_shape[-1])
                elif activate_type == 'mish':
                    self.batchnorm2d = BatchNorm2d(act=Mish, num_features=filters_shape[-1])
            else:
                self.batchnorm2d = BatchNorm2d(act=None, num_features=filters_shape[-1])

    def forward(self, input):
        if self.downsample:
            input = self.zeropad(input)

        output = self.conv(input)

        if self.bn:
            output = self.batchnorm2d(output)
        return output

class residual_block(Module):
    def __init__(self, input_channel, filter_num1, filter_num2, activate_type='leaky'):
        super(residual_block, self).__init__()
        self.conv1 = Convolutional(filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
        self.conv2 = Convolutional(filters_shape=(3, 3, filter_num1, filter_num2), activate_type=activate_type)
        self.add = Elementwise(tl.add)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.add([inputs, output])
        return output

def residual_block_num(num, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    residual_list = []
    for i in range(num):
        residual_list.append(residual_block(input_channel, filter_num1, filter_num2, activate_type=activate_type))
    return SequentialLayer(residual_list)

class cspdarknet53(Module):
    def __init__(self):
        super(cspdarknet53, self).__init__()
        self._built = True
        self.conv1_1 = Convolutional((3, 3, 3, 32), activate_type='mish')
        self.conv1_2 = Convolutional((3, 3, 32, 64), downsample=True, activate_type='mish')
        self.conv1_3 = Convolutional((1, 1, 64, 64), activate_type='mish', name='conv_rote_block_1')
        self.conv1_4 = Convolutional((1, 1, 64, 64), activate_type='mish')
        self.residual_1 = residual_block_num(1, 64, 32, 64, activate_type="mish")

        self.conv2_1 = Convolutional((1, 1, 64, 64), activate_type='mish')
        self.concat = Concat()
        self.conv2_2 = Convolutional((1, 1, 128, 64), activate_type='mish')
        self.conv2_3 = Convolutional((3, 3, 64, 128), downsample=True, activate_type='mish')
        self.conv2_4 = Convolutional((1, 1, 128, 64), activate_type='mish', name='conv_rote_block_2')
        self.conv2_5 = Convolutional((1, 1, 128, 64), activate_type='mish')
        self.residual_2 = residual_block_num(2, 64, 64, 64, activate_type='mish')

        self.conv3_1 = Convolutional((1, 1, 64, 64), activate_type='mish')
        self.conv3_2 = Convolutional((1, 1, 128, 128), activate_type='mish')
        self.conv3_3 = Convolutional((3, 3, 128, 256), downsample=True, activate_type='mish')
        self.conv3_4 = Convolutional((1, 1, 256, 128), activate_type='mish', name='conv_rote_block_3')
        self.conv3_5 = Convolutional((1, 1, 256, 128), activate_type='mish')
        self.residual_3 = residual_block_num(8, 128, 128, 128, activate_type="mish")

        self.conv4_1 = Convolutional((1, 1, 128, 128), activate_type='mish')
        self.conv4_2 = Convolutional((1, 1, 256, 256), activate_type='mish')
        self.conv4_3 = Convolutional((3, 3, 256, 512), downsample=True, activate_type='mish')
        self.conv4_4 = Convolutional((1, 1, 512, 256), activate_type='mish', name='conv_rote_block_4')
        self.conv4_5 = Convolutional((1, 1, 512, 256), activate_type='mish')
        self.residual_4 = residual_block_num(8, 256, 256, 256, activate_type="mish")

        self.conv5_1 = Convolutional((1, 1, 256, 256), activate_type='mish')
        self.conv5_2 = Convolutional((1, 1, 512, 512), activate_type='mish')
        self.conv5_3 = Convolutional((3, 3, 512, 1024), downsample=True, activate_type='mish')
        self.conv5_4 = Convolutional((1, 1, 1024, 512), activate_type='mish', name='conv_rote_block_5')
        self.conv5_5 = Convolutional((1, 1, 1024, 512), activate_type='mish')
        self.residual_5 = residual_block_num(4, 512, 512, 512, activate_type="mish")


        self.conv6_1 = Convolutional((1, 1, 512, 512), activate_type='mish')
        self.conv6_2 = Convolutional((1, 1, 1024, 1024), activate_type='mish')
        self.conv6_3 = Convolutional((1, 1, 1024, 512))
        self.conv6_4 = Convolutional((3, 3, 512, 1024))
        self.conv6_5 = Convolutional((1, 1, 1024, 512))

        self.maxpool1 = MaxPool2d(filter_size=(13, 13), strides=(1, 1))
        self.maxpool2 = MaxPool2d(filter_size=(9, 9), strides=(1, 1))
        self.maxpool3 = MaxPool2d(filter_size=(5, 5), strides=(1, 1))

        self.conv7_1 = Convolutional((1, 1, 2048, 512))
        self.conv7_2 = Convolutional((3, 3, 512, 1024))
        self.conv7_3 = Convolutional((1, 1, 1024, 512))

    def forward(self, input_data):
        input_data = self.conv1_1(input_data)
        input_data = self.conv1_2(input_data)
        route = input_data
        route = self.conv1_3(route)
        input_data = self.conv1_4(input_data)
        input_data = self.residual_1(input_data)

        input_data = self.conv2_1(input_data)
        input_data = self.concat([input_data, route])
        input_data = self.conv2_2(input_data)
        input_data = self.conv2_3(input_data)
        route = input_data
        route = self.conv2_4(route)
        input_data = self.conv2_5(input_data)
        input_data = self.residual_2(input_data)

        input_data = self.conv3_1(input_data)
        input_data = self.concat([input_data, route])
        input_data = self.conv3_2(input_data)
        input_data = self.conv3_3(input_data)
        route = input_data
        route = self.conv3_4(route)
        input_data = self.conv3_5(input_data)
        input_data = self.residual_3(input_data)

        input_data = self.conv4_1(input_data)
        input_data = self.concat([input_data, route])
        input_data = self.conv4_2(input_data)
        route_1 = input_data
        input_data = self.conv4_3(input_data)
        route = input_data
        route = self.conv4_4(route)
        input_data = self.conv4_5(input_data)
        input_data = self.residual_4(input_data)

        input_data = self.conv5_1(input_data)
        input_data = self.concat([input_data, route])
        input_data = self.conv5_2(input_data)
        route_2 = input_data
        input_data = self.conv5_3(input_data)
        route = input_data
        route = self.conv5_4(route)
        input_data = self.conv5_5(input_data)
        input_data = self.residual_5(input_data)

        input_data = self.conv6_1(input_data)
        input_data = self.concat([input_data, route])

        input_data = self.conv6_2(input_data)
        input_data = self.conv6_3(input_data)
        input_data = self.conv6_4(input_data)
        input_data = self.conv6_5(input_data)

        maxpool1 = self.maxpool1(input_data)
        maxpool2 = self.maxpool2(input_data)
        maxpool3 = self.maxpool3(input_data)
        input_data = self.concat([maxpool1, maxpool2, maxpool3, input_data])

        input_data = self.conv7_1(input_data)
        input_data = self.conv7_2(input_data)
        input_data = self.conv7_3(input_data)

        return route_1, route_2, input_data


class YOLOv4_model(Module):
    def __init__(self, NUM_CLASS):
        super(YOLOv4_model, self).__init__()
        self.cspdarnnet = cspdarknet53()

        self.conv1_1 = Convolutional((1, 1, 512, 256))
        self.upsamle = UpSampling2d(scale=2)
        self.conv1_2 = Convolutional((1, 1, 512, 256), name='conv_yolo_1')
        self.concat = Concat()

        self.conv2_1 = Convolutional((1, 1, 512, 256))
        self.conv2_2 = Convolutional((3, 3, 256, 512))
        self.conv2_3 = Convolutional((1, 1, 512, 256))
        self.conv2_4 = Convolutional((3, 3, 256, 512))
        self.conv2_5 = Convolutional((1, 1, 512, 256))

        self.conv3_1 = Convolutional((1, 1, 256, 128))
        self.conv3_2 = Convolutional((1, 1, 256, 128), name='conv_yolo_2')

        self.conv4_1 = Convolutional((1, 1, 256, 128))
        self.conv4_2 = Convolutional((3, 3, 128, 256))
        self.conv4_3 = Convolutional((1, 1, 256, 128))
        self.conv4_4 = Convolutional((3, 3, 128, 256))
        self.conv4_5 = Convolutional((1, 1, 256, 128))

        self.conv5_1 = Convolutional((3, 3, 128, 256), name='conv_route_1')
        self.conv5_2 = Convolutional((1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        self.conv6_1 = Convolutional((3, 3, 128, 256), downsample=True, name='conv_route_2')
        self.conv6_2 = Convolutional((1, 1, 512, 256))
        self.conv6_3 = Convolutional((3, 3, 256, 512))
        self.conv6_4 = Convolutional((1, 1, 512, 256))
        self.conv6_5 = Convolutional((3, 3, 256, 512))
        self.conv6_6 = Convolutional((1, 1, 512, 256))

        self.conv7_1 = Convolutional((3, 3, 256, 512), name='conv_route_3')
        self.conv7_2 = Convolutional((1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
        self.conv7_3 = Convolutional((3, 3, 256, 512), downsample=True, name='conv_route_4')

        self.conv8_1 = Convolutional((1, 1, 1024, 512))
        self.conv8_2 = Convolutional((3, 3, 512, 1024))
        self.conv8_3 = Convolutional((1, 1, 1024, 512))
        self.conv8_4 = Convolutional((3, 3, 512, 1024))
        self.conv8_5 = Convolutional((1, 1, 1024, 512))

        self.conv9_1 = Convolutional((3, 3, 512, 1024))
        self.conv9_2 = Convolutional((1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    def forward(self, inputs):
        route_1, route_2, conv = self.cspdarnnet(inputs)

        route = conv
        conv = self.conv1_1(conv)
        conv = self.upsamle(conv)
        route_2 = self.conv1_2(route_2)
        conv = self.concat([route_2, conv])

        conv = self.conv2_1(conv)
        conv = self.conv2_2(conv)
        conv = self.conv2_3(conv)
        conv = self.conv2_4(conv)
        conv = self.conv2_5(conv)

        route_2 = conv
        conv = self.conv3_1(conv)
        conv = self.upsamle(conv)
        route_1 = self.conv3_2(route_1)
        conv = self.concat([route_1, conv])

        conv = self.conv4_1(conv)
        conv = self.conv4_2(conv)
        conv = self.conv4_3(conv)
        conv = self.conv4_4(conv)
        conv = self.conv4_5(conv)

        route_1 = conv
        conv = self.conv5_1(conv)
        conv_sbbox = self.conv5_2(conv)

        conv = self.conv6_1(route_1)
        conv = self.concat([conv, route_2])

        conv = self.conv6_2(conv)
        conv = self.conv6_3(conv)
        conv = self.conv6_4(conv)
        conv = self.conv6_5(conv)
        conv = self.conv6_6(conv)

        route_2 = conv
        conv = self.conv7_1(conv)
        conv_mbbox = self.conv7_2(conv)
        conv = self.conv7_3(route_2)
        conv = self.concat([conv, route])

        conv = self.conv8_1(conv)
        conv = self.conv8_2(conv)
        conv = self.conv8_3(conv)
        conv = self.conv8_4(conv)
        conv = self.conv8_5(conv)

        conv = self.conv9_1(conv)
        conv_lbbox = self.conv9_2(conv)

        return conv_sbbox, conv_mbbox, conv_lbbox

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

    network = YOLOv4_model(NUM_CLASS=NUM_CLASS)

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

    txt_path = 'model/yolov4_weights3_config.txt'
    f = open(txt_path, "r")
    line = f.readlines()
    for i in range(len(line)):
        network.all_weights[i].assign(npz[line[i].strip()])
        logging.info("  Loading weights %s in %s" % (network.all_weights[i].shape, network.all_weights[i].name))

def tl2_weights_to_tl3_weights(weights_2_path='model/weights_2.txt', weights_3_path='model/weights_3.txt', txt_path='model/yolov4_weights_config.txt'):
    weights_2_path = weights_2_path
    weights_3_path = weights_3_path
    txt_path = txt_path
    f1 = open(weights_2_path, "r")
    f2 = open(weights_3_path, "r")
    f3 = open(txt_path, "r")
    line1 = f1.readlines()
    line2 = f2.readlines()
    line3 = f3.readlines()
    _dicts = {}
    for i in range(len(line1)):
        _dicts[line1[i].strip()] = line3[i].strip()
    for j in range(len(line2)):
        print(_dicts[line2[j].strip()])
