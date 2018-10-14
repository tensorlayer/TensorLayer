#! /usr/bin/python
# -*- coding: utf8 -*-

"""
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
https://github.com/rcmalli/keras-mobilenet/blob/master/keras_mobilenet/mobilenet.py
"""

import json
import os
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (BatchNorm, Conv2d, DepthwiseConv2d, Flatten, GlobalMeanPool2d, Input, Reshape)
tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

MODEL_PATH = os.path.join("models", "mobilenet.npz")


def conv_block(n, n_filter, filter_size=(3, 3), strides=(1, 1), is_train=False, name='conv_block'):
    # ref: https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py
    with tf.variable_scope(name):
        n = Conv2d(n_filter, filter_size, strides, b_init=None, name='conv')(n)
        n = BatchNorm(decay=0.999, act=tf.nn.relu6, name='batchnorm')(n, is_train=is_train)
    return n


def depthwise_conv_block(n, n_filter, strides=(1, 1), is_train=False, name="depth_block"):
    with tf.variable_scope(name):
        n = DepthwiseConv2d((3, 3), strides, b_init=None, name='depthwise')(n)
        n = BatchNorm(decay=0.999, act=tf.nn.relu6, name='batchnorm1')(n, is_train=is_train)
        n = Conv2d(n_filter, (1, 1), (1, 1), b_init=None, name='conv')(n)
        n = BatchNorm(decay=0.999, act=tf.nn.relu6, name='batchnorm2')(n, is_train=is_train)
    return n


def decode_predictions(preds, top=5):  # keras.applications.resnet50
    fpath = os.path.join("data", "imagenet_class_index.json")
    if tl.files.file_exists(fpath) is False:
        raise Exception(
            "{} / download imagenet_class_index.json from: https://github.com/tensorlayer/tensorlayer/tree/master/example/data"
        )
    if isinstance(preds, np.ndarray) is False:
        preds = np.asarray(preds)
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError(
            '`decode_predictions` expects '
            'a batch of predictions '
            '(i.e. a 2D array of shape (samples, 1000)). '
            'Found array with shape: ' + str(preds.shape)
        )
    with open(fpath) as f:
        CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i], ) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def mobilenet(x, is_train=True, reuse=False):
    with tf.variable_scope("mobilenet", reuse=reuse):
        n = Input(x)
        n = conv_block(n, 32, strides=(2, 2), is_train=is_train, name="conv")
        n = depthwise_conv_block(n, 64, is_train=is_train, name="depth1")

        n = depthwise_conv_block(n, 128, strides=(2, 2), is_train=is_train, name="depth2")
        n = depthwise_conv_block(n, 128, is_train=is_train, name="depth3")

        n = depthwise_conv_block(n, 256, strides=(2, 2), is_train=is_train, name="depth4")
        n = depthwise_conv_block(n, 256, is_train=is_train, name="depth5")

        n = depthwise_conv_block(n, 512, strides=(2, 2), is_train=is_train, name="depth6")
        n = depthwise_conv_block(n, 512, is_train=is_train, name="depth7")
        n = depthwise_conv_block(n, 512, is_train=is_train, name="depth8")
        n = depthwise_conv_block(n, 512, is_train=is_train, name="depth9")
        n = depthwise_conv_block(n, 512, is_train=is_train, name="depth10")
        n = depthwise_conv_block(n, 512, is_train=is_train, name="depth11")

        n = depthwise_conv_block(n, 1024, strides=(2, 2), is_train=is_train, name="depth12")
        n = depthwise_conv_block(n, 1024, is_train=is_train, name="depth13")

        n = GlobalMeanPool2d(n)
        # n = Dropout(n, 1-1e-3, True, is_train, name='drop')
        # n = Dense(n, 1000, act=None, name='output')   # equal
        n = Reshape([-1, 1, 1, 1024])(n)
        n = Conv2d(1000, (1, 1), (1, 1), name='out')(n)
        n = Flatten()(n)
    return n


x = tf.placeholder(tf.float32, (None, 224, 224, 3))
n = mobilenet(x, False, False)
softmax = tf.nn.softmax(n.outputs)
n.print_layers()
n.print_weights(False)

sess = tf.InteractiveSession()
# tl.layers.initialize_global_variables(sess)

if not os.path.isfile(MODEL_PATH):
    raise Exception("Please download mobilenet.npz from : https://github.com/tensorlayer/pretrained-models")

tl.files.load_and_assign_npz(sess=sess, name=MODEL_PATH, network=n)

img = tl.vis.read_image('data/tiger.jpeg')
img = tl.prepro.imresize(img, (224, 224)) / 255
prob = sess.run(softmax, feed_dict={x: [img]})[0]  # the 1st time need time to compile
start_time = time.time()
prob = sess.run(softmax, feed_dict={x: [img]})[0]

print("  End time : %.5ss" % (time.time() - start_time))
print('Predicted :', decode_predictions([prob], top=3)[0])
# tl.files.save_npz(n.all_weights, name=MODEL_PATH, sess=sess)
