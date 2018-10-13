#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
VGG-19 for ImageNet.

Pre-trained model in this example - VGG19 NPZ and
trainable examples of VGG16/19 in TensorFlow can be found here:
https://github.com/machrisaa/tensorflow-vgg

For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.

"""

import os
import time
import numpy as np
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

try:
    from tensorlayer.models.imagenet_classes import *
except Exception as e:
    raise Exception(
        "{} / download the file from: https://github.com/tensorlayer/tensorlayer/tree/master/example/data".format(e)
    )

VGG_MEAN = [103.939, 116.779, 123.68]

MODEL_DIR = "models"
MODEL_NAME = "vgg19.npy"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if ((0 <= img).all() and (img <= 1.0).all()) is False:
        raise Exception("image value should be [0, 1]")
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224), anti_aliasing=False)
    return resized_img


def print_prob(prob):
    synset = class_names
    # print prob
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1


def Vgg19(rgb):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    start_time = time.time()
    print("build model started")
    rgb_scaled = rgb * 255.0
    # Convert RGB to BGR
    red, green, blue = tf.split(rgb_scaled, 3, 3)

    if red.get_shape().as_list()[1:] != [224, 224, 1]:
        raise Exception("image size unmatch")

    if green.get_shape().as_list()[1:] != [224, 224, 1]:
        raise Exception("image size unmatch")

    if blue.get_shape().as_list()[1:] != [224, 224, 1]:
        raise Exception("image size unmatch")

    bgr = tf.concat([
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ], axis=3)

    if bgr.get_shape().as_list()[1:] != [224, 224, 3]:
        raise Exception("image size unmatch")

    # input layer
    net_in = Input(name='input')(bgr)
    # conv1
    net = Conv2d(64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')(net_in)
    net = Conv2d(n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')(net)
    net = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')(net)
    # conv2
    net = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')(net)
    net = Conv2d(n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')(net)
    net = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')(net)
    # conv3
    net = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')(net)
    net = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')(net)
    net = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')(net)
    net = Conv2d(n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')(net)
    net = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')(net)
    # conv4
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')(net)
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')(net)
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')(net)
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')(net)
    net = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')(net)
    # conv5
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')(net)
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')(net)
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')(net)
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')(net)
    net = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')(net)
    # fc 6~8
    net = Flatten(name='flatten')(net)
    net = Dense(n_units=4096, act=tf.nn.relu, name='fc6')(net)
    net = Dense(n_units=4096, act=tf.nn.relu, name='fc7')(net)
    net = Dense(n_units=1000, act=None, name='fc8')(net)
    print("build model finished: %fs" % (time.time() - start_time))
    return net


sess = tf.InteractiveSession()
x = tf.placeholder("float", [None, 224, 224, 3])
net = Vgg19(x)
y = net.outputs
probs = tf.nn.softmax(y, name="prob")
tl.layers.initialize_global_variables(sess)

# You need to download the pre-trained model - VGG19 NPY
if not os.path.isfile(MODEL_PATH):
    print("Please download vgg19.npy from : https://github.com/machrisaa/tensorflow-vgg")
    exit()

npy_file = np.load(MODEL_PATH, encoding='latin1').item()

params = []
for val in sorted(npy_file.items()):
    W = np.asarray(val[1][0])
    b = np.asarray(val[1][1])
    print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
    params.extend([W, b])

print("Restoring model from npy file")
tl.files.assign_weights(sess, params, net)

img1 = load_image("data/tiger.jpeg")  # test data in github
img1 = img1.reshape((1, 224, 224, 3))

start_time = time.time()
prob = sess.run(probs, feed_dict={x: img1})
print("End time : %.5ss" % (time.time() - start_time))

print_prob(prob[0])
