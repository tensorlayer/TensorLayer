#! /usr/bin/python
# -*- coding: utf-8 -*-




import inspect
import os
import time

import numpy as np
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import tensorlayer as tl
from data.imagenet_classes import *

"""
VGG-19 for ImageNet
--------------------
Pre-trained model in this example - VGG19 NPZ and
trainable examples of VGG16/19 in TensorFlow can be found here:
https://github.com/machrisaa/tensorflow-vgg

For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.
"""


VGG_MEAN = [103.939, 116.779, 123.68]

def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
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
    red, green, blue = tf.split(3, 3, rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(3, [
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    """ input layer """
    net_in = tl.layers.InputLayer(bgr, name='input_layer')
    """ conv1 """
    network = tl.layers.Conv2dLayer(net_in,
                    act = tf.nn.relu,
                    shape = [3, 3, 3, 64],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv1_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 64, 64],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv1_2')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool1')
    """ conv2 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 64, 128],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv2_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 128, 128],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv2_2')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool2')
    """ conv3 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 128, 256],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 256],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 256],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_3')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 256],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3_4')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool3')
    """ conv4 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 256, 512],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_3')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4_4')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool4')
    """ conv5 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_1')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_2')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_3')
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [3, 3, 512, 512],
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5_4')
    network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool5')
    """ fc 6~8 """
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DenseLayer(network, n_units=4096,
                        act = tf.nn.relu, name = 'fc6')
    network = tl.layers.DenseLayer(network, n_units=4096,
                        act = tf.nn.relu, name = 'fc7')
    network = tl.layers.DenseLayer(network, n_units=1000,
                        act = tf.identity, name = 'fc8')

    print("build model finished: %fs" % (time.time() - start_time))
    return network

sess = tf.InteractiveSession()
x = tf.placeholder("float", [None, 224, 224, 3])
network = Vgg19(x)
y = network.outputs
probs = tf.nn.softmax(y, name="prob")
# sess.run(tf.initialize_all_variables())
tl.layers.initialize_global_variables(sess)

# You need to download the pre-trained model - VGG19 NPZ
# in https://github.com/machrisaa/tensorflow-vgg
vgg19_npy_path = "vgg19.npy"
npz = np.load(vgg19_npy_path, encoding='latin1').item()

params = []
for val in sorted( npz.items() ):
    W = np.asarray(val[1][0])
    b = np.asarray(val[1][1])
    print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
    params.extend([W, b])

print("Restoring model from npz file")
tl.files.assign_params(sess, params, network)

img1 = load_image("data/tiger.jpeg") # test data in github
img1 = img1.reshape((1, 224, 224, 3))
start_time = time.time()
prob = sess.run(probs, feed_dict= {x : img1})
print("End time : %.5ss" % (time.time() - start_time))

print_prob(prob[0])
