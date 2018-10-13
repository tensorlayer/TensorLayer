#! /usr/bin/python
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
feeding images of multiple sizes is by doing center cropping, as shown in the
following snippet:

>>> image_h, image_w, _ = np.shape(img)
>>> shorter_side = min(image_h, image_w)
>>> scale = 224. / shorter_side
>>> image_h, image_w = np.ceil([scale * image_h, scale * image_w]).astype('int32')
>>> img = imresize(img, (image_h, image_w))
>>> crop_x = (image_w - 224) / 2
>>> crop_y = (image_h - 224) / 2
>>> img = img[crop_y:crop_y+224,crop_x:crop_x+224,:]

"""

import os
import time
import numpy as np
from scipy.misc import imread, imresize
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


def conv_layers_simple_api(net_in):
    with tf.name_scope('preprocess'):
        # Notice that we include a preprocessing layer that takes the RGB image
        # with pixels values in the range of 0-1 and subtracts the mean image
        # values (calculated over the entire ImageNet training set).

        # Rescale the input tensor with pixels values in the range of 0-255
        net_in.outputs = net_in.outputs * 255.0

        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - mean

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
    net = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')(net)

    # conv4
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')(net)
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')(net)
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')(net)
    net = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')(net)

    # conv5
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')(net)
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')(net)
    net = Conv2d(n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')(net)
    net = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')(net)
    return net


def fc_layers(net):
    net = Flatten(name='flatten')(net)
    net = Dense(n_units=4096, act=tf.nn.relu, name='fc1_relu')(net)
    net = Dense(n_units=4096, act=tf.nn.relu, name='fc2_relu')(net)
    net = Dense(n_units=1000, act=None, name='fc3_relu')(net)
    return net


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
# y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

net_in = Input(name='input')(x)
# net_cnn = conv_layers(net_in)               # professional CNN APIs
net_cnn = conv_layers_simple_api(net_in)  # simplified CNN APIs
net = fc_layers(net_cnn)

y = net.outputs
probs = tf.nn.softmax(y)
# y_op = tf.argmax(tf.nn.softmax(y), 1)
# cost = tl.cost.cross_entropy(y, y_, name='cost')
# correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(y_, tf.float32))
# acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tl.layers.initialize_global_variables(sess)
net.print_weights()
net.print_layers()

tl.files.maybe_download_and_extract(
    'vgg16_weights.npz', 'models', 'http://www.cs.toronto.edu/~frossard/vgg16/', expected_bytes=553436134
)
npz = np.load(os.path.join('models', 'vgg16_weights.npz'))

params = []
for val in sorted(npz.items()):
    print("  Loading params %s" % str(val[1].shape))
    params.append(val[1])

tl.files.assign_weights(sess, params, net)

img1 = imread('data/laska.png', mode='RGB')  # test data in github
img1 = imresize(img1, (224, 224))
# rescale pixels values in the range of 0-1
img1 = img1 / 255.0
if ((0 <= img1).all() and (img1 <= 1.0).all()) is False:
    raise Exception("image value should be [0, 1]")

_ = sess.run(probs, feed_dict={x: [img1]})[0]  # 1st time take time to compile
start_time = time.time()
prob = sess.run(probs, feed_dict={x: [img1]})[0]
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])
