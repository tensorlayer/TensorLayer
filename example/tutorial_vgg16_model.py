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
import tensorflow as tf
import tensorlayer as tl
from scipy.misc import imread, imresize

try:
    from data.imagenet_classes import *
except Exception as e:
    raise Exception("{} / download the file from: https://github.com/zsdonghao/tensorlayer/tree/master/example/data".format(e))

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
# y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

model = tl.models.Vgg16Simple()
# model = tl.models.Vgg16Professional()  # professional CNN APIs
network = model(x)

y = network.outputs
probs = tf.nn.softmax(y)
# y_op = tf.argmax(tf.nn.softmax(y), 1)
# cost = tl.cost.cross_entropy(y, y_, name='cost')
# correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(y_, tf.float32))
# acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tl.layers.initialize_global_variables(sess)
network.print_params()
network.print_layers()

if not os.path.isfile("vgg16_weights.npz"):
    print("Please download vgg16_weights.npz from : http://www.cs.toronto.edu/~frossard/post/vgg16/")
    exit()
npz = np.load('vgg16_weights.npz')

params = []
for val in sorted(npz.items()):
    print("  Loading %s" % str(val[1].shape))
    params.append(val[1])

tl.files.assign_params(sess, params, network)

img1 = imread('data/laska.png', mode='RGB')  # test data in github
img1 = imresize(img1, (224, 224))

start_time = time.time()
prob = sess.run(probs, feed_dict={x: [img1]})[0]
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])
