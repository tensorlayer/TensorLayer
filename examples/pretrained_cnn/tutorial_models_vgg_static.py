#! /usr/bin/python
# -*- coding: utf-8 -*-
"""VGG for ImageNet using TL models."""

import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.models.imagenet_classes import class_names

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


# get the whole model
sess = tf.InteractiveSession()
vgg = tl.models.vgg.vgg16(pretrained=True, sess=sess)
# sess.run(tf.global_variables_initializer())


x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
y = vgg(x, is_train=False)

img1 = tl.vis.read_image('data/tiger.jpeg')
img1 = tl.prepro.imresize(img1, (224, 224))
img1 = img1.astype(np.float32)
mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape([1, 1, 1, 3])
img1 = img1 - mean

start_time = time.time()
output = sess.run(y, feed_dict={x: img1})
probs = tf.nn.softmax(output)[0].eval()
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(probs)[::-1])[0:5]
for p in preds:
    print(class_names[p], probs[p])
