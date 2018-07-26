#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
ResnetV1 for ImageNet using TL models

- tf.slim     : https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py
"""

import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models.imagenet_classes import class_names

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

x = tf.placeholder(tf.float32, [None, 224, 224, 3])

# get the whole model
resnetv1_50 = tl.models.ResNetV1_50(x)

# restore pre-trained parameters
sess = tf.InteractiveSession()

resnetv1_50.restore_params(sess)

# probs = tf.nn.softmax(resnetv1_50.outputs)

resnetv1_50.print_params(False)

resnetv1_50.print_layers()

img1 = tl.vis.read_image('data/tiger.jpeg')
img1 = tl.prepro.imresize(img1, (224, 224)) / 255

_ = sess.run(probs, feed_dict={x: [img1]})[0]  # 1st time takes time to compile
start_time = time.time()
prob = sess.run(probs, feed_dict={x: [img1]})[0]
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])
