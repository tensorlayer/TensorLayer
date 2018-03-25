#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
VGG-16 for ImageNet using TL models
"""

import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models.imagenet_classes import class_names

x = tf.placeholder(tf.float32, [None, 224, 224, 3])

# get the whole model
vgg = tl.models.Vgg16(x)

# restore pre-trained VGG parameters
sess = tf.InteractiveSession()

vgg.restore_params(sess)

probs = tf.nn.softmax(vgg.outputs)

vgg.print_params(False)

vgg.print_layers()

img1 = tl.vis.read_image('data/tiger.jpeg')
img1 = tl.prepro.imresize(img1, (224, 224))

_ = sess.run(probs, feed_dict={x: [img1]})[0]  # 1st time takes time to compile
start_time = time.time()
prob = sess.run(probs, feed_dict={x: [img1]})[0]
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])
