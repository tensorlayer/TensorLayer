#! /usr/bin/python
# -*- coding: utf-8 -*-

"""SqueezeNet for ImageNet using TL models."""

import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models.imagenet_classes import class_names

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

x = tf.placeholder(tf.float32, [None, 224, 224, 3])

# get the whole model
squeezenet = tl.models.SqueezeNetV1(x)

# restore pre-trained parameters
sess = tf.InteractiveSession()

squeezenet.restore_weights(sess)

probs = tf.nn.softmax(squeezenet.outputs)

squeezenet.print_weights(False)

squeezenet.print_layers()

img1 = tl.vis.read_image('data/tiger.jpeg')
img1 = tl.prepro.imresize(img1, (224, 224))

_ = sess.run(probs, feed_dict={x: [img1]})[0]  # 1st time takes time to compile
start_time = time.time()
prob = sess.run(probs, feed_dict={x: [img1]})[0]
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])
