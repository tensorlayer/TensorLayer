#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
MobileNetV1 for ImageNet using TL models

- mobilenetv2 : https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
- tf.slim     : https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models
"""

import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.models.imagenet_classes import class_names

# tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

# get the whole model
mobilenetv1 = tl.models.MobileNetV1(pretrained=True)

img1 = tl.vis.read_image('data/tiger.jpeg')
img1 = tl.prepro.imresize(img1, (224, 224)) / 255
img1 = img1.astype(np.float32)[np.newaxis, ...]

start_time = time.time()
output = mobilenetv1(img1, is_train=False)
prob = tf.nn.softmax(output)[0].numpy()
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])
