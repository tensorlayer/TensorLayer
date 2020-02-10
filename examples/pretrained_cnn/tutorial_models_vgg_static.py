#! /usr/bin/python
# -*- coding: utf-8 -*-
"""VGG for ImageNet using TL models."""

import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.models.imagenet_classes import class_names

tl.logging.set_verbosity(tl.logging.DEBUG)

# get the whole model
vgg = tl.models.vgg16(pretrained=True, mode='static')

img = tl.vis.read_image('data/tiger.jpeg')
img = tl.prepro.imresize(img, (224, 224)).astype(np.float32) / 255

start_time = time.time()
output = vgg(img, is_train=False)
probs = tf.nn.softmax(output)[0].numpy()
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(probs)[::-1])[0:5]
for p in preds:
    print(class_names[p], probs[p])
