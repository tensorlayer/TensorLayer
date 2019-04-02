#! /usr/bin/python
# -*- coding: utf-8 -*-
"""VGG-16 for ImageNet using TL models."""

import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.models.imagenet_classes import class_names

tl.logging.set_verbosity(tl.logging.DEBUG)


# get the whole model
vgg = tl.models.vgg16(pretrained=True)

img1 = tl.vis.read_image('data/tiger.jpeg')
img1 = tl.prepro.imresize(img1, (224, 224))
img1 = img1.astype(np.float32)
mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape([1, 1, 1, 3])
img1 = img1 - mean

start_time = time.time()
vgg.eval()
output = vgg(img1)
probs = tf.nn.softmax(output)[0].numpy()
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(probs)[::-1])[0:5]
for p in preds:
    print(class_names[p], probs[p])
