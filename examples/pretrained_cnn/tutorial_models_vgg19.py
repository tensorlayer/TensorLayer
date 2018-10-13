#! /usr/bin/python
# -*- coding: utf-8 -*-

"""VGG-19 for ImageNet using TL models."""

import time
import skimage
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models.imagenet_classes import class_names

x = tf.placeholder(tf.float32, [None, 224, 224, 3])

# get the whole model
vgg = tl.models.VGG19(x)

# restore pre-trained VGG parameters
sess = tf.InteractiveSession()

vgg.restore_weights(sess)

probs = tf.nn.softmax(vgg.outputs)

vgg.print_weights(False)

vgg.print_layers()


# img1 = tl.vis.read_image('data/tiger.jpeg')
# img1 = tl.prepro.imresize(img1, (224, 224))
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if ((0 <= img).all() and (img <= 1.0).all()) is False:
        raise Exception("image value should be [0, 1]")
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224), anti_aliasing=False)
    return resized_img


img1 = load_image("data/tiger.jpeg")  # test data in github
img1 = img1.reshape((1, 224, 224, 3))

# rescale pixels values in the range of 0-1
# img1 = img1 / 255.0
if ((0 <= img1).all() and (img1 <= 1.0).all()) is False:
    raise Exception("image value should be [0, 1]")

_ = sess.run(probs, feed_dict={x: img1})[0]  # 1st time takes time to compile
start_time = time.time()
prob = sess.run(probs, feed_dict={x: img1})[0]
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])
