#! /usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Concat, Conv2d, Dropout, GlobalMeanPool2d, Input, MaxPool2d)

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

MODEL_PATH = os.path.join("models", "squeezenet.npz")


def decode_predictions(preds, top=5):  # keras.applications.resnet50
    fpath = os.path.join("data", "imagenet_class_index.json")
    if tl.files.file_exists(fpath) is False:
        raise Exception(
            "{} / download imagenet_class_index.json from: https://github.com/tensorlayer/tensorlayer/tree/master/example/data"
        )
    if isinstance(preds, np.ndarray) is False:
        preds = np.asarray(preds)
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError(
            '`decode_predictions` expects '
            'a batch of predictions '
            '(i.e. a 2D array of shape (samples, 1000)). '
            'Found array with shape: ' + str(preds.shape)
        )
    with open(fpath) as f:
        CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i], ) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def squeezenet(x, is_train=True, reuse=False):
    # model from: https://github.com/wohlert/keras-squeezenet
    #             https://github.com/DT42/squeezenet_demo/blob/master/model.py
    with tf.variable_scope("squeezenet", reuse=reuse):
        with tf.variable_scope("input"):
            n = Input(x)
            # n = Conv2d(96, (7,7),(2,2),tf.nn.relu,'SAME',name='conv1')(n)
            n = Conv2d(64, (3, 3), (2, 2), tf.nn.relu, 'SAME', name='conv1')(n)
            n = MaxPool2d((3, 3), (2, 2), 'VALID', name='max')(n)

        with tf.variable_scope("fire2"):
            n = Conv2d(16, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')(n)
            n1 = Conv2d(64, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')(n)
            n2 = Conv2d(64, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')(n)
            n = Concat(-1, name='concat')([n1, n2])

        with tf.variable_scope("fire3"):
            n = Conv2d(16, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')(n)
            n1 = Conv2d(64, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')(n)
            n2 = Conv2d(64, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')(n)
            n = Concat(-1, name='concat')([n1, n2])
            n = MaxPool2d((3, 3), (2, 2), 'VALID', name='max')(n)

        with tf.variable_scope("fire4"):
            n = Conv2d(32, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')(n)
            n1 = Conv2d(128, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')(n)
            n2 = Conv2d(128, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')(n)
            n = Concat(-1, name='concat')([n1, n2])

        with tf.variable_scope("fire5"):
            n = Conv2d(32, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')(n)
            n1 = Conv2d(128, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')(n)
            n2 = Conv2d(128, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')(n)
            n = Concat(-1, name='concat')([n1, n2])
            n = MaxPool2d((3, 3), (2, 2), 'VALID', name='max')(n)

        with tf.variable_scope("fire6"):
            n = Conv2d(48, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')(n)
            n1 = Conv2d(192, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')(n)
            n2 = Conv2d(192, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')(n)
            n = Concat(-1, name='concat')([n1, n2])

        with tf.variable_scope("fire7"):
            n = Conv2d(48, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')(n)
            n1 = Conv2d(192, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')(n)
            n2 = Conv2d(192, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')(n)
            n = Concat(-1, name='concat')([n1, n2])

        with tf.variable_scope("fire8"):
            n = Conv2d(64, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')(n)
            n1 = Conv2d(256, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')(n)
            n2 = Conv2d(256, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')(n)
            n = Concat(-1, name='concat')([n1, n2])

        with tf.variable_scope("fire9"):
            n = Conv2d(64, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='squeeze1x1')(n)
            n1 = Conv2d(256, (1, 1), (1, 1), tf.nn.relu, 'SAME', name='expand1x1')(n)
            n2 = Conv2d(256, (3, 3), (1, 1), tf.nn.relu, 'SAME', name='expand3x3')(n)
            n = Concat(-1, name='concat')([n1, n2])

        with tf.variable_scope("output"):
            n = Dropout(keep=0.5, is_fix=True, is_train=is_train, name='drop1')(n)
            n = Conv2d(1000, (1, 1), (1, 1), padding='VALID', name='conv10')(n)  # 13, 13, 1000
            n = GlobalMeanPool2d()(n)
        return n


x = tf.placeholder(tf.float32, (None, 224, 224, 3))
n = squeezenet(x, False, False)
softmax = tf.nn.softmax(n.outputs)
n.print_layers()
n.print_weights(False)

sess = tf.InteractiveSession()
tl.layers.initialize_global_variables(sess)

if tl.files.file_exists(MODEL_PATH):
    tl.files.load_and_assign_npz(sess=sess, name=MODEL_PATH, network=n)
else:
    raise Exception(
        "please download the pre-trained squeezenet.npz from https://github.com/tensorlayer/pretrained-models"
    )

img = tl.vis.read_image('data/tiger.jpeg', '')
img = tl.prepro.imresize(img, (224, 224))
prob = sess.run(softmax, feed_dict={x: [img]})[0]  # the 1st time need time to compile
start_time = time.time()
prob = sess.run(softmax, feed_dict={x: [img]})[0]
print("  End time : %.5ss" % (time.time() - start_time))

print('Predicted:', decode_predictions([prob], top=3)[0])
tl.files.save_npz(n.all_weights, name=MODEL_PATH, sess=sess)
