#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from keras.layers import *
from tensorlayer.layers import *
slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import (inception_v3, inception_v3_arg_scope)

sess = tf.InteractiveSession()

# LambdaLayer
x = tf.placeholder(tf.float32, shape=[None, 784])


def keras_block(x):
    x = Dropout(0.8)(x)
    x = Dense(100, activation='relu')(x)
    # x = Dropout(0.8)(x)
    # x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    logits = Dense(10, activation='linear')(x)
    return logits


network = InputLayer(x, name='input')
network = LambdaLayer(network, fn=keras_block, name='keras')

# SlimNetsLayer
x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
net_in = tl.layers.InputLayer(x, name='input_layer')
with slim.arg_scope(inception_v3_arg_scope()):

    # Alternatively, you should implement inception_v3 without TensorLayer as follow.
    # logits, end_points = inception_v3(X, num_classes=1001,
    #                                   is_training=False)
    network = tl.layers.SlimNetsLayer(
        prev_layer=net_in,
        slim_layer=inception_v3,
        slim_args={
            'num_classes': 1001,
            'is_training': False,
            #  'dropout_keep_prob' : 0.8,       # for training
            #  'min_depth' : 16,
            #  'depth_multiplier' : 1.0,
            #  'prediction_fn' : slim.softmax,
            #  'spatial_squeeze' : True,
            #  'reuse' : None,
            #  'scope' : 'InceptionV3'
        },
        name='InceptionV3'  # <-- the name should be the same with the ckpt model
    )
