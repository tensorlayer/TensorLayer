# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

with tf.Graph().as_default() as graph:
    # - Classify ImageNet classes with VGG16, see `tutorial_models_vgg16.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_vgg16.py>__`
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # get the whole model
    vgg = tl.models.VGG16(x)
    # restore pre-trained VGG parameters
    # sess = tf.InteractiveSession()
    # vgg.restore_params(sess)
    # use for inferencing
    probs = tf.nn.softmax(vgg.outputs)
    if len(vgg.all_layers) != 22:
        raise Exception("layers do not match")

    if len(vgg.all_params) != 32:
        raise Exception("params do not match")

with tf.Graph().as_default() as graph:
    # - Extract features with VGG16 and Train a classifier with 100 classes
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # get VGG without the last layer
    vgg = tl.models.VGG16(x, end_with='fc2_relu')

    if len(vgg.all_layers) != 21:
        raise Exception("layers do not match")

    if len(vgg.all_params) != 30:
        raise Exception("params do not match")

    # add one more layer
    net = tl.layers.DenseLayer(vgg, 100, name='out')
    # initialize all parameters
    # sess = tf.InteractiveSession()
    # tl.layers.initialize_global_variables(sess)
    # restore pre-trained VGG parameters
    # vgg.restore_params(sess)
    # train your own classifier (only update the last layer)
    train_params = tl.layers.get_variables_with_name('out')
    if len(train_params) != 2:
        raise Exception("params do not match")

with tf.Graph().as_default() as graph:
    # - Reuse model
    x1 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    x2 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # get VGG without the last layer
    vgg1 = tl.models.VGG16(x1, end_with='fc2_relu')
    # reuse the parameters of vgg1 with different input
    vgg2 = tl.models.VGG16(x2, end_with='fc2_relu', reuse=True)
    # restore pre-trained VGG parameters (as they share parameters, we don’t need to restore vgg2)
    # sess = tf.InteractiveSession()
    # vgg1.restore_params(sess)

    if len(vgg1.all_layers) != 21:
        raise Exception("layers do not match")

    if len(vgg1.all_params) != 30:
        raise Exception("params do not match")
