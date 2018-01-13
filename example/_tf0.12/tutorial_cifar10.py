#! /usr/bin/python
# -*- coding: utf-8 -*-


import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep

"""Example of CNN, CIFAR-10

main_test_cnn_naive       : No distorted image / Low accuracy
main_test_cnn_advanced    : Uses distorted image / High accurcy but Slow
tutorial_cifar10_tfrecord : Preparing distorted image with Queue and Thread
                            / High accurcy and Fast
"""
exit("This example is deprecated, please see tutorial_cifar10_tfrecord.py (TFRecord) or tl.prepro (Python Threading)")

def main_test_cnn_naive():
    """Without any distorting, whitening and cropping for training data.
    This method work well for MNIST, but not CIFAR-10.

    For simplified CNN layer see "Convolutional layer (Simplified)"
    in read the docs website.
    """
    model_file_name = "model_cifar10_naive.ckpt"
    resume = False # load model, resume from previous checkpoint?

    X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(
                                        shape=(-1, 32, 32, 3), plotable=False)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    print('X_train.shape', X_train.shape)   # (50000, 32, 32, 3)
    print('y_train.shape', y_train.shape)   # (50000,)
    print('X_test.shape', X_test.shape)     # (10000, 32, 32, 3)
    print('y_test.shape', y_test.shape)     # (10000,)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    sess = tf.InteractiveSession()

    batch_size = 128

    x = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
                                # [batch_size, height, width, channels]
    y_ = tf.placeholder(tf.int32, shape=[batch_size,])

    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [5, 5, 3, 64],  # 64 features for each 5x5x3 patch
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        name ='cnn_layer1')     # output: (?, 32, 32, 64)
    network = tl.layers.PoolLayer(network,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='pool_layer1')   # output: (?, 16, 16, 64)
    # local response normalization, you can also try batch normalization.
    # References: ImageNet Classification with Deep Convolutional Neural Networks
    #   it increases the accuracy but consume more time.
    #     https://www.tensorflow.org/versions/master/api_docs/python/nn.html#local_response_normalization
    network.outputs = tf.nn.lrn(network.outputs, 4, bias=1.0, alpha=0.001 / 9.0,
                                                        beta=0.75, name='norm1')
    network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [5, 5, 64, 64], # 64 features for each 5x5 patch
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        name ='cnn_layer2')     # output: (?, 16, 16, 64)
    # Another local response normalization.
    network.outputs = tf.nn.lrn(network.outputs, 4, bias=1.0, alpha=0.001 / 9.0,
                                                        beta=0.75, name='norm2')
    network = tl.layers.PoolLayer(network,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='pool_layer2')   # output: (?, 8, 8, 64)

    network = tl.layers.FlattenLayer(network, name='flatten_layer')
                                                            # output: (?, 4096)
    network = tl.layers.DenseLayer(network, n_units=384,
                            act = tf.nn.relu, name='relu1') # output: (?, 384)
    network = tl.layers.DenseLayer(network, n_units=192,
                            act = tf.nn.relu, name='relu2') # output: (?, 192)
    network = tl.layers.DenseLayer(network, n_units=10,
                            act = tf.identity,
                            name='output_layer')            # output: (?, 10)

    y = network.outputs

    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    cost = ce

    correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    n_epoch = 10000
    learning_rate = 0.0001
    print_freq = 1

    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
        epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    sess.run(tf.initialize_all_variables())
    if resume:
        print("Load existing model " + "!"*10)
        saver = tf.train.Saver()
        saver.restore(sess, model_file_name)

    network.print_params()
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )        # enable all dropout/dropconnect/denoising layers
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                assert not np.isnan(err), 'Model diverged with cost = NaN'
                train_loss += err; train_acc += ac; n_batch += 1
            print("   train loss: %f" % (train_loss/ n_batch))
            print("   train acc: %f" % (train_acc/ n_batch))
            test_loss, test_acc, n_batch = 0, 0, 0
            for X_test_a, y_test_a in tl.iterate.minibatches(
                                    X_test, y_test, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
                feed_dict = {x: X_test_a, y_: y_test_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                test_loss += err; test_acc += ac; n_batch += 1
            print("   test loss: %f" % (test_loss/ n_batch))
            print("   test acc: %f" % (test_acc/ n_batch))
            try:
                pass
                # tl.visualize.CNN2d(network.all_params[0].eval(), second=10, saveable=True, name='cnn1_'+str(epoch+1), fig_idx=2012)
            except:
                raise Exception("# You should change visualize.CNN(), \
                if you want to save the feature images for different dataset")

        if (epoch + 1) % 1 == 0:
            print("Save model " + "!"*10);
            saver = tf.train.Saver()
            save_path = saver.save(sess, model_file_name)


def main_test_cnn_advanced():
    """Reimplementation of the TensorFlow official CIFAR-10 CNN tutorials:

    This model has 1,068,298 paramters, after few hours of training with GPU,
    accurcy of 86% was found.

    For simplified CNN layer see "Convolutional layer (Simplified)"
    in read the docs website.

    Links
    -------
    .. https://www.tensorflow.org/versions/r0.9/tutorials/deep_cnn/index.html
    .. https://github.com/tensorflow/tensorflow/tree/r0.9/tensorflow/models/image/cifar10

    Note
    ------
    The optimizers between official code and this code are different.

    Description
    -----------
    The images are processed as follows:
    .. They are cropped to 24 x 24 pixels, centrally for evaluation or randomly for training.
    .. They are approximately whitened to make the model insensitive to dynamic range.

    For training, we additionally apply a series of random distortions to
    artificially increase the data set size:
    .. Randomly flip the image from left to right.
    .. Randomly distort the image brightness.
    .. Randomly distort the image contrast.

    Speed Up
    ---------
    see `tutorial_cifar10_tfrecord.py`
    Reading images from disk and distorting them can use a non-trivial amount
    of processing time. To prevent these operations from slowing down training,
    we run them inside 16 separate threads which continuously fill a TensorFlow queue.
    """
    model_file_name = "model_cifar10_advanced.ckpt"
    resume = False # load model, resume from previous checkpoint?

    X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(
                                        shape=(-1, 32, 32, 3), plotable=False)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    print('X_train.shape', X_train.shape)   # (50000, 32, 32, 3)
    print('y_train.shape', y_train.shape)   # (50000,)
    print('X_test.shape', X_test.shape)     # (10000, 32, 32, 3)
    print('y_test.shape', y_test.shape)     # (10000,)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    sess = tf.InteractiveSession()

    batch_size = 128

    x = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])   # [batch_size, height, width, channels]
    x_crop = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])
    y_ = tf.placeholder(tf.int32, shape=[batch_size,])

    ## distorted images for training.
    distorted_images_op = tl.prepro.distorted_images(images=x, height=24, width=24)
    ## crop the central of images and whiten it for evaluation.
    central_images_op = tl.prepro.crop_central_whiten_images(images=x, height=24, width=24)

    ## You can display the distorted images and evaluation images as follow:
    # sess.run(tf.initialize_all_variables())
    # feed_dict={x: X_train[0:batch_size,:,:,:]}
    #
    # distorted_images, idx = sess.run(distorted_images_op, feed_dict=feed_dict)
    # tl.visualize.images2d(X_train[0:batch_size,:,:,:], second=1, saveable=False, name='cifar10', dtype=np.uint8, fig_idx=211)
    # tl.visualize.images2d(distorted_images, second=1, saveable=False, name='distorted_images', dtype=np.uint8, fig_idx=3032)
    #
    # central_images, idx = sess.run(central_images_op, feed_dict=feed_dict)
    # tl.visualize.images2d(central_images, second=10, saveable=False, name='central_images', dtype=None, fig_idx=419)
    # exit()

    # Network is the same with:
    #  https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/models/image/cifar10/cifar10.py
    #  using relu, the biases should be better to initialize to positive value.
    network = tl.layers.InputLayer(x_crop, name='input_layer')
    network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [5, 5, 3, 64],  # 64 features for each 5x5x3 patch
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=tf.truncated_normal_initializer(stddev=5e-2),
                        W_init_args={},
                        b_init=tf.constant_initializer(value=0.0),
                        b_init_args={},
                        name ='cnn_layer1')     # output: (batch_size, 24, 24, 64)
    network = tl.layers.PoolLayer(network,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='pool_layer1',)   # output: (batch_size, 12, 12, 64)
    network.outputs = tf.nn.lrn(network.outputs, 4, bias=1.0, alpha=0.001 / 9.0,
                                                    beta=0.75, name='norm1')
    network = tl.layers.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [5, 5, 64, 64], # 64 features for each 5x5 patch
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=tf.truncated_normal_initializer(stddev=5e-2),
                        W_init_args={},
                        b_init=tf.constant_initializer(value=0.1),
                        b_init_args={},
                        name ='cnn_layer2')     # output: (batch_size, 12, 12, 64)
    network.outputs = tf.nn.lrn(network.outputs, 4, bias=1.0, alpha=0.001 / 9.0,
                                                    beta=0.75, name='norm2')
    network = tl.layers.PoolLayer(network,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='pool_layer2')   # output: (batch_size, 6, 6, 64)
    network = tl.layers.FlattenLayer(network, name='flatten_layer')                        # output: (batch_size, 2304)
    network = tl.layers.DenseLayer(network, n_units=384, act = tf.nn.relu,
                        W_init=tf.truncated_normal_initializer(stddev=0.04),
                        W_init_args={},
                        b_init=tf.constant_initializer(value=0.1),
                        b_init_args={}, name='relu1')       # output: (batch_size, 384)
    network = tl.layers.DenseLayer(network, n_units=192, act = tf.nn.relu,
                        W_init=tf.truncated_normal_initializer(stddev=0.04),
                        W_init_args={},
                        b_init=tf.constant_initializer(value=0.1),
                        b_init_args={}, name='relu2')       # output: (batch_size, 192)
    network = tl.layers.DenseLayer(network, n_units=10, act = tf.identity,
                        W_init=tf.truncated_normal_initializer(stddev=1/192.0),
                        W_init_args={},
                        b_init = tf.constant_initializer(value=0.0),
                        name='output_layer')    # output: (batch_size, 10)
    y = network.outputs

    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    # L2 for the MLP, without this, the accuracy will be reduced by 15%.
    L2 = tf.contrib.layers.l2_regularizer(0.004)(network.all_params[4]) + \
            tf.contrib.layers.l2_regularizer(0.004)(network.all_params[6])
    cost = ce + L2

    # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
    correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # view you graph here
    # tensorboard --logdir=/tmp/cifar10_logs/train
    # http://0.0.0.0:6006
    # print('a')
    # merged = tf.merge_all_summaries()
    # train_writer = tf.train.SummaryWriter('/tmp/cifar10_logs/train', sess.graph)
    # sess.run(merged)
    # print('b')

    # train
    n_epoch = 50000
    learning_rate = 0.0001
    print_freq = 5

    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
        epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    sess.run(tf.initialize_all_variables())
    if resume:
        print("Load existing model " + "!"*10)
        saver = tf.train.Saver()
        saver.restore(sess, model_file_name)

    network.print_params()
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
            X_train_a = sess.run(distorted_images_op, feed_dict={x: X_train_a})[0][1:]  # preprocess the training images, took about 0.11s per batch (128)
            feed_dict = {x_crop: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )        # enable all dropout/dropconnect/denoising layers
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
                X_train_a = sess.run(central_images_op, feed_dict={x: X_train_a})[0][1:]  # preprocess the images for evaluation
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
                feed_dict = {x_crop: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                assert not np.isnan(err), 'Model diverged with cost = NaN'
                train_loss += err; train_acc += ac; n_batch += 1
            print("   train loss: %f" % (train_loss/ n_batch))
            print("   train acc: %f" % (train_acc/ n_batch))
            test_loss, test_acc, n_batch = 0, 0, 0
            for X_test_a, y_test_a in tl.iterate.minibatches(
                                    X_test, y_test, batch_size, shuffle=True):
                X_test_a = sess.run(central_images_op, feed_dict={x: X_test_a})[0][1:]  # preprocess the images for evaluation
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
                feed_dict = {x_crop: X_test_a, y_: y_test_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                test_loss += err; test_acc += ac; n_batch += 1
            print("   test loss: %f" % (test_loss/ n_batch))
            print("   test acc: %f" % (test_acc/ n_batch))
            try:
                pass
                # tl.visualize.CNN2d(network.all_params[0].eval(), second=10, saveable=True, name='cnn1_'+str(epoch+1), fig_idx=2012)
            except:
                raise Exception("# You should change visualize.CNN(), if you \
                        want to save the feature images for different dataset")

        if (epoch + 1) % (print_freq * 5) == 0:
            print("Save model " + "!"*10)
            saver = tf.train.Saver()
            save_path = saver.save(sess, model_file_name)


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess = tl.ops.set_gpu_fraction(sess, gpu_fraction = .5)
    try:
        """Without image distorting"""
        # main_test_cnn_naive()
        """With image distorting"""
        main_test_cnn_advanced()
        tl.ops.exit_tf(sess)   # close sess, tensorboard and nvidia-process
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        tl.ops.exit_tf(sess)
