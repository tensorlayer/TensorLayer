#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer import set_keep
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from sys import platform as _platform

def main_test_layers(model='relu'):
    X_train, y_train, X_val, y_val, X_test, y_test = tl.load_mnist_dataset(shape=(-1,784))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    sess = tf.InteractiveSession()

    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    if model == 'relu':
        network = tl.InputLayer(x, name='input_layer')
        network = tl.DropoutLayer(network, keep=0.8, name='drop1')
        network = tl.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
        network = tl.DropoutLayer(network, keep=0.5, name='drop2')
        network = tl.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')
        network = tl.DropoutLayer(network, keep=0.5, name='drop3')
        network = tl.DenseLayer(network, n_units=10, act = tl.identity, name='output_layer')
    elif model == 'resnet':
        network = tl.InputLayer(x, name='input_layer')
        network = tl.DropoutLayer(network, keep=0.8, name='drop1')
        network = tl.ResnetLayer(network, act = tf.nn.relu, name='resnet1')
        network = tl.DropoutLayer(network, keep=0.5, name='drop2')
        network = tl.ResnetLayer(network, act = tf.nn.relu, name='resnet2')
        network = tl.DropoutLayer(network, keep=0.5, name='drop3')
        network = tl.DenseLayer(network, act = tl.identity, name='output_layer')
    elif model == 'dropconnect':
        network = tl.InputLayer(x, name='input_layer')
        network = tl.DropconnectDenseLayer(network, keep = 0.8, n_units=800, act = tf.nn.relu, name='dropconnect_relu1')
        network = tl.DropconnectDenseLayer(network, keep = 0.5, n_units=800, act = tf.nn.relu, name='dropconnect_relu2')
        network = tl.DropconnectDenseLayer(network, keep = 0.5, n_units=10, act = tl.identity, name='output_layer')

    # attrs = vars(network)
    # print(', '.join("%s: %s\n" % item for item in attrs.items()))

    # print(network.all_drop)     # {'drop1': 0.8, 'drop2': 0.5, 'drop3': 0.5}
    # print(drop1, drop2, drop3)  # Tensor("Placeholder_2:0", dtype=float32) Tensor("Placeholder_3:0", dtype=float32) Tensor("Placeholder_4:0", dtype=float32)
    # exit()

    y = network.outputs
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    cost = ce

    # cost = cost + tl.maxnorm_regularizer(1.0)(network.all_params[0]) + tl.maxnorm_regularizer(1.0)(network.all_params[2])
    # cost = cost + tl.lo_regularizer(0.0001)(network.all_params[0]) + tl.lo_regularizer(0.0001)(network.all_params[2])
    # cost = cost + tl.maxnorm_o_regularizer(0.001)(network.all_params[0]) + tl.maxnorm_o_regularizer(0.001)(network.all_params[2])


    params = network.all_params
    # train
    n_epoch = 500
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 10
    # train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost)

    sess.run(tf.initialize_all_variables()) # initialize all variables

    network.print_params()
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )    # enable all dropout/dropconnect/denoising layers
            sess.run(train_op, feed_dict=feed_dict)

            # The optional feed_dict argument allows the caller to override the value of tensors in the graph. Each key in feed_dict can be one of the following types:
            # If the key is a Tensor, the value may be a Python scalar, string, list, or numpy ndarray that can be converted to the same dtype as that tensor. Additionally, if the key is a placeholder, the shape of the value will be checked for compatibility with the placeholder.
            # If the key is a SparseTensor, the value should be a SparseTensorValue.

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            dp_dict = tl.Layer.dict_to_one( network.all_drop ) # disable all dropout/dropconnect/denoising layers
            feed_dict = {x: X_train, y_: y_train}
            feed_dict.update(dp_dict)
            print("   train loss: %f" % sess.run(cost, feed_dict=feed_dict))
            dp_dict = tl.Layer.dict_to_one( network.all_drop )
            feed_dict = {x: X_val, y_: y_val}
            feed_dict.update(dp_dict)
            print("   val loss: %f" % sess.run(cost, feed_dict=feed_dict))
            print("   val acc: %f" % np.mean(y_val == sess.run(y_op, feed_dict=feed_dict)))
            try:
                tl.visualize_W(network.all_params[0].eval(), second=10, saveable=True, name='w1_'+str(epoch+1), fig_idx=2012)
            except:
                raise Exception("You should change visualize_W(), if you want to save the feature images for different dataset")

    print('Evaluation')
    dp_dict = tl.Layer.dict_to_one( network.all_drop )
    feed_dict = {x: X_test, y_: y_test}
    feed_dict.update(dp_dict)
    print("   test loss: %f" % sess.run(cost, feed_dict=feed_dict))
    print("   test acc: %f" % np.mean(y_test == sess.run(y_op, feed_dict=feed_dict)))

    # Add ops to save and restore all the variables.
    # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
    saver = tf.train.Saver()
    # you may want to save the model
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()

def main_test_denoise_AE(model='relu'):
    X_train, y_train, X_val, y_val, X_test, y_test = tl.load_mnist_dataset(shape=(-1,784))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    sess = tf.InteractiveSession()

    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    print("Build Network")
    if model == 'relu':
        network = tl.InputLayer(x, name='input_layer')
        network = tl.DropoutLayer(network, keep=0.5, name='denoising1')    # if drop some inputs, it is denoise AE
        network = tl.DenseLayer(network, n_units=196, act = tf.nn.relu, name='relu1')
        recon_layer1 = tl.ReconLayer(network, x_recon=x, n_units=784, act = tf.nn.softplus, name='recon_layer1')
    elif model == 'sigmoid':
        # sigmoid - set keep to 1.0, if you want a vanilla Autoencoder
        network = tl.InputLayer(x, name='input_layer')
        network = tl.DropoutLayer(network, keep=0.5, name='denoising1')
        network = tl.DenseLayer(network, n_units=200, act=tf.nn.sigmoid, name='sigmoid1')
        recon_layer1 = tl.ReconLayer(network, x_recon=x, n_units=784, act=tf.nn.sigmoid, name='recon_layer1')

    ## ready to train
    sess.run(tf.initialize_all_variables())

    ## print all params
    print("All Network Params")
    network.print_params()

    ## pretrain
    print("Pre-train Layer 1")
    recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=200, batch_size=128, print_freq=10, save=True, save_name='w1pre_')
        # recon_layer1.pretrain(sess, X_train=X_train, X_val=X_val, denoise_name=None, n_epoch=1000, batch_size=128, print_freq=10)

    # Add ops to save and restore all the variables.
    # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
    saver = tf.train.Saver()
    # you may want to save the model
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()

def main_test_stacked_denoise_AE(model='relu'):
    # Load MNIST dataset
    X_train, y_train, X_val, y_val, X_test, y_test = tl.load_mnist_dataset(shape=(-1,784))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    if model == 'relu':
        act = tf.nn.relu
        act_recon = tf.nn.softplus
    elif model == 'sigmoid':
        act = tf.nn.sigmoid
        act_recon = act

    # Define network
    print("\nBuild Network")
    network = tl.InputLayer(x, name='input_layer')
    # denoise layer for AE
    network = tl.DropoutLayer(network, keep=0.5, name='denoising1')
    # 1st layer
    network = tl.DropoutLayer(network, keep=0.8, name='drop1')
    network = tl.DenseLayer(network, n_units=800, act = act, name=model+'1')
    x_recon1 = network.outputs
    recon_layer1 = tl.ReconLayer(network, x_recon=x, n_units=784, act = act_recon, name='recon_layer1')
    # 2nd layer
    network = tl.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.DenseLayer(network, n_units=800, act = act, name=model+'2')
    recon_layer2 = tl.ReconLayer(network, x_recon=x_recon1, n_units=800, act = act_recon, name='recon_layer2')
    # 3rd layer
    network = tl.DropoutLayer(network, keep=0.5, name='drop3')
    network = tl.DenseLayer(network, n_units=10, act = tl.identity, name='output_layer')

    # Define fine-tune process
    y = network.outputs
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    cost = ce

    n_epoch = 200
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 10

    train_params = network.all_params

        # train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    train_op = tf.train.AdamOptimizer(learning_rate , beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    # Initialize all variables including weights, biases and the variables in train_op
    sess.run(tf.initialize_all_variables())

    # Pre-train
    print("\nAll Network Params before pre-train")
    network.print_params()
    print("\nPre-train Layer 1")
    recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10, save=True, save_name='w1pre_')
    print("\nPre-train Layer 2")
    recon_layer2.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10, save=False)
    print("\nAll Network Params after pre-train")
    network.print_params()

    # Fine-tune
    print("\nFine-tune Network")
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )        # enable all dropout/dropconnect/denoising layers
            feed_dict[set_keep['denoising1']] = 1    # disable denoising layer
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.Layer.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print("   train loss: %f" % (train_loss/ n_batch))
            print("   train acc: %f" % (train_acc/ n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate_minibatches(X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.Layer.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print("   val loss: %f" % (val_loss/ n_batch))
            print("   val acc: %f" % (val_acc/ n_batch))
            try:
                tl.visualize_W(network.all_params[0].eval(), second=10, saveable=True, name='w1_'+str(epoch+1), fig_idx=2012)
            except:
                raise Exception("# You should change visualize_W(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate_minibatches(X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.Layer.dict_to_one( network.all_drop )    # disable all dropout layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print("   test loss: %f" % (test_loss/n_batch))
    print("   test acc: %f" % (test_acc/n_batch))
        # print("   test acc: %f" % np.mean(y_test == sess.run(y_op, feed_dict=feed_dict)))

    # Add ops to save and restore all the variables.
    # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
    saver = tf.train.Saver()
    # you may want to save the model
    save_path = saver.save(sess, "model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()

def main_test_cnn_layer():
    '''
        Reimplementation of the tensorflow official MNIST CNN tutorials:
        # https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py
    '''
    X_train, y_train, X_val, y_val, X_test, y_test = tl.load_mnist_dataset(shape=(-1, 28, 28, 1))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])   # [batch_size, height, width, channels]
    y_ = tf.placeholder(tf.int64, shape=[None,])

    network = tl.InputLayer(x, name='input_layer')
    network = tl.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        name ='cnn_layer1')     # output: (?, 28, 28, 32)
    network = tl.Pool2dLayer(network,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='pool_layer1',)   # output: (?, 14, 14, 32)
    network = tl.Conv2dLayer(network,
                        act = tf.nn.relu,
                        shape = [5, 5, 32, 64], # 64 features for each 5x5 patch
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        name ='cnn_layer2')     # output: (?, 14, 14, 64)
    network = tl.Pool2dLayer(network,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='pool_layer2',)   # output: (?, 7, 7, 64)
    network = tl.FlattenLayer(network, name='flatten_layer')                                # output: (?, 3136)
    network = tl.DropoutLayer(network, keep=0.5, name='drop1')                              # output: (?, 3136)
    network = tl.DenseLayer(network, n_units=256, act = tf.nn.relu, name='relu1')           # output: (?, 256)
    network = tl.DropoutLayer(network, keep=0.5, name='drop2')                              # output: (?, 256)
    network = tl.DenseLayer(network, n_units=10, act = tl.identity, name='output_layer')    # output: (?, 10)

    y = network.outputs

    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    cost = ce

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    n_epoch = 100
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 10

    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    sess.run(tf.initialize_all_variables())
    network.print_params()
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )        # enable all dropout/dropconnect/denoising layers
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.Layer.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err; train_acc += ac; n_batch += 1
            print("   train loss: %f" % (train_loss/ n_batch))
            print("   train acc: %f" % (train_acc/ n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate_minibatches(X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.Layer.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err; val_acc += ac; n_batch += 1
            print("   val loss: %f" % (val_loss/ n_batch))
            print("   val acc: %f" % (val_acc/ n_batch))
            # try:
            #     tl.visualize_CNN(network.all_params[0].eval(), second=10, saveable=True, name='w1_'+str(epoch+1), fig_idx=2012)
            # except:
            #     raise Exception("# You should change visualize_CNN(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate_minibatches(X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.Layer.dict_to_one( network.all_drop )    # disable all dropout layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err; test_acc += ac; n_batch += 1
    print("   test loss: %f" % (test_loss/n_batch))
    print("   test acc: %f" % (test_acc/n_batch))


if __name__ == '__main__':
    sess = tl.set_gpu_fraction(gpu_fraction = 0.3)
    try:
        main_test_layers(model='relu')                # model = relu, resnet, dropconnect
        tl.clear_all()
        main_test_denoise_AE(model='relu')            # model = relu, sigmoid
        tl.clear_all()
        main_test_stacked_denoise_AE(model='relu')    # model = relu, sigmoid
        tl.clear_all()
        main_test_cnn_layer()
        tl.exit_tf(sess)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        tl.exit_tf(sess)










# act       / n_units / pre-train              / train                     / test acc

# relu      / 800-800 / None                   / e500 adam 1e-4 b128 d255  / 98.77  98.80
#                     / None                   / e500 adam 1e-4 b128 d255  / 98.77
#                                                   max-norm               /
#                     / None                   / e500 adam 1e-4 b128 d255  / 98.58
#                                                   max-norm out           /
#                     / e100 adam 1e-3 b128 d5 / e500 adam 1e-4 b128 d255  / 98.7881      Note: relu can reach it's best performance without pre-train
#                     / P_o instead of L2_w    / e500 adam 1e-4 b128 d255  / 98.6779

# sig       / 800-800 / None                   / e500 adam 1e-4 b128 d255  / 98.6579
#                     / e100 adam 1e-3 b128 d5 / e500 adam 1e-4 b128 d255  / 98.6879
#                     / P_o instead of L2_w    / e500 adam 1e-4 b128 d255  /

# resnet+relu/784-784 / None                   / e500 adam 1e-4 b128 d255  / 98.73
#                     / e100 adam 1e-3 b128 d5 / e500 adam 1e-4 b128 d255  /

# dropcon+relu/800-800/ None                   / e150 adam 1e-4 b128 d255  / 98.00 no more increase
#                                                P_o instead of L2_w 0.001
#                     / e100 adam 1e-3 b128 d5 / e500 adam 1e-4 b128 d255  /
