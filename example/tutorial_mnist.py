#! /usr/bin/python
# -*- coding: utf-8 -*-

import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep


"""Examples of Stacked Denoising Autoencoder, Dropout, Dropconnect and CNN.

This tutorial uses placeholder to control all keeping probabilities,
so we need to set the non-one probabilities during training, and set them to 1
during evaluating and testing.

$ Set keeping probabilities.
>>> feed_dict = {x: X_train_a, y_: y_train_a}
>>> feed_dict.update( network.all_drop )

$ Set all keeping probabilities to 1 for evaluating and testing.
>>> dp_dict = tl.utils.dict_to_one( network.all_drop )
>>> feed_dict = {x: X_train_a, y_: y_train_a}
>>> feed_dict.update(dp_dict)

Alternatively, if you don't want to use placeholder to control them, you can
build different inferences for training, evaluating and testing,
and all inferences share the same model parameters.
(see tutorial_ptb_lstm.py)

"""

def main_test_layers(model='relu'):
    X_train, y_train, X_val, y_val, X_test, y_test = \
                                    tl.files.load_mnist_dataset(shape=(-1,784))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

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

    # Note: the softmax is implemented internally in tl.cost.cross_entropy(y, y_)
    # to speed up computation, so we use identity in the last layer.
    # see tf.nn.sparse_softmax_cross_entropy_with_logits()
    if model == 'relu':
        network = tl.layers.InputLayer(x, name='input')
        network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
        network = tl.layers.DenseLayer(network, n_units=800,
                                        act=tf.nn.relu, name='relu1')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
        network = tl.layers.DenseLayer(network, n_units=800,
                                        act=tf.nn.relu, name='relu2')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
        network = tl.layers.DenseLayer(network, n_units=10,
                                        act=tf.identity,
                                        name='output')
    elif model == 'dropconnect':
        network = tl.layers.InputLayer(x, name='input')
        network = tl.layers.DropconnectDenseLayer(network, keep = 0.8,
                                                n_units=800, act = tf.nn.relu,
                                                name='dropconnect_relu1')
        network = tl.layers.DropconnectDenseLayer(network, keep = 0.5,
                                                n_units=800, act = tf.nn.relu,
                                                name='dropconnect_relu2')
        network = tl.layers.DropconnectDenseLayer(network, keep = 0.5,
                                                n_units=10,
                                                act=tf.identity,
                                                name='output')

    # To print all attributes of a Layer.
    # attrs = vars(network)
    # print(', '.join("%s: %s\n" % item for item in attrs.items()))
    # print(network.all_drop)     # {'drop1': 0.8, 'drop2': 0.5, 'drop3': 0.5}

    y = network.outputs
    cost = tl.cost.cross_entropy(y, y_, name='xentropy')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_op = tf.argmax(tf.nn.softmax(y), 1)

    # You can add more penalty to the cost function as follow.
    # cost = cost + tl.cost.maxnorm_regularizer(1.0)(network.all_params[0]) + tl.cost.maxnorm_regularizer(1.0)(network.all_params[2])
    # cost = cost + tl.cost.lo_regularizer(0.0001)(network.all_params[0]) + tl.cost.lo_regularizer(0.0001)(network.all_params[2])
    # cost = cost + tl.cost.maxnorm_o_regularizer(0.001)(network.all_params[0]) + tl.cost.maxnorm_o_regularizer(0.001)(network.all_params[2])

    params = network.all_params
    # train
    n_epoch = 100
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 5
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                epsilon=1e-08, use_locking=False).minimize(cost)

    tl.layers.initialize_global_variables(sess)

    network.print_params()
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train,
                                                    batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )    # enable dropout or dropconnect layers
            sess.run(train_op, feed_dict=feed_dict)

            # The optional feed_dict argument allows the caller to override the value of tensors in the graph. Each key in feed_dict can be one of the following types:
            # If the key is a Tensor, the value may be a Python scalar, string, list, or numpy ndarray that can be converted to the same dtype as that tensor. Additionally, if the key is a placeholder, the shape of the value will be checked for compatibility with the placeholder.
            # If the key is a SparseTensor, the value should be a SparseTensorValue.

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err; train_acc += ac; n_batch += 1
            print("   train loss: %f" % (train_loss/ n_batch))
            # print("   train acc: %f" % (train_acc/ n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(
                                        X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err; val_acc += ac; n_batch += 1
            print("   val loss: %f" % (val_loss/ n_batch))
            print("   val acc: %f" % (val_acc/ n_batch))
            try:
                # You can visualize the weight of 1st hidden layer as follow.
                tl.vis.W(network.all_params[0].eval(), second=10,
                                        saveable=True, shape=[28, 28],
                                        name='w1_'+str(epoch+1), fig_idx=2012)
                # You can also save the weight of 1st hidden layer to .npz file.
                # tl.files.save_npz([network.all_params[0]] , name='w1'+str(epoch+1)+'.npz')
            except:
                print("You should change vis.W(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(
                                X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err; test_acc += ac; n_batch += 1
    print("   test loss: %f" % (test_loss/n_batch))
    print("   test acc: %f" % (test_acc/n_batch))

    # Add ops to save and restore all the variables, including variables for training.
    # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)


    # You can also save the parameters into .npz file.
    tl.files.save_npz(network.all_params , name='model.npz')
    # You can only save one parameter as follow.
    # tl.files.save_npz([network.all_params[0]] , name='model.npz')
    # Then, restore the parameters as follow.
    # load_params = tl.files.load_npz(path='', name='model.npz')
    # tl.files.assign_params(sess, load_params, network)

    # In the end, close TensorFlow session.
    sess.close()

def main_test_denoise_AE(model='relu'):
    X_train, y_train, X_val, y_val, X_test, y_test = \
                                tl.files.load_mnist_dataset(shape=(-1,784))

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
        network = tl.layers.InputLayer(x, name='input')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='denoising1')    # if drop some inputs, it is denoise AE
        network = tl.layers.DenseLayer(network, n_units=196,
                                    act = tf.nn.relu, name='relu1')
        recon_layer1 = tl.layers.ReconLayer(network, x_recon=x, n_units=784,
                                    act = tf.nn.softplus, name='recon_layer1')
    elif model == 'sigmoid':
        # sigmoid - set keep to 1.0, if you want a vanilla Autoencoder
        network = tl.layers.InputLayer(x, name='input')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='denoising1')
        network = tl.layers.DenseLayer(network, n_units=196,
                                    act=tf.nn.sigmoid, name='sigmoid1')
        recon_layer1 = tl.layers.ReconLayer(network, x_recon=x, n_units=784,
                                    act=tf.nn.sigmoid, name='recon_layer1')

    ## ready to train
    tl.layers.initialize_global_variables(sess)

    ## print all params
    print("All Network Params")
    network.print_params()

    ## pretrain
    print("Pre-train Layer 1")
    recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val,
                            denoise_name='denoising1', n_epoch=200,
                            batch_size=128, print_freq=10, save=True,
                            save_name='w1pre_')
    # You can also disable denoisong by setting denoise_name=None.
    # recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val,
    #                           denoise_name=None, n_epoch=500, batch_size=128,
    #                           print_freq=10, save=True, save_name='w1pre_')

    # Add ops to save and restore all the variables.
    # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
    saver = tf.train.Saver()
    # you may want to save the model
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()

def main_test_stacked_denoise_AE(model='relu'):
    X_train, y_train, X_val, y_val, X_test, y_test = \
                                tl.files.load_mnist_dataset(shape=(-1,784))

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
    network = tl.layers.InputLayer(x, name='input')
    # denoise layer for AE
    network = tl.layers.DropoutLayer(network, keep=0.5, name='denoising1')
    # 1st layer
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=800, act=act, name=model+'1')
    x_recon1 = network.outputs
    recon_layer1 = tl.layers.ReconLayer(network, x_recon=x, n_units=784,
                                        act=act_recon, name='recon_layer1')
    # 2nd layer
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=800, act = act, name=model+'2')
    recon_layer2 = tl.layers.ReconLayer(network, x_recon=x_recon1, n_units=800,
                                        act=act_recon, name='recon_layer2')
    # 3rd layer
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    network = tl.layers.DenseLayer(network, 10, act=tf.identity, name='output')

    # Define fine-tune process
    y = network.outputs
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    cost = tl.cost.cross_entropy(y, y_, name='cost')

    n_epoch = 200
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 10

    train_params = network.all_params

        # train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    train_op = tf.train.AdamOptimizer(learning_rate , beta1=0.9, beta2=0.999,
        epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    # Initialize all variables including weights, biases and the variables in train_op
    tl.layers.initialize_global_variables(sess)

    # Pre-train
    print("\nAll Network Params before pre-train")
    network.print_params()
    print("\nPre-train Layer 1")
    recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val,
                            denoise_name='denoising1', n_epoch=100,
                            batch_size=128, print_freq=10, save=True,
                            save_name='w1pre_')
    print("\nPre-train Layer 2")
    recon_layer2.pretrain(sess, x=x, X_train=X_train, X_val=X_val,
                            denoise_name='denoising1', n_epoch=100,
                            batch_size=128, print_freq=10, save=False)
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
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )     # enable noise layers
            feed_dict[set_keep['denoising1']] = 1    # disable denoising layer
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print("   train loss: %f" % (train_loss/ n_batch))
            print("   train acc: %f" % (train_acc/ n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(
                                        X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print("   val loss: %f" % (val_loss/ n_batch))
            print("   val acc: %f" % (val_acc/ n_batch))
            try:
                # visualize the 1st hidden layer during fine-tune
                tl.vis.W(network.all_params[0].eval(), second=10,
                            saveable=True, shape=[28, 28],
                            name='w1_'+str(epoch+1), fig_idx=2012)
            except:
                print("You should change vis.W(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(
                                X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
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
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()

def main_test_cnn_layer():
    """Reimplementation of the TensorFlow official MNIST CNN tutorials:
    - https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py

    More TensorFlow official CNN tutorials can be found here:
    - tutorial_cifar10.py
    - https://www.tensorflow.org/versions/master/tutorials/deep_cnn/index.html

    - For simplified CNN layer see "Convolutional layer (Simplified)"
      in read the docs website.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = \
                    tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

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

    # Define the batchsize at the begin, you can give the batchsize in x and y_
    # rather than 'None', this can allow TensorFlow to apply some optimizations
    # – especially for convolutional layers.
    batch_size = 128

    x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])   # [batch_size, height, width, channels]
    y_ = tf.placeholder(tf.int64, shape=[batch_size,])

    network = tl.layers.InputLayer(x, name='input')
    ## Professional conv API for tensorflow user
    # network = tl.layers.Conv2dLayer(network,
    #                     act = tf.nn.relu,
    #                     shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
    #                     strides=[1, 1, 1, 1],
    #                     padding='SAME',
    #                     name ='cnn1')     # output: (?, 28, 28, 32)
    # network = tl.layers.PoolLayer(network,
    #                     ksize=[1, 2, 2, 1],
    #                     strides=[1, 2, 2, 1],
    #                     padding='SAME',
    #                     pool = tf.nn.max_pool,
    #                     name ='pool1',)   # output: (?, 14, 14, 32)
    # network = tl.layers.Conv2dLayer(network,
    #                     act = tf.nn.relu,
    #                     shape = [5, 5, 32, 64], # 64 features for each 5x5 patch
    #                     strides=[1, 1, 1, 1],
    #                     padding='SAME',
    #                     name ='cnn2')     # output: (?, 14, 14, 64)
    # network = tl.layers.PoolLayer(network,
    #                     ksize=[1, 2, 2, 1],
    #                     strides=[1, 2, 2, 1],
    #                     padding='SAME',
    #                     pool = tf.nn.max_pool,
    #                     name ='pool2',)   # output: (?, 7, 7, 64)
    ## Simplified conv API for beginner (the same with the above layers)
    network = tl.layers.Conv2d(network, 32, (5, 5), (1, 1),
            act=tf.nn.relu, padding='SAME', name='cnn1')
    network = tl.layers.MaxPool2d(network, (2, 2), (2, 2),
            padding='SAME', name='pool1')
    network = tl.layers.Conv2d(network, 64, (5, 5), (1, 1),
            act=tf.nn.relu, padding='SAME', name='cnn2')
    network = tl.layers.MaxPool2d(network, (2, 2), (2, 2),
            padding='SAME', name='pool2')
    ## end of conv
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
    network = tl.layers.DenseLayer(network, 256, act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, 10, act=tf.identity, name='output')

    y = network.outputs

    cost = tl.cost.cross_entropy(y, y_, 'cost')

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    n_epoch = 200
    learning_rate = 0.0001
    print_freq = 10

    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
        epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    tl.layers.initialize_global_variables(sess)
    network.print_params()
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )        # enable noise layers
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err; train_acc += ac; n_batch += 1
            print("   train loss: %f" % (train_loss/ n_batch))
            print("   train acc: %f" % (train_acc/ n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(
                                        X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err; val_acc += ac; n_batch += 1
            print("   val loss: %f" % (val_loss/ n_batch))
            print("   val acc: %f" % (val_acc/ n_batch))
            try:
                tl.vis.CNN2d(network.all_params[0].eval(),
                                    second=10, saveable=True,
                                    name='cnn1_'+str(epoch+1), fig_idx=2012)
            except:
                print("You should change vis.CNN(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(
                                X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err; test_acc += ac; n_batch += 1
    print("   test loss: %f" % (test_loss/n_batch))
    print("   test acc: %f" % (test_acc/n_batch))



if __name__ == '__main__':
    sess = tf.InteractiveSession()
    """Dropout and Dropconnect"""
    main_test_layers(model='relu')                # model = relu, dropconnect
    """Single Denoising Autoencoder"""
    # main_test_denoise_AE(model='sigmoid')       # model = relu, sigmoid
    """Stacked Denoising Autoencoder"""
    # main_test_stacked_denoise_AE(model='relu')  # model = relu, sigmoid
    """CNN"""
    # main_test_cnn_layer()
