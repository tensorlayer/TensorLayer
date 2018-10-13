#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Examples of Stacked Denoising Autoencoder, Dropout, Dropconnect and CNN.

- Multi-layer perceptron (MNIST) - Classification task, see tutorial_mnist_simple.py
  https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_simple.py

- Multi-layer perceptron (MNIST) - Classification using Iterator, see:
  method1 : https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mlp_dropout1.py
  method2 : https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mlp_dropout2.py

"""

import time
import tensorflow as tf
import tensorlayer as tl

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


def main_test_layers(model='relu'):
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

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
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    # Note: the softmax is implemented internally in tl.cost.cross_entropy(y, y_)
    # to speed up computation, so we use identity in the last layer.
    # see tf.nn.sparse_softmax_cross_entropy_with_logits()
    if model == 'relu':
        net = tl.layers.Input(name='input')(x)
        net = tl.layers.Dropout(keep=0.8, name='drop1')(net)
        net = tl.layers.Dense(n_units=800, act=tf.nn.relu, name='relu1')(net)
        net = tl.layers.Dropout(keep=0.5, name='drop2')(net)
        net = tl.layers.Dense(n_units=800, act=tf.nn.relu, name='relu2')(net)
        net = tl.layers.Dropout(keep=0.5, name='drop3')(net)
        net = tl.layers.Dense(n_units=10, act=None, name='output')(net)
    elif model == 'dropconnect':
        net = tl.layers.Input(name='input')(x)
        net = tl.layers.DropconnectDense(keep=0.8, n_units=800, act=tf.nn.relu, name='dropconnect1')(net)
        net = tl.layers.DropconnectDense(keep=0.5, n_units=800, act=tf.nn.relu, name='dropconnect2')(net)
        net = tl.layers.DropconnectDense(keep=0.5, n_units=10, act=None, name='output')(net)

    # To print all attributes of a Layer.
    # attrs = vars(net)
    # print(', '.join("%s: %s\n" % item for item in attrs.items()))
    # print(net.all_drop)     # {'drop1': 0.8, 'drop2': 0.5, 'drop3': 0.5}

    y = net.outputs
    cost = tl.cost.cross_entropy(y, y_, name='xentropy')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # y_op = tf.argmax(tf.nn.softmax(y), 1)

    # You can add more penalty to the cost function as follow.
    # cost = cost + tl.cost.maxnorm_regularizer(1.0)(net.all_weights[0]) + tl.cost.maxnorm_regularizer(1.0)(net.all_weights[2])
    # cost = cost + tl.cost.lo_regularizer(0.0001)(net.all_weights[0]) + tl.cost.lo_regularizer(0.0001)(net.all_weights[2])
    # cost = cost + tl.cost.maxnorm_o_regularizer(0.001)(net.all_weights[0]) + tl.cost.maxnorm_o_regularizer(0.001)(net.all_weights[2])

    # train
    n_epoch = 100
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 5
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    tl.layers.initialize_global_variables(sess)

    net.print_weights()
    net.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)  # enable dropout or dropconnect layers
            sess.run(train_op, feed_dict=feed_dict)

            # The optional feed_dict argument allows the caller to override the value of tensors in the graph. Each key in feed_dict can be one of the following types:
            # If the key is a Tensor, the value may be a Python scalar, string, list, or numpy ndarray that can be converted to the same dtype as that tensor. Additionally, if the key is a placeholder, the shape of the value will be checked for compatibility with the placeholder.
            # If the key is a SparseTensor, the value should be a SparseTensorValue.

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print("   train loss: %f" % (train_loss / n_batch))
            # print("   train acc: %f" % (train_acc/ n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print("   val loss: %f" % (val_loss / n_batch))
            print("   val acc: %f" % (val_acc / n_batch))
            # try:
            #     # You can visualize the weight of 1st hidden layer as follow.
            #     tl.vis.draw_weights(net.all_weights[0].eval(), second=10, saveable=True, shape=[28, 28], name='w1_' + str(epoch + 1), fig_idx=2012)
            #     # You can also save the weight of 1st hidden layer to .npz file.
            #     # tl.files.save_npz([net.all_weights[0]] , name='w1'+str(epoch+1)+'.npz')
            # except:  # pylint: disable=bare-except
            #     print("You should change vis.draw_weights(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print("   test loss: %f" % (test_loss / n_batch))
    print("   test acc: %f" % (test_acc / n_batch))

    # Add ops to save and restore all the variables, including variables for training.
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)

    # You can also save the parameters into .npz file.
    tl.files.save_npz(net.all_weights, name='model.npz')
    # You can only save one parameter as follow.
    # tl.files.save_npz([net.all_weights[0]] , name='model.npz')
    # Then, restore the parameters as follow.
    # load_weights = tl.files.load_npz(path='', name='model.npz')
    # tl.files.assign_weights(sess, load_weights, net)

    # In the end, close TensorFlow session.
    sess.close()


def main_test_denoise_AE(model='relu'):
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

    sess = tf.InteractiveSession()

    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')

    print("Build net")
    if model == 'relu':
        net = tl.layers.Input(name='input')(x)
        net = tl.layers.Dropout(keep=0.5, name='denoising1')(net)  # if drop some inputs, it is denoise AE
        net = tl.layers.Dense(n_units=196, act=tf.nn.relu, name='relu1')(net)
        recon_layer1 = tl.layers.Recon(x_recon=x, n_units=784, act=tf.nn.softplus, name='recon_layer1')(net)
    elif model == 'sigmoid':
        # sigmoid - set keep to 1.0, if you want a vanilla Autoencoder
        net = tl.layers.Input(name='input')(x)
        net = tl.layers.Dropout(keep=0.5, name='denoising1')(net)
        net = tl.layers.Dense(n_units=196, act=tf.nn.sigmoid, name='sigmoid1')(net)
        recon_layer1 = tl.layers.Recon(x_recon=x, n_units=784, act=tf.nn.sigmoid, name='recon_layer1')(net)

    # ready to train
    tl.layers.initialize_global_variables(sess)

    # print all params
    print("All net Params")
    net.print_weights()

    # pretrain
    print("Pre-train Layer 1")
    recon_layer1.pretrain(
        sess,
        x=x,
        X_train=X_train,
        X_val=X_val,
        denoise_name='denoising1',
        n_epoch=200,
        batch_size=128,
        print_freq=10,
        save=True,
        save_name='w1pre_'
    )
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
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    if model == 'relu':
        act = tf.nn.relu
        act_recon = tf.nn.softplus
    elif model == 'sigmoid':
        act = tf.nn.sigmoid
        act_recon = act

    # Define net
    print("\nBuild net")
    net = tl.layers.Input(name='input')(x)
    # denoise layer for AE
    net = tl.layers.Dropout(keep=0.5, name='denoising1')(net)
    # 1st layer
    net = tl.layers.Dropout(keep=0.8, name='drop1')(net)
    net = tl.layers.Dense(n_units=800, act=act, name=model + '1')(net)
    x_recon1 = net.outputs
    recon_layer1 = tl.layers.ReconLayer(x_recon=x, n_units=784, act=act_recon, name='recon_layer1')(net)
    # 2nd layer
    net = tl.layers.Dropout(keep=0.5, name='drop2')(net)
    net = tl.layers.Dense(n_units=800, act=act, name=model + '2')(net)
    recon_layer2 = tl.layers.Recon(x_recon=x_recon1, n_units=800, act=act_recon, name='recon_layer2')(net)
    # 3rd layer
    net = tl.layers.Dropout(keep=0.5, name='drop3')(net)
    net = tl.layers.Dense(10, act=None, name='output')(net)

    # Define fine-tune process
    y = net.outputs
    cost = tl.cost.cross_entropy(y, y_, name='cost')

    n_epoch = 200
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 10

    train_weights = net.all_weights

    # train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_weights)

    # Initialize all variables including weights, biases and the variables in train_op
    tl.layers.initialize_global_variables(sess)

    # Pre-train
    print("\nAll net Params before pre-train")
    net.print_weights()
    print("\nPre-train Layer 1")
    recon_layer1.pretrain(
        sess,
        x=x,
        X_train=X_train,
        X_val=X_val,
        denoise_name='denoising1',
        n_epoch=100,
        batch_size=128,
        print_freq=10,
        save=True,
        save_name='w1pre_'
    )
    print("\nPre-train Layer 2")
    recon_layer2.pretrain(
        sess,
        x=x,
        X_train=X_train,
        X_val=X_val,
        denoise_name='denoising1',
        n_epoch=100,
        batch_size=128,
        print_freq=10,
        save=False
    )
    print("\nAll net Params after pre-train")
    net.print_weights()

    # Fine-tune
    print("\nFine-tune net")
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)  # enable noise layers
            feed_dict[tl.layers.LayersConfig.set_keep['denoising1']] = 1  # disable denoising layer
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print("   val loss: %f" % (val_loss / n_batch))
            print("   val acc: %f" % (val_acc / n_batch))
            # try:
            #     # visualize the 1st hidden layer during fine-tune
            #     tl.vis.draw_weights(net.all_weights[0].eval(), second=10, saveable=True, shape=[28, 28], name='w1_' + str(epoch + 1), fig_idx=2012)
            # except:  # pylint: disable=bare-except
            #     print("You should change vis.draw_weights(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print("   test loss: %f" % (test_loss / n_batch))
    print("   test acc: %f" % (test_acc / n_batch))
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
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

    sess = tf.InteractiveSession()

    # Define the batchsize at the begin, you can give the batchsize in x and y_
    # rather than 'None', this can allow TensorFlow to apply some optimizations
    # – especially for convolutional layers.
    batch_size = 128

    x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])  # [batch_size, height, width, channels]
    y_ = tf.placeholder(tf.int64, shape=[batch_size])

    net = tl.layers.Input(name='input')(x)
    net = tl.layers.Conv2d(32, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn1')(net)
    net = tl.layers.MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool1')(net)
    net = tl.layers.Conv2d(64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn2')(net)
    net = tl.layers.MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool2')(net)
    # end of conv
    net = tl.layers.Flatten(name='flatten')(net)
    net = tl.layers.Dropout(keep=0.5, name='drop1')(net)
    net = tl.layers.Dense(256, act=tf.nn.relu, name='relu1')(net)
    net = tl.layers.Dropout(keep=0.5, name='drop2')(net)
    net = tl.layers.Dense(10, act=None, name='output')(net)

    y = net.outputs

    cost = tl.cost.cross_entropy(y, y_, 'cost')

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    n_epoch = 200
    learning_rate = 0.0001
    print_freq = 10

    train_weights = net.all_weights
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_weights)

    tl.layers.initialize_global_variables(sess)
    net.print_weights()
    net.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)  # enable noise layers
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print("   val loss: %f" % (val_loss / n_batch))
            print("   val acc: %f" % (val_acc / n_batch))
            # try:
            #     tl.vis.CNN2d(net.all_weights[0].eval(), second=10, saveable=True, name='cnn1_' + str(epoch + 1), fig_idx=2012)
            # except:  # pylint: disable=bare-except
            #     print("You should change vis.CNN(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(net.all_drop)  # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print("   test loss: %f" % (test_loss / n_batch))
    print("   test acc: %f" % (test_acc / n_batch))


if __name__ == '__main__':
    sess = tf.InteractiveSession()

    # Dropout and Dropconnect
    # main_test_layers(model='relu')  # model = relu, dropconnect

    # Single Denoising Autoencoder
    # main_test_denoise_AE(model='sigmoid')       # model = relu, sigmoid

    # Stacked Denoising Autoencoder
    # main_test_stacked_denoise_AE(model='relu')  # model = relu, sigmoid

    # CNN
    main_test_cnn_layer()
