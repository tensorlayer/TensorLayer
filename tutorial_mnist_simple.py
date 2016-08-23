#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time

"""Examples of MLP.

tensorflow (0.9.0)
"""

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = \
                                    tl.files.load_mnist_dataset(shape=(-1,784))

    n_epoch = 200
    batch_size = 500
    learning_rate = 0.0001
    print_freq = 10
    is_val = True

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=800,
                                    act = tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=800,
                                    act = tf.nn.relu, name='relu2')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    network = tl.layers.DenseLayer(network, n_units=10,
                                    act = tl.activation.identity,
                                    name='output_layer')

    y = network.outputs
    y_op = tf.argmax(tf.nn.softmax(y), 1)

    cost = tl.cost.cross_entropy(y, y_)

    train_params = network.all_params
§
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    sess.run(tf.initialize_all_variables())

    network.print_params()
    network.print_layers()
    print(network.all_params)
    print(network.all_layers)
    print(network.all_drop)

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    start_time_begin = time.time()
    for epoch in range(n_epoch):
        start_time = time.time()
        loss_ep = 0; n_step = 0
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train,
                                                    batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )    # enable dropout
            loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
            loss_ep += loss
            n_step += 1
        loss_ep = loss_ep/ n_step


        if is_val:
            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                dp_dict = tl.utils.dict_to_one( network.all_drop ) # disable dropout
                feed_dict = {x: X_train, y_: y_train}
                feed_dict.update(dp_dict)
                print("   train loss: %f" % sess.run(cost, feed_dict=feed_dict))
                dp_dict = tl.utils.dict_to_one( network.all_drop )
                feed_dict = {x: X_val, y_: y_val}
                feed_dict.update(dp_dict)
                print("   val loss: %f" % sess.run(cost, feed_dict=feed_dict))
                print("   val acc: %f" % np.mean(y_val ==
                                        sess.run(y_op, feed_dict=feed_dict)))
        else:
            print("Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep))
    print("Total training time: %f" % (time.time() - start_time_begin))

    print('Evaluation')
    dp_dict = tl.utils.dict_to_one( network.all_drop )
    feed_dict = {x: X_test, y_: y_test}
    feed_dict.update(dp_dict)
    print("   test loss: %f" % sess.run(cost, feed_dict=feed_dict))
    print("   test acc: %f" % np.mean(y_test == sess.run(y_op,
                                                        feed_dict=feed_dict)))

    tl.files.save_npz(network.all_params , name='model.npz')
    sess.close()


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    main()
