#! /usr/bin/python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import tensorlayer as tl

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
# X_train, y_train, X_test, y_test = tl.files.load_cropped_svhn(include_extra=False)

sess = tf.InteractiveSession()

batch_size = 128

x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
y_ = tf.placeholder(tf.int64, shape=[batch_size])


def model(x, is_train=True, reuse=False):
    # In BNN, all the layers inputs are binary, with the exception of the first layer.
    # ref: https://github.com/itayhubara/BinaryNet.tf/blob/master/models/BNN_cifar10.py
    with tf.variable_scope("binarynet", reuse=reuse):
        net = tl.layers.Input(name='input')(x)
        net = tl.layers.DorefaConv2d(1, 3, 32, (5, 5), (1, 1), padding='SAME', b_init=None, name='bcnn1')(net)  #pylint: disable=bare-except
        net = tl.layers.MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool1')(net)
        net = tl.layers.BatchNorm(decay=0.95, act=tl.act.htanh, name='bn1')(net, is_train=is_train)

        # net = tl.layers.Sign()(net)
        net = tl.layers.DorefaConv2d(1, 3, 64, (5, 5), (1, 1), padding='SAME', b_init=None, name='bcnn2')(net)  #pylint: disable=bare-except
        net = tl.layers.MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool2')(net)
        net = tl.layers.BatchNorm(decay=0.95, act=tl.act.htanh, name='bn2')(net, is_train=is_train)

        net = tl.layers.Flatten()(net)
        # net = tl.layers.Dropout(0.8, True, name='drop1')(net, is_train=is_train)
        # net = tl.layers.Sign()(net)
        net = tl.layers.DorefaDense(1, 3, 256, b_init=None, name='dense')(net)
        net = tl.layers.BatchNorm(decay=0.95, act=tl.act.htanh, name='bn3')(net, is_train=is_train)

        # net = tl.layers.Dropout(0.8, True, name='drop2')(net, is_train=is_train)
        # net = tl.layers.Sign()(net)
        net = tl.layers.Dense(10, b_init=None, name='bout')(net)
        net = tl.layers.BatchNorm(decay=0.95, name='bno')(net, is_train=is_train)
    return net


# define inferences
net_train = model(x, is_train=True, reuse=False)
net_test = model(x, is_train=False, reuse=True)

# cost for training
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')

# cost and accuracy for evalution
y2 = net_test.outputs
cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# define the optimizer
train_weights = tl.layers.get_variables_with_name('binarynet', True, True)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_weights)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

net_train.print_weights()
net_train.print_layers()

n_epoch = 200
print_freq = 5

# print(sess.run(net_test.all_weights)) # print real values of parameters

for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))
        val_loss, val_acc, n_batch = 0, 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("   val loss: %f" % (val_loss / n_batch))
        print("   val acc: %f" % (val_acc / n_batch))

print('Evaluation')
test_loss, test_acc, n_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
    err, ac = sess.run([cost_test, acc], feed_dict={x: X_test_a, y_: y_test_a})
    test_loss += err
    test_acc += ac
    n_batch += 1
print("   test loss: %f" % (test_loss / n_batch))
print("   test acc: %f" % (test_acc / n_batch))
