#! /usr/bin/python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import tensorlayer as tl

X_train, y_train, X_val, y_val, X_test, y_test = \
                tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

sess = tf.InteractiveSession()

batch_size = 128

x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])  # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.int64, shape=[batch_size])


def model(x, is_train=True, reuse=False):
    with tf.variable_scope("binarynet", reuse=reuse):
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.BinaryConv2d(net, 32, (5, 5), (1, 1), padding='SAME', name='bcnn1')
        # drop
        net = tl.layers.BatchNormLayer(net, is_train=is_train, name='bn1')
        net = tl.layers.SignLayer(net, name='sign2')
        net = tl.layers.BinaryConv2d(net, 64, (5, 5), (1, 1), padding='SAME', name='bcnn2')
        # drop
        net = tl.layers.BatchNormLayer(net, is_train=is_train, name='bn2')
        net = tl.layers.SignLayer(net, name='sign2')
        net = tl.layers.FlattenLayer(net, name='flatten')
        net = tl.layers.DropoutLayer(net, 0.5, True, is_train, name='drop1')
        # net = tl.layers.DenseLayer(net, 256, act=tf.nn.relu, name='dense')
        net = tl.layers.BinaryDenseLayer(net, 256, name='dense')
        net = tl.layers.DropoutLayer(net, 0.5, True, is_train, name='drop2')
        # net = tl.layers.DenseLayer(net, 10, act=tf.identity, name='output')
        net = tl.layers.BinaryDenseLayer(net, 10, name='bout')
        # net = tl.layers.ScaleLayer(net, name='scale')
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
train_params = tl.layers.get_variables_with_name('binarynet', True, True)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

net_train.print_params()
net_train.print_layers()

n_epoch = 200
print_freq = 5

v = tl.layers.get_quantize_sign_params(sess, net_test.all_params)
print(v)

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
