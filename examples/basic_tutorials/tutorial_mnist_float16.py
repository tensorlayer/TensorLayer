#! /usr/bin/python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

LayersConfig.tf_dtype = tf.float16  # tf.float32  tf.float16

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

sess = tf.InteractiveSession()

batch_size = 128

x = tf.placeholder(LayersConfig.tf_dtype, shape=[batch_size, 28, 28, 1])
y_ = tf.placeholder(tf.int64, shape=[batch_size])


def model(x, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        n = Input(name='input')(x)
        # cnn
        n = Conv2d(32, (5, 5), (1, 1), padding='SAME', name='cnn1')(n)
        n = BatchNorm(act=tf.nn.relu, decay=0.95, name='bn1')(n, is_train=is_train)
        n = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool1')(n)
        n = Conv2d(64, (5, 5), (1, 1), padding='SAME', name='cnn2')(n)
        n = BatchNorm(act=tf.nn.relu, decay=0.95, name='bn2')(n, is_train=is_train)
        n = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool2')(n)
        # mlp
        n = Flatten(name='flatten')(n)
        n = Dropout(0.5, True, is_train, name='drop1')(n)
        n = Dense(256, act=tf.nn.relu, name='relu1')(n)
        n = Dropout(0.5, True, is_train, name='drop2')(n)
        n = Dense(10, act=None, name='output')(n)
    return n


# define inferences
net_train = model(x, is_train=True, reuse=False)
net_test = model(x, is_train=False, reuse=True)

net_train.print_weights(False)

# cost for training
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')

# cost and accuracy for evalution
y2 = net_test.outputs
cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, LayersConfig.tf_dtype))

# define the optimizer
train_weights = tl.layers.get_variables_with_name('model', train_only=True, printable=False)
# for float16 epsilon=1e-4 see https://stackoverflow.com/questions/42064941/tensorflow-float16-support-is-broken
# for float32 epsilon=1e-08
train_op = tf.train.AdamOptimizer(
    learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-4, use_locking=False
).minimize(
    cost, var_list=train_weights
)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# train the network
n_epoch = 500
print_freq = 1

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
