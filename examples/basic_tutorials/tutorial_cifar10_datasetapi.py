#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Using Dataset API and tf.image can obtain the best performance."""

import time
import multiprocessing
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Conv2d, BatchNormLayer, MaxPool2d, FlattenLayer, Dense

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

batch_size = 128
n_epoch = 50000
learning_rate = 0.0001
n_step_epoch = int(len(y_train) / batch_size)
n_step = n_epoch * n_step_epoch
shuffle_buffer_size = 100


def model_batch_norm(x_crop, y_, is_train, reuse):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init2 = tf.constant_initializer(value=0.1)
    with tf.variable_scope("model", reuse=reuse):
        net = Input(name='input')(x_crop)
        net = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='cnn1')(net)
        net = BatchNorm(decay=0.99, act=tf.nn.relu, name='batch1')(net, is_train=is_train)
        net = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(net)

        net = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='cnn2')(net)
        net = BatchNorm(decay=0.99, act=tf.nn.relu, name='batch2')(net, is_train=is_train)
        net = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(net)

        net = Flatten(name='flatten')
        net = Dense(384, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='d1relu')(net)
        net = Dense(192, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='d2relu')(net)
        net = Dense(n_units=10, act=None, W_init=W_init2, name='output')(net)
        y = net.outputs

        ce = tl.cost.cross_entropy(y, y_, name='cost')
        # L2 for the MLP, without this, the accuracy will be reduced by 15%.
        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        cost = ce + L2

        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc


def generator_train():
    inputs = X_train
    targets = y_train
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
        # yield _input.encode('utf-8'), _target.encode('utf-8')
        yield _input, _target


def generator_test():
    inputs = X_test
    targets = y_test
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
        # yield _input.encode('utf-8'), _target.encode('utf-8')
        yield _input, _target


def _map_fn_train(img, target):
    # 1. Randomly crop a [height, width] section of the image.
    img = tf.random_crop(img, [24, 24, 3])
    # 2. Randomly flip the image horizontally.
    img = tf.image.random_flip_left_right(img)
    # 3. Randomly change brightness.
    img = tf.image.random_brightness(img, max_delta=63)
    # 4. Randomly change contrast.
    img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    # 5. Subtract off the mean and divide by the variance of the pixels.
    img = tf.image.per_image_standardization(img)
    target = tf.reshape(target, ())
    return img, target


def _map_fn_test(img, target):
    # 1. Crop the central [height, width] of the image.
    img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
    # 2. Subtract off the mean and divide by the variance of the pixels.
    img = tf.image.per_image_standardization(img)
    img = tf.reshape(img, (24, 24, 3))
    target = tf.reshape(target, ())
    return img, target


# dataset API and augmentation
ds = tf.data.Dataset().from_generator(
    generator_train, output_types=(tf.float32, tf.int32)
)  # , output_shapes=((24, 24, 3), (1)))
ds = ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
ds = ds.repeat(n_epoch)
ds = ds.shuffle(shuffle_buffer_size)
ds = ds.prefetch(buffer_size=4096)
ds = ds.batch(batch_size)
value = ds.make_one_shot_iterator().get_next()

ds = tf.data.Dataset().from_generator(
    generator_test, output_types=(tf.float32, tf.int32)
)  # , output_shapes=((24, 24, 3), (1)))
ds = ds.shuffle(shuffle_buffer_size)
ds = ds.map(_map_fn_test, num_parallel_calls=multiprocessing.cpu_count())
ds = ds.repeat(n_epoch)
ds = ds.prefetch(buffer_size=4096)
ds = ds.batch(batch_size)
value_test = ds.make_one_shot_iterator().get_next()

with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
        net, cost, acc, = model_batch_norm(value[0], value[1], True, False)
        _, cost_test, acc_test = model_batch_norm(value_test[0], value_test[1], False, True)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    sess.run(tf.global_variables_initializer())

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)
    print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))

    step = 0
    epoch = 0
    train_loss, train_acc, n_batch = 0, 0, 0
    start_time = time.time()
    while step < n_step:
        # train one batch
        err, ac, _ = sess.run([cost, acc, train_op])
        step += 1
        train_loss += err
        train_acc += ac
        n_batch += 1
        # one epoch finished, start evaluation
        if (step % n_step_epoch) == 0:
            print(
                "Epoch %d : Step %d-%d of %d took %fs" %
                (epoch, step, step + n_step_epoch, n_step, time.time() - start_time)
            )
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))

            test_loss, test_acc, n_batch = 0, 0, 0
            for _ in range(int(len(y_test) / batch_size)):
                err, ac = sess.run([cost_test, acc_test])
                test_loss += err
                test_acc += ac
                n_batch += 1
            print("   test loss: %f" % (test_loss / n_batch))
            print("   test acc: %f" % (test_acc / n_batch))
            train_loss, train_acc, n_batch = 0, 0, 0
            epoch += 1
            start_time = time.time()
            # save model
            if (step % (n_step_epoch * 50)) == 0:
                tl.files.save_npz(net.all_weights, name='model.npz', sess=sess)
