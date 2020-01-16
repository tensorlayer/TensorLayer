#! /usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
1. Before you start, run this script: https://github.com/tensorlayer/tensorlayer/blob/distributed/scripts/download_and_install_openmpi3_linux.sh
2. Update the PATH with OpenMPI bin by running: PATH=$PATH:$HOME/local/openmpi/bin
   Update the PATH in ~/.bashrc if you want OpenMPI to be ready once the machine start
3. Then  XXXXX   Milo please add this part
    mpirun -np 2 \
        -bind-to none -map-by slot \
        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
        -mca pml ob1 -mca btl ^openib \
        python3 xxxxx.py
"""

import multiprocessing

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import (BatchNormLayer, Conv2d, DenseLayer, FlattenLayer, InputLayer, MaxPool2d)

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


def make_dataset(images, labels, num_epochs=1, shuffle_data_seed=0):
    img = tf.data.Dataset.from_tensor_slices(images)
    lab = tf.data.Dataset.from_tensor_slices(np.array(labels, dtype=np.int64))
    dataset = tf.data.Dataset.zip((img, lab))
    dataset = dataset.repeat(num_epochs).shuffle(buffer_size=10000, seed=shuffle_data_seed)
    return dataset


def data_aug_train(img, ann):
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
    return img, ann


def data_aug_valid(img, ann):
    # 1. Crop the central [height, width] of the image.
    img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
    # 2. Subtract off the mean and divide by the variance of the pixels.
    img = tf.image.per_image_standardization(img)
    return img, ann


def model(x, is_train):
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        net = InputLayer(x, name='input')
        net = Conv2d(net, 64, (5, 5), (1, 1), padding='SAME', b_init=None, name='cnn1')
        net = BatchNormLayer(net, decay=0.99, is_train=is_train, act=tf.nn.relu, name='batch1')
        net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool1')

        net = Conv2d(net, 64, (5, 5), (1, 1), padding='SAME', b_init=None, name='cnn2')
        net = BatchNormLayer(net, decay=0.99, is_train=is_train, act=tf.nn.relu, name='batch2')
        net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool2')

        net = FlattenLayer(net, name='flatten')
        net = DenseLayer(net, 384, act=tf.nn.relu, name='d1relu')
        net = DenseLayer(net, 192, act=tf.nn.relu, name='d2relu')
        net = DenseLayer(net, 10, act=None, name='output')
    return net


def build_train(x, y_):
    net = model(x, is_train=True)
    cost = tl.cost.cross_entropy(net.outputs, y_, name='cost_train')
    L2 = 0
    for p in tl.layers.get_variables_with_name('relu/W', True, True):
        L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
    cost = cost + L2
    accurate_prediction = tf.equal(tf.argmax(net.outputs, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(accurate_prediction, tf.float32), name='accuracy_train')
    log_tensors = {'cost': cost, 'accuracy': accuracy}
    return net, cost, log_tensors


def build_validation(x, y_):
    net = model(x, is_train=False)
    cost = tl.cost.cross_entropy(net.outputs, y_, name='cost_test')
    accurate_prediction = tf.equal(tf.argmax(net.outputs, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(accurate_prediction, tf.float32), name='accuracy_test')
    return net, [cost, accuracy]


if __name__ == '__main__':
    # Load CIFAR10 data
    X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

    # Setup the trainer
    training_dataset = make_dataset(X_train, y_train)
    training_dataset = training_dataset.map(data_aug_train, num_parallel_calls=multiprocessing.cpu_count())
    # validation_dataset = make_dataset(X_test, y_test)
    # validation_dataset = training_dataset.map(data_aug_valid, num_parallel_calls=multiprocessing.cpu_count())
    trainer = tl.distributed.Trainer(
        build_training_func=build_train, training_dataset=training_dataset, optimizer=tf.train.AdamOptimizer,
        optimizer_args={'learning_rate': 0.0001}, batch_size=128, prefetch_size=128
        # validation_dataset=validation_dataset, build_validation_func=build_validation
    )

    # There are multiple ways to use the trainer:
    # 1. Easiest way to train all data: trainer.train_to_end()
    # 2. Train with validation in the middle: trainer.train_and_validate_to_end(validate_step_size=100)
    # 3. Train with full control like follows:
    while not trainer.session.should_stop():
        try:
            # Run a training step synchronously.
            trainer.train_on_batch()
            # TODO: do whatever you like to the training session.
        except tf.errors.OutOfRangeError:
            # The dataset would throw the OutOfRangeError when it reaches the end
            break

    # TODO: Test the trained model
