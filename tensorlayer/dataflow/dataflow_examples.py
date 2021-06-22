#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from tensorlayer.dataflow import Dataset
import numpy as np

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


def generator_train():
    inputs = X_train
    targets = y_train
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
        # yield _input.encode('utf-8'), _target.encode('utf-8')
        yield (_input, np.array(_target))


batch_size = 128
shuffle_buffer_size = 128
n_epoch = 10

import tensorflow as tf


def _map_fn_train(img, target):
    # 1. Randomly crop a [height, width] section of the image.
    img = tf.image.random_crop(img, [24, 24, 3])
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


import multiprocessing
train_ds = Dataset.from_generator(
    generator=generator_train, output_types=(tl.float32, tl.int32)
)  # , output_shapes=((24, 24, 3), (1)))

train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())

train_ds = train_ds.repeat(n_epoch)
train_ds = train_ds.shuffle(shuffle_buffer_size)
train_ds = train_ds.prefetch(buffer_size=4096)
train_ds = train_ds.batch(batch_size)

for X_batch, y_batch in train_ds:
    print(X_batch.shape, y_batch.shape)
