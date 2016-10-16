#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time
import numbers


def distorted_images(images=None, height=24, width=24):
    """Distort images for generating more training data.

    Features
    ---------
    They are cropped to height * width pixels randomly.

    They are approximately whitened to make the model insensitive to dynamic range.

    Randomly flip the image from left to right.

    Randomly distort the image brightness.

    Randomly distort the image contrast.

    Whiten (Normalize) the images.

    Parameters
    ----------
    images : 4D Tensor
        The tensor or placeholder of images
    height : int
        The height for random crop.
    width : int
        The width for random crop.

    Returns
    -------
    result : tuple of Tensor
        (Tensor for distorted images, Tensor for while loop index)

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
    >>> sess = tf.InteractiveSession()
    >>> batch_size = 128
    >>> x = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
    >>> distorted_images_op = tl.preprocess.distorted_images(images=x, height=24, width=24)
    >>> sess.run(tf.initialize_all_variables())
    >>> feed_dict={x: X_train[0:batch_size,:,:,:]}
    >>> distorted_images, idx = sess.run(distorted_images_op, feed_dict=feed_dict)
    >>> tl.visualize.images2d(X_train[0:9,:,:,:], second=2, saveable=False, name='cifar10', dtype=np.uint8, fig_idx=20212)
    >>> tl.visualize.images2d(distorted_images[1:10,:,:,:], second=10, saveable=False, name='distorted_images', dtype=None, fig_idx=23012)

    Notes
    ------
    - The first image in 'distorted_images' should be removed.

    References
    -----------
    - `tensorflow.models.image.cifar10.cifar10_input <https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/models/image/cifar10/cifar10_input.py>`_
    """
    print(" [Warning] distorted_images will be deprecated due to speed, see TFRecord tutorial for more info...")
    try:
        batch_size = int(images._shape[0])
    except:
        raise Exception('unknow batch_size of images')
    distorted_x = tf.Variable(tf.constant(0.1, shape=[1, height, width, 3]))
    i = tf.Variable(tf.constant(0))

    c = lambda distorted_x, i: tf.less(i, batch_size)

    def body(distorted_x, i):
        # 1. Randomly crop a [height, width] section of the image.
        image = tf.random_crop(tf.gather(images, i), [height, width, 3])
        # 2. Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        # 3. Randomly change brightness.
        image = tf.image.random_brightness(image, max_delta=63)
        # 4. Randomly change contrast.
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        # 5. Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_whitening(image)
        # 6. Append the image to a batch.
        image = tf.expand_dims(image, 0)
        return tf.concat(0, [distorted_x, image]), tf.add(i, 1)

    result = tf.while_loop(cond=c, body=body, loop_vars=(distorted_x, i), parallel_iterations=16)
    return result


def crop_central_whiten_images(images=None, height=24, width=24):
    """Crop the central of image, and normailize it for test data.

    They are cropped to central of height * width pixels.

    Whiten (Normalize) the images.

    Parameters
    ----------
    images : 4D Tensor
        The tensor or placeholder of images
    height : int
        The height for central crop.
    width: int
        The width for central crop.

    Returns
    -------
    result : tuple Tensor
        (Tensor for distorted images, Tensor for while loop index)

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
    >>> sess = tf.InteractiveSession()
    >>> batch_size = 128
    >>> x = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
    >>> central_images_op = tl.preprocess.crop_central_whiten_images(images=x, height=24, width=24)
    >>> sess.run(tf.initialize_all_variables())
    >>> feed_dict={x: X_train[0:batch_size,:,:,:]}
    >>> central_images, idx = sess.run(central_images_op, feed_dict=feed_dict)
    >>> tl.visualize.images2d(X_train[0:9,:,:,:], second=2, saveable=False, name='cifar10', dtype=np.uint8, fig_idx=20212)
    >>> tl.visualize.images2d(central_images[1:10,:,:,:], second=10, saveable=False, name='central_images', dtype=None, fig_idx=23012)

    Notes
    ------
    The first image in 'central_images' should be removed.

    Code References
    ----------------
    - ``tensorflow.models.image.cifar10.cifar10_input``
    """
    print(" [Warning] crop_central_whiten_images will be deprecated due to speed, see TFRecord tutorial for more info...")
    try:
        batch_size = int(images._shape[0])
    except:
        raise Exception('unknow batch_size of images')
    central_x = tf.Variable(tf.constant(0.1, shape=[1, height, width, 3]))
    i = tf.Variable(tf.constant(0))

    c = lambda central_x, i: tf.less(i, batch_size)

    def body(central_x, i):
        # 1. Crop the central [height, width] of the image.
        image = tf.image.resize_image_with_crop_or_pad(tf.gather(images, i), height, width)
        # 2. Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_whitening(image)
        # 5. Append the image to a batch.
        image = tf.expand_dims(image, 0)
        return tf.concat(0, [central_x, image]), tf.add(i, 1)

    result = tf.while_loop(cond=c, body=body, loop_vars=(central_x, i), parallel_iterations=16)
    return result












#
