#! /usr/bin/python
# -*- coding: utf-8 -*-

import io
import os
import time

import numpy as np
# import matplotlib
# matplotlib.use('GTK')
import tensorflow as tf
import tensorlayer as tl
from PIL import Image
from tensorlayer.layers import set_keep

"""
You will learn:
1. How to convert CIFAR-10 dataset into TFRecord format file.
2. How to read CIFAR-10 from TFRecord format file.

More:
1. tutorial_tfrecord.py
2. tutoral_cifar10_tfrecord.py


"""

## Download data, and convert to TFRecord format, see ```tutorial_tfrecord.py```
X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(
                                    shape=(-1, 32, 32, 3), plotable=False)

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int64)

print('X_train.shape', X_train.shape)   # (50000, 32, 32, 3)
print('y_train.shape', y_train.shape)   # (50000,)
print('X_test.shape', X_test.shape)     # (10000, 32, 32, 3)
print('y_test.shape', y_test.shape)     # (10000,)
print('X %s   y %s' % (X_test.dtype, y_test.dtype))

cwd = os.getcwd()
writer = tf.python_io.TFRecordWriter("train.cifar10")
for index, img in enumerate(X_train):
    img_raw = img.tobytes()
    ## Visualize a image
    # tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
    label = int(y_train[index])
    # print(label)
    ## Convert the bytes back to image as follow:
        # image = Image.frombytes('RGB', (32, 32), img_raw)
    # image = np.fromstring(img_raw, np.float32)
    # image = image.reshape([32, 32, 3])
    # tl.visualize.frame(np.asarray(image, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
    }))
    writer.write(example.SerializeToString())  # Serialize To String
writer.close()




## Read Data by Queue and Thread =======================================
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    # img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5    # don't need to cast here, as it is float32 already
    label = tf.cast(features['label'], tf.int32)
    return img, label
img, label = read_and_decode("train.cifar10")

## Use shuffle_batch or batch
# see https://www.tensorflow.org/versions/master/api_docs/python/io_ops.html#shuffle_batch
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=4,
                                                capacity=50000,
                                                min_after_dequeue=10000,
                                                num_threads=1)

print("img_batch   : %s" % img_batch._shape)
print("label_batch : %s" % label_batch._shape)
# init = tf.initialize_all_variables()
with tf.Session() as sess:
    # sess.run(init)
    tl.layers.initialize_global_variables(sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(3):  # number of mini-batch (step)
        print("Step %d" % i)
        val, l = sess.run([img_batch, label_batch])
        # exit()
        print(val.shape, l)
        tl.visualize.images2d(val, second=1, saveable=False, name='batch'+str(i), dtype=np.uint8, fig_idx=2020121)
        tl.vis.save_images(val, [2, 2], '_batch_%d.png' % i)

    coord.request_stop()
    coord.join(threads)
    sess.close()
