#! /usr/bin/python
# -*- coding: utf-8 -*-


import io
import os

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from PIL import Image


"""
You will learn:
1. How to save data into TFRecord format file.
2. How to read data from TFRecord format file by using Queue and Thread.

Reference:
-----------
English : https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
          https://www.tensorflow.org/versions/master/how_tos/reading_data/index.html
          https://www.tensorflow.org/versions/master/api_docs/python/io_ops.html#readers
Chinese : http://blog.csdn.net/u012759136/article/details/52232266
          https://github.com/ycszen/tf_lab/blob/master/reading_data/TensorFlow高效加载数据的方法.md

More
------
1. tutorial_tfrecord2.py
2. tutorial_cifar10_tfrecord.py

"""

## Save data ==================================================================
classes = ['/data/cat', '/data/dog']  # cat is 0, dog is 1
cwd = os.getcwd()
writer = tf.python_io.TFRecordWriter("train.tfrecords")
for index, name in enumerate(classes):
    class_path = cwd + name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.resize((224, 224))
        ## Visualize the image as follow:
        # tl.visualize.frame(I=img, second=5, saveable=False, name='frame', fig_idx=12836)
        ## Converts a image to bytes
        img_raw = img.tobytes()
        ## Convert the bytes back to image as follow:
        # image = Image.frombytes('RGB', (224,224), img_raw)
        # tl.visualize.frame(I=image, second=1, saveable=False, name='frame', fig_idx=1236)
        ## Write the data into TF format
        # image     : Feature + BytesList
        # label     : Feature + Int64List or FloatList
        # sentence  : FeatureList + Int64List , see Google's im2txt example
        example = tf.train.Example(features=tf.train.Features(feature={ # SequenceExample for seuqnce example
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        writer.write(example.SerializeToString())  # Serialize To String
writer.close()


## Load Data Method 1: Simple read ============================================
# read data one by one in order
for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    example = tf.train.Example()    # SequenceExample for seuqnce example
    example.ParseFromString(serialized_example)
    img_raw = example.features.feature['img_raw'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    ## converts a image from bytes
    image = Image.frombytes('RGB', (224, 224), img_raw[0])
    tl.visualize.frame(np.asarray(image), second=0.5, saveable=False, name='frame', fig_idx=1283)
    print(label)


## Read Data Method 2: Queue and Thread =======================================
# use sess.run to get a batch of data
def read_and_decode(filename):
    # generate a queue with a given file name
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)     # return the file and the name of file
    features = tf.parse_single_example(serialized_example,  # see parse_single_sequence_example for sequence example
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

img, label = read_and_decode("train.tfrecords")

## Use shuffle_batch or batch
# see https://www.tensorflow.org/versions/master/api_docs/python/io_ops.html#shuffle_batch
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=4,
                                                capacity=2000,
                                                min_after_dequeue=1000,
                                                num_threads=16
                                                )
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
        print(val.shape, l)
        tl.visualize.images2d(val, second=1, saveable=False, name='batch', dtype=None, fig_idx=2020121)

    coord.request_stop()
    coord.join(threads)
    sess.close()


















#
