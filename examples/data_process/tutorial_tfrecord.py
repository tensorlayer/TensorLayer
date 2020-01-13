#! /usr/bin/python
# -*- coding: utf-8 -*-
"""You will learn.

1. How to save data into TFRecord format file.
2. How to read data from TFRecord format file.

Reference:
-----------
English : https://www.tensorflow.org/alpha/tutorials/load_data/images#build_a_tfdatadataset
          https://www.tensorflow.org/alpha/tutorials/load_data/tf_records#tfrecord_files_using_tfdata
Chinese : http://blog.csdn.net/u012759136/article/details/52232266
          https://github.com/ycszen/tf_lab/blob/master/reading_data/TensorFlow高效加载数据的方法.md

More
------
1. tutorial_tfrecord2.py
2. tutorial_cifar10_tfrecord.py

"""

import os

import numpy as np
import tensorflow as tf
from PIL import Image

import tensorlayer as tl

## Save data ==================================================================
# see https://www.tensorflow.org/alpha/tutorials/load_data/tf_records#writing_a_tfrecord_file
classes = ['/data/cat', '/data/dog']  # cat is 0, dog is 1
cwd = os.getcwd()
writer = tf.io.TFRecordWriter("train.tfrecords")
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
# see https://www.tensorflow.org/alpha/tutorials/load_data/tf_records#reading_a_tfrecord_file_2
# read data one by one in order
raw_dataset = tf.data.TFRecordDataset("train.tfrecords")
for serialized_example in raw_dataset:
    example = tf.train.Example()  # SequenceExample for seuqnce example
    example.ParseFromString(serialized_example.numpy())
    img_raw = example.features.feature['img_raw'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    ## converts a image from bytes
    image = Image.frombytes('RGB', (224, 224), img_raw[0])
    # tl.visualize.frame(np.asarray(image), second=0.5, saveable=False, name='frame', fig_idx=1283)
    print(label)


## Read Data Method 2: using tf.data =======================================
# see https://www.tensorflow.org/alpha/tutorials/load_data/tf_records#reading_a_tfrecord_file
# use shuffle and batch
def read_and_decode(filename):
    # generate a queue with a given file name
    raw_dataset = tf.data.TFRecordDataset([filename]).shuffle(1000).batch(4)
    for serialized_example in raw_dataset:
        features = tf.io.parse_example(
            serialized_example, features={
                'label': tf.io.FixedLenFeature([], tf.int64),
                'img_raw': tf.io.FixedLenFeature([], tf.string),
            }
        )
        # You can do more image distortion here for training data
        img_batch = tf.io.decode_raw(features['img_raw'], tf.uint8)
        img_batch = tf.reshape(img_batch, [4, 224, 224, 3])
        # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        label_batch = tf.cast(features['label'], tf.int32)
        yield img_batch, label_batch


img_batch, label_batch = next(read_and_decode("train.tfrecords"))
print("img_batch   : %s" % img_batch.shape)
print("label_batch : %s" % label_batch.shape)
tl.visualize.images2d(img_batch, second=1, saveable=False, name='batch', dtype=None, fig_idx=2020121)
