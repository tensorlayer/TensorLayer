#! /usr/bin/python
# -*- coding: utf8 -*-

# tf import data dataset.map https://www.tensorflow.org/programmers_guide/datasets#applying_arbitrary_python_logic_with_tfpy_func
# tf.py_func https://www.tensorflow.org/api_docs/python/tf/py_func
# tl ref: https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_imagenet_inceptionV3_distributed.py
# cn ref: https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/79342369
# cn ref: https://zhuanlan.zhihu.com/p/31466173

import json
import multiprocessing
import random
import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl

# tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

imgs_file_list, _, _, _, classes, _, _, _, objs_info_list, _ = tl.files.load_voc_dataset(dataset="2007")

ann_list = []
for info in objs_info_list:
    ann = tl.prepro.parse_darknet_ann_str_to_list(info)
    c, b = tl.prepro.parse_darknet_ann_list_to_cls_box(ann)
    ann_list.append([c, b])

n_epoch = 10
batch_size = 64
im_size = [416, 416]
jitter = 0.2
shuffle_buffer_size = 100


def generator():
    inputs = imgs_file_list
    targets = objs_info_list

    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")

    for _input, _target in zip(inputs, targets):
        yield _input.encode('utf-8'), _target.encode('utf-8')


def _data_aug_fn(im, ann):
    ## parse annotation
    ann = ann.decode()
    ann = tl.prepro.parse_darknet_ann_str_to_list(ann)
    clas, coords = tl.prepro.parse_darknet_ann_list_to_cls_box(ann)
    ## random brightness, contrast and saturation (tf.image API is faster)
    # im = tl.prepro.brightness(im, gamma=0.5, gain=1, is_random=True)
    # im = tl.prepro.illumination(im, gamma=(0.5, 1.5),
    #          contrast=(0.5, 1.5), saturation=(0.5, 1.5), is_random=True)    # TypeError: Cannot handle this data type
    ## random horizontal flip
    im, coords = tl.prepro.obj_box_left_right_flip(im, coords, is_rescale=True, is_center=True, is_random=True)
    ## random resize and crop
    tmp0 = random.randint(1, int(im_size[0] * jitter))
    tmp1 = random.randint(1, int(im_size[1] * jitter))
    im, coords = tl.prepro.obj_box_imresize(im, coords, [im_size[0] + tmp0, im_size[1] + tmp1], \
        is_rescale=True, interp='bicubic')
    im, clas, coords = tl.prepro.obj_box_crop(im, clas, coords, wrg=im_size[1], hrg=im_size[0], \
        is_rescale=True, is_center=True, is_random=True)
    ## value [0, 255] to [-1, 1] (optional)
    # im = im / 127.5 - 1
    ## value [0, 255] to [0, 1] (optional)
    im = im / 255
    im = np.array(im, dtype=np.float32)  # important
    return im, str([clas, coords]).encode('utf-8')


def _map_fn(filename, annotation):
    ## read image
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    ## data augmentation for image only  0.02s
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # subtract off the mean and divide by the variance of the pixels. (optional)
    # img = tf.image.per_image_standardization(img)
    ## data augmentation for image and bounding box
    image, annotation = tf.numpy_function(_data_aug_fn, [image, annotation], [tf.float32, tf.string])
    return image, annotation


ds = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.string))
ds = ds.shuffle(shuffle_buffer_size)
ds = ds.map(_map_fn, num_parallel_calls=multiprocessing.cpu_count())
ds = ds.repeat(n_epoch)
ds = ds.prefetch(buffer_size=2048)
ds = ds.batch(batch_size)

st = time.time()
im, annbyte = next(iter(ds))
print('took {}s'.format(time.time() - st))

im = im.numpy()

ann = []
for a in annbyte:
    a = a.numpy().decode()
    ann.append(json.loads(a))

## save all images
for i in range(len(im)):
    print(ann[i][1])
    tl.vis.draw_boxes_and_labels_to_image(
        im[i] * 255, ann[i][0], ann[i][1], [], classes, True, save_name='_bbox_vis_%d.png' % i
    )
