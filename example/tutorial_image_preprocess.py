import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *

"""
Data Augmentation by numpy, scipy, threading and queue.

Alternatively, we can use TFRecord to preprocess data,
see `tutorial_cifar10_tfrecord.py` for more details.
"""

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(
                                    shape=(-1, 32, 32, 3), plotable=False)

def distort_img(x):
    x = tl.prepro.flip_axis(x, axis=1, is_random=True)
    x = tl.prepro.crop(x, wrg=28, hrg=28, is_random=True)
    return x

s = time.time()
results = tl.prepro.threading_data(X_train[0:100], distort_img)
print("took %.3fs" % (time.time()-s))
print(results.shape)

tl.vis.save_images(X_train[0:10], [1, 10], '_original.png')
tl.vis.save_images(results[0:10], [1, 10], '_distorted.png')
