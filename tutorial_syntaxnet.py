#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time


"""
More TensorFlow official SyntaxNet tutorials can be found here:
# https://www.tensorflow.org/versions/master/tutorials/syntaxnet/index.html#syntaxnet
"""

def main_test_syntaxnet():
    pass


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess = tl.ops.set_gpu_fraction(sess, gpu_fraction = 0.3)
    try:
        main_test_syntaxnet()
        tl.ops.exit_tf(sess)                              # close sess, tensorboard and nvidia-process
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        tl.ops.exit_tf(sess)
