#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time


"""
References
-----------
tensorflow/models/rnn/translate

Data
----
http://www.statmt.org/wmt10/
"""

def main_test():
    pass


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess = tl.os.set_gpu_fraction(sess, gpu_fraction = 0.9)
    try:
        main_test()
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        tl.os.exit_tf(sess)
