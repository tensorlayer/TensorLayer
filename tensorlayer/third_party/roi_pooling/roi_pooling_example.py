from __future__ import print_function

import tensorflow as tf
import numpy as np

from roi_pooling.roi_pooling_ops import roi_pooling

# input feature map going into the RoI pooling 
input_value = [[
    [[1], [2], [4], [4]],
    [[3], [4], [1], [2]],
    [[6], [2], [1], [7.0]],
    [[1], [3], [2], [8]]
]]
input_value = np.asarray(input_value, dtype='float32')

# Regions of interest as lists of:
# feature map index, upper left, bottom right coordinates
rois_value = [
    [0, 0, 0, 1, 1],
    [0, 1, 1, 2, 2],
    [0, 2, 2, 3, 3],
    [0, 0, 0, 2, 2],
    [0, 0, 0, 3, 3]
]
rois_value = np.asarray(rois_value, dtype='int32')

# the pool_height and width are parameters of the ROI layer
pool_height, pool_width = (2, 2)
n_rois = len(rois_value)
y_shape = [n_rois, 1, pool_height, pool_width]

print('Input: ', input_value, ', shape: ', input_value.shape)
print('ROIs: ', rois_value, ', shape: ', rois_value.shape)

# precise semantics is now only defined by the kernel, need tests
input = tf.placeholder(tf.float32)
rois = tf.placeholder(tf.int32)

y = roi_pooling(input, rois, pool_height=2, pool_width=2)
mean = tf.reduce_mean(y)

grads = tf.gradients(mean, input)
print(type(grads))
print(len(grads))
print(grads)
print(input_value.shape)

with tf.Session('') as sess:
    input_const = tf.constant(input_value, tf.float32)
    rois_const = tf.constant(rois_value, tf.int32)
    y = roi_pooling(input_const, rois_const, pool_height=2, pool_width=2)
    mean = tf.reduce_mean(y)

    numerical_grad_error_1 = tf.test.compute_gradient_error([input_const], [input_value.shape], y, y_shape)
    numerical_grad_error_2 = tf.test.compute_gradient_error([input_const], [input_value.shape], mean, [])
    print(numerical_grad_error_1, numerical_grad_error_2)

with tf.Session('') as sess:
    y_output = sess.run(y, feed_dict={input: input_value, rois: rois_value})
    print('y: ', y_output)
    grads_output = sess.run(grads, feed_dict={input: input_value, rois: rois_value})
    print('grads: ', grads_output)
