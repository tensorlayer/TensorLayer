#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time

"""
Examples of Tensor indexing, expend_dim, slicing, append, while_loop, condition.

"""


X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int64)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int64)

print('X_train.shape', X_train.shape)   # (50000, 32, 32, 3)
print('y_train.shape', y_train.shape)   # (50000,)
print('X_test.shape', X_test.shape)     # (10000, 32, 32, 3)
print('y_test.shape', y_test.shape)     # (10000,)
print('X %s   y %s' % (X_test.dtype, y_test.dtype))

sess = tf.InteractiveSession()

batch_size = 128

x = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])   # [batch_size, height, width, channels] Tensor("Placeholder:0", shape=(128, 32, 32, 3), dtype=float32)
y_ = tf.placeholder(tf.int64, shape=[batch_size,])              # Tensor("Placeholder_1:0", shape=(128,), dtype=int64)

# x_distorted, y_ = distorted_image(x, y_, batch_size)
# x_distorted, y_ = distorted_image(X_train[0], y_train[0], batch_size)    # Tensor("batch:0", shape=(128, 24, 24, 3), dtype=float32) Tensor("Reshape:0", shape=(128,), dtype=int64)
# x_distorted, y_ = distorted_image(X_train, y_train, batch_size)
# print(x_distorted, y_)


## Tensor Indexing
# i = tf.gather(x, tf.constant(0))
# print(i) # Tensor("Gather:0", shape=(32, 32, 3), dtype=float32)
# g = sess.run(i ,feed_dict={x: X_train[0:batch_size,:,:,:]})
# print(g.shape)          # (32, 32, 3)
# print(g == X_train[])   # True
# exit()

## Tensor Slicing
# t1 = tf.constant(0.2, shape=[2, 6])
# t2 = tf.constant(0.5, shape=[2, 6])
# t = tf.concat(0, [t1, t2])
# t1 = tf.slice(t, [1,0], [-1,-1])    # remove the 1st row
# print('t:\n',t.eval())
# print('t1:\n',t1.eval())
# exit()

## Tensor Append
## example 1
# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
# t1 = [[1, 2, 3, 4, 5, 6]]
# t2 = [[7, 8, 9, 10, 11, 12]]
# t3 = tf.concat(0, [t1, t2])
# print(t3.eval())
# t4 = tf.concat(1, [t1, t2])
# print(t4.eval())
# exit()
## example 2
# t3 = tf.Variable(tf.constant(0, shape=[1, 6]))
# t4 = tf.Variable(tf.constant(1, shape=[1, 6]))
# t5 = tf.concat(0, [t3, t4])
# sess.run(tf.initialize_all_variables())
# print(t5.eval())
# exit()

## Tensor Wile Loop:
#   https://www.tensorflow.org/versions/master/api_docs/python/control_flow_ops.html#while_loop
## example 1
# i = tf.constant(0)
# sess.run(tf.initialize_all_variables())
# c = lambda i: tf.less(i, 10)
# b = lambda i: tf.add(i, 1)
# r = tf.while_loop(c, b, [i])
# print('result',r.eval())    # 10
# exit()
#
## example 2
# def body(x):
#     a = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100)
#     b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
#     c = a + b
#     return tf.nn.relu(x + c)
# def condition(x):
#     return tf.reduce_sum(x) < 100
# x = tf.Variable(tf.constant(0, shape=[2, 2]))
# sess.run(tf.initialize_all_variables())
# with tf.Session():
#     tf.initialize_all_variables().run()
#     result = tf.while_loop(condition, body, [x])
#     print('result',result.eval())
# exit()
## example 3
# t1 = tf.constant(0.2, shape=[3,3,1])
# t2 = tf.constant(0.5, shape=[3,3,1])
# batch_size = 3
# x = tf.Variable(tf.constant(0.1, shape=[3,3,1]), dtype=tf.float32)
# i = tf.Variable(tf.constant(0))
# def fn1(i):
#     return tf.concat(0, [t1, t2]), tf.add(i, 1)
# def fn2(x, i):
#     return tf.concat(0, [x, t1, t2]), tf.add(i, 1)
# # fn1 = lambda x, i: (tf.concat(0, [t1, t2]), tf.add(i, 1))
# # fn2 = lambda x, i: (tf.concat(0, [x, t1, t2]), tf.add(i, 1))
# def body(x, i):
#     # tf.cond(tf.equal(i, batch_size), fn1(i), fn2(x, i))
#     # tf.cond(tf.equal(i, batch_size), fn1(i), fn2(x,i))
#
#     # return tf.cond(tf.equal(i, batch_size), tf.concat(0, [t1, t2]), tf.concat(0, [x, t1, t2])), tf.add(i, 1)
#
#     # if tf.equal(i, batch_size) is not None:
#     #     return tf.concat(0, [t1, t2]), tf.add(i, 1)
#     # else:
#     #     return tf.concat(0, [x, t1, t2]), tf.add(i, 1)
#     return tf.concat(0, [x, t1, t2]), tf.add(i, 1)
# # body = tf.cond(tf.equal(i, batch_size), lambda x, i: (tf.concat(0, [t1, t2]), tf.add(i, 1)), lambda x, i: (tf.concat(0, [x, t1, t2]), tf.add(i, 1)))
# def condition(x, i):
#     return tf.less(i, batch_size)
#
# sess.run(tf.initialize_all_variables())
# result = tf.while_loop(condition, body, (x, i))
# print('result[0]:\n',result[0].eval(), result[0].eval().shape)
# print('result[1]:\n',result[1].eval())
# exit()

## generate distorted images by using TensorFlow functions
# height, width = 24, 24
# distorted_x = tf.Variable(tf.constant(0.1, shape=[1, height, width, 3]))
# # print(vars(tf.random_crop(tf.gather(x, 1), [height, width, 3])))    # TensorShape([Dimension(24), Dimension(24), Dimension(3)])
# # exit()
#
# c = lambda distorted_x, i: tf.less(i, batch_size-1)
# def body(distorted_x, i):
#     image = tf.random_crop(tf.gather(x, i), [height, width, 3])
#     image = tf.expand_dims(image, 0)
#     return tf.concat(0, [distorted_x, image]), tf.add(i, 1)
# i = tf.Variable(tf.constant(0))
#
# result = tf.while_loop(cond=c, body=body, loop_vars=(distorted_x, i))#, parallel_iterations=16)
#
# # print(X_train[0:batch_size,:,:,:].shape)    # (128, 32, 32, 3)
# sess.run(tf.initialize_all_variables())
# feed_dict={x: X_train[0:batch_size,:,:,:]}
# distorted_images, idx = sess.run(result, feed_dict=feed_dict)
# print(distorted_images.shape, idx)  # error:(3072, 24, 3) 127
# # exit()
# # print(result[1].eval())
# # X_train = np.asarray(X_train, dtype=np.uint8)
# tl.visualize.images2d(X_train[0:10,:,:,:], second=10, saveable=False, name='X_train', dtype=np.uint8, fig_idx=20212)
# tl.visualize.images2d(distorted_images[0:10,:,:,:], second=10, saveable=False, name='X_train', dtype=np.uint8, fig_idx=23012)
# exit()
