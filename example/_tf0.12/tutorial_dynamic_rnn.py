#! /usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import tensorlayer as tl

"""
You will learn:
1. How to use Dynamic RNN with TensorFlow only.
2. For TensorLayer, please see DynamicRNNLayer.

Reference
----------
1. `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`
2. `dynamic_rnn.ipynb <https://github.com/dennybritz/tf-rnn/blob/master/dynamic_rnn.ipynb>`_
"""


## 1. Simple example
# Create input data of 2 example
X = np.random.randn(2, 10, 8)   # [batch_size, n_step, n_features]

# The second example is of length 6, the first example is of length 10
X[1,6:] = 0
X_lengths = [10, 6]

cell =tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)
# cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True) # TF 0.12

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    inputs=X,
    sequence_length=X_lengths,
    dtype=tf.float64,
    )

result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states}, n=1, feed_dict=None)

assert result[0]["outputs"].shape == (2, 10, 64)    # [batch_size, n_step, n_hidden]

# Outputs for the second example past past length 6 should be 0
assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()


## 2. How to control the initial state
batch_size = X.shape[0]
with tf.variable_scope('name') as vs:
    cell =tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)
    # cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True) # TF 0.12
    initial_state = cell.zero_state(batch_size, dtype=tf.float64)#"float")
    outputs, last_states = tf.nn.dynamic_rnn(
                        cell=cell,
                        inputs=X,
                        sequence_length=X_lengths,
                        initial_state=initial_state,
                        dtype=tf.float64,
                        )
    result = tf.contrib.learn.run_n(
        {"outputs": outputs, "last_states": last_states}, n=1, feed_dict=None)

assert result[0]["outputs"].shape == (2, 10, 64)

assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()


## 3. How to automatically get the last outputs by automatically compute X_lengths
def advanced_indexing_op(input, index):
    """ Advanced Indexing for Sequences. """
    batch_size = tf.shape(input)[0]
    max_length = int(input.get_shape()[1])
    dim_size = int(input.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (index - 1)
    flat = tf.reshape(input, [-1, dim_size])
    relevant = tf.gather(flat, index)
    return relevant

def retrieve_seq_length_op(data):
    """ An op to compute the length of a sequence. 0 are masked. """
    with tf.name_scope('GetLength'):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
    return length

sequence_length = retrieve_seq_length_op(
            incoming if isinstance(X, tf.Tensor) else tf.stack(X))#tf.pack(X))

batch_size = X.shape[0]
with tf.variable_scope('name2') as vs: #, initializer=tf.constant_initializer(value=0.1)) as vs:
    cell =tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)
    # cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True) # TF 0.12
    initial_state = cell.zero_state(batch_size, dtype=tf.float64)#"float")
    outputs, last_states = tf.nn.dynamic_rnn(
                        cell=cell,
                        inputs=X,
                        # sequence_length=X_lengths,
                        sequence_length= sequence_length,
                        initial_state=initial_state,
                        dtype=tf.float64,
                        )
    result = tf.contrib.learn.run_n(
        {"outputs": outputs, "last_states": last_states}, n=1, feed_dict=None)
print('all outputs', result[0]["outputs"])
# print(' outputs 2nd', result[0]["outputs"][1,5])
# exit()
# print(sequence_length)
# exit()
# automatically get the last output
outputs = tf.transpose(tf.stack(outputs), [1, 0, 2]) #outputs = tf.transpose(tf.pack(outputs), [1, 0, 2])
last_outputs = advanced_indexing_op(outputs, sequence_length)
last_states = result[0]["last_states"]
sess = tf.Session()
tl.layers.initialize_global_variables(sess)
# print('last outputs',sess.run(last_outputs)) # (2, 64)  # TO DO
# print('last lstm states',last_states, last_states.c.shape, last_states.h.shape)




# 4. How to use DynamicRNNLayer
# data = [[1,2,3,4,5,6,7,8],
#         [4,5,6,7],
#         [2,5,7]]
# # [batch_size, n_step, n_features]
# # x = tf.placeholder(tf.int32, [3, -1, 1])
#
#
# batched_data = tf.train.batch(
#     tensors=[data],
#     batch_size=3,
#     dynamic_pad=True,
#     name="x_batch"
# )
# res = tf.contrib.learn.run_n({"y": batched_data}, n=1, feed_dict=None)
#
# # Print the result
# print("Batch shape: {}".format(res[0]["y"].shape))
# print(res[0]["y"])

























#
