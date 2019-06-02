#! /usr/bin/python
# -*- coding: utf-8 -*-
r"""Example of Synced sequence input and output.

This is a reimpmentation of the TensorFlow official PTB example in :
tensorflow/models/rnn/ptb

The batch_size can be seem as how many concurrent computations.\n
As the following example shows, the first batch learn the sequence information by using 0 to 9.\n
The second batch learn the sequence information by using 10 to 19.\n
So it ignores the information from 9 to 10 !\n
If only if we set the batch_size = 1, it will consider all information from 0 to 20.\n

The meaning of batch_size here is not the same with the MNIST example. In MNIST example,
batch_size reflects how many examples we consider in each iteration, while in
PTB example, batch_size is how many concurrent processes (segments)
for speed up computation.

Some Information will be ignored if batch_size > 1, however, if your dataset
is "long" enough (a text corpus usually has billions words), the ignored
information would not effect the final result.

In PTB tutorial, we setted batch_size = 20, so we cut the dataset into 20 segments.
At the begining of each epoch, we initialize (reset) the 20 RNN states for 20
segments, then go through 20 segments separately.

The training data will be generated as follow:\n

>>> train_data = [i for i in range(20)]
>>> for batch in tl.iterate.ptb_iterator(train_data, batch_size=2, num_steps=3):
>>>     x, y = batch
>>>     print(x, '\n',y)
... [[ 0  1  2] <---x                       1st subset/ iteration
...  [10 11 12]]
... [[ 1  2  3] <---y
...  [11 12 13]]
...
... [[ 3  4  5]  <--- 1st batch input       2nd subset/ iteration
...  [13 14 15]] <--- 2nd batch input
... [[ 4  5  6]  <--- 1st batch target
...  [14 15 16]] <--- 2nd batch target
...
... [[ 6  7  8]                             3rd subset/ iteration
...  [16 17 18]]
... [[ 7  8  9]
...  [17 18 19]]

Hao Dong: This example can also be considered as pre-training of the word
embedding matrix.

About RNN
----------
$ Karpathy Blog : http://karpathy.github.io/2015/05/21/rnn-effectiveness/

More TensorFlow official RNN examples can be found here
---------------------------------------------------------
$ RNN for PTB : https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html#recurrent-neural-networks
$ Seq2seq : https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html#sequence-to-sequence-models
$ translation : tensorflow/models/rnn/translate

Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz


A) use the zero_state function on the cell object

B) for an rnn, all time steps share weights. We use one matrix to keep all
gate weights. Split by column into 4 parts to get the 4 gate weight matrices.

"""
import argparse
import sys
import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.models import Model

tl.logging.set_verbosity(tl.logging.DEBUG)


def process_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', default='small', choices=['small', 'medium', 'large'],
        help="A type of model. Possible options are: small, medium, large."
    )
    parameters = parser.parse_args(args)
    return parameters


class PTB_Net(Model):

    def __init__(self, vocab_size, hidden_size, init, keep):
        super(PTB_Net, self).__init__()

        self.embedding = tl.layers.Embedding(vocab_size, hidden_size, init)
        self.dropout1 = tl.layers.Dropout(keep=keep)
        self.lstm1 = tl.layers.RNN(
            cell=tf.keras.layers.LSTMCell(hidden_size), return_last_output=False, return_last_state=True,
            return_seq_2d=False, in_channels=hidden_size
        )
        self.dropout2 = tl.layers.Dropout(keep=keep)
        self.lstm2 = tl.layers.RNN(
            cell=tf.keras.layers.LSTMCell(hidden_size), return_last_output=False, return_last_state=True,
            return_seq_2d=True, in_channels=hidden_size
        )
        self.dropout3 = tl.layers.Dropout(keep=keep)
        self.out_dense = tl.layers.Dense(vocab_size, in_channels=hidden_size, W_init=init, b_init=init, act=None)

    def forward(self, inputs, lstm1_initial_state=None, lstm2_initial_state=None):
        inputs = self.embedding(inputs)
        inputs = self.dropout1(inputs)
        lstm1_out, lstm1_state = self.lstm1(inputs, initial_state=lstm1_initial_state)
        inputs = self.dropout2(lstm1_out)
        lstm2_out, lstm2_state = self.lstm2(inputs, initial_state=lstm2_initial_state)
        inputs = self.dropout3(lstm2_out)
        logits = self.out_dense(inputs)
        return logits, lstm1_state, lstm2_state


def main():
    """
    The core of the model consists of an LSTM cell that processes one word at
    a time and computes probabilities of the possible continuations of the
    sentence. The memory state of the network is initialized with a vector
    of zeros and gets updated after reading each word. Also, for computational
    reasons, we will process data in mini-batches of size batch_size.

    """
    param = process_args(sys.argv[1:])

    if param.model == "small":
        init_scale = 0.1
        learning_rate = 1e-3
        max_grad_norm = 5
        num_steps = 20
        hidden_size = 200
        max_epoch = 4
        max_max_epoch = 13
        keep_prob = 1.0
        lr_decay = 0.5
        batch_size = 20
        vocab_size = 10000
    elif param.model == "medium":
        init_scale = 0.05
        learning_rate = 1e-3
        max_grad_norm = 5
        # num_layers = 2
        num_steps = 35
        hidden_size = 650
        max_epoch = 6
        max_max_epoch = 39
        keep_prob = 0.5
        lr_decay = 0.8
        batch_size = 20
        vocab_size = 10000
    elif param.model == "large":
        init_scale = 0.04
        learning_rate = 1e-3
        max_grad_norm = 10
        # num_layers = 2
        num_steps = 35
        hidden_size = 1500
        max_epoch = 14
        max_max_epoch = 55
        keep_prob = 0.35
        lr_decay = 1 / 1.15
        batch_size = 20
        vocab_size = 10000
    else:
        raise ValueError("Invalid model: %s", param.model)

    # Load PTB dataset
    train_data, valid_data, test_data, vocab_size = tl.files.load_ptb_dataset()
    # train_data = train_data[0:int(100000/5)]    # for fast testing
    print('len(train_data) {}'.format(len(train_data)))  # 929589 a list of int
    print('len(valid_data) {}'.format(len(valid_data)))  # 73760  a list of int
    print('len(test_data)  {}'.format(len(test_data)))  # 82430  a list of int
    print('vocab_size      {}'.format(vocab_size))  # 10000

    # One int represents one word, the meaning of batch_size here is not the
    # same with MNIST example, it is the number of concurrent processes for
    # computational reasons.

    init = tf.random_uniform_initializer(-init_scale, init_scale)
    net = PTB_Net(hidden_size=hidden_size, vocab_size=vocab_size, init=init, keep=keep_prob)

    # Truncated Backpropagation for training
    lr = tf.Variable(0.0, trainable=False)
    train_weights = net.weights
    optimizer = tf.optimizers.Adam(lr=lr)

    print(net)

    print("\nStart learning a language model by using PTB dataset")
    for i in range(max_max_epoch):
        # decreases the initial learning rate after several
        # epoachs (defined by ``max_epoch``), by multipling a ``lr_decay``.
        new_lr_decay = lr_decay**max(i - max_epoch, 0.0)
        lr.assign(learning_rate * new_lr_decay)

        # Training
        net.train()
        print("Epoch: %d/%d Learning rate: %.3f" % (i + 1, max_max_epoch, lr.value()))
        epoch_size = ((len(train_data) // batch_size) - 1) // num_steps
        start_time = time.time()
        costs = 0.0
        iters = 0
        # reset all states at the begining of every epoch
        lstm1_state = None
        lstm2_state = None

        for step, (x, y) in enumerate(tl.iterate.ptb_iterator(train_data, batch_size, num_steps)):

            with tf.GradientTape() as tape:
                ## compute outputs
                logits, lstm1_state, lstm2_state = net(
                    x, lstm1_initial_state=lstm1_state, lstm2_initial_state=lstm2_state
                )
                ## compute loss and update model
                cost = tl.cost.cross_entropy(logits, tf.reshape(y, [-1]), name='train_loss')

            grad, _ = tf.clip_by_global_norm(tape.gradient(cost, train_weights), max_grad_norm)
            optimizer.apply_gradients(zip(grad, train_weights))

            costs += cost
            iters += 1

            if step % (epoch_size // 10) == 10:
                print(
                    "%.3f perplexity: %.3f speed: %.0f wps" % (
                        step * 1.0 / epoch_size, np.exp(costs / iters), iters * batch_size * num_steps /
                        (time.time() - start_time)
                    )
                )
        train_perplexity = np.exp(costs / iters)
        print("Epoch: %d/%d Train Perplexity: %.3f" % (i + 1, max_max_epoch, train_perplexity))

        # Validing
        net.eval()
        start_time = time.time()
        costs = 0.0
        iters = 0
        # reset all states at the begining of every epoch
        lstm1_state = None
        lstm2_state = None
        for step, (x, y) in enumerate(tl.iterate.ptb_iterator(valid_data, batch_size, num_steps)):
            ## compute outputs
            logits, lstm1_state, lstm2_state = net(x, lstm1_initial_state=lstm1_state, lstm2_initial_state=lstm2_state)
            ## compute loss and update model
            cost = tl.cost.cross_entropy(logits, tf.reshape(y, [-1]), name='train_loss')
            costs += cost
            iters += 1
        valid_perplexity = np.exp(costs / iters)
        print("Epoch: %d/%d Valid Perplexity: %.3f" % (i + 1, max_max_epoch, valid_perplexity))

    print("Evaluation")
    # Testing
    net.eval()
    # go through the test set step by step, it will take a while.
    start_time = time.time()
    costs = 0.0
    iters = 0
    # reset all states at the begining
    lstm1_state = None
    lstm2_state = None
    for step, (x, y) in enumerate(tl.iterate.ptb_iterator(test_data, batch_size=1, num_steps=1)):
        ## compute outputs
        logits, lstm1_state, lstm2_state = net(x, lstm1_initial_state=lstm1_state, lstm2_initial_state=lstm2_state)
        ## compute loss and update model
        cost = tl.cost.cross_entropy(logits, tf.reshape(y, [-1]), name='train_loss')
        costs += cost
        iters += 1
    test_perplexity = np.exp(costs / iters)
    print("Test Perplexity: %.3f took %.2fs" % (test_perplexity, time.time() - start_time))

    print(
        "More example: Text generation using Trump's speech data: https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_generate_text.py -- def main_lstm_generate_text():"
    )


if __name__ == "__main__":
    main()

# log of SmallConfig
# Start learning a language model by using PTB dataset
# Epoch: 1 Learning rate: 1.000
# 0.004 perplexity: 5512.735 speed: 4555 wps
# 0.104 perplexity: 841.289 speed: 8823 wps
# 0.204 perplexity: 626.273 speed: 9292 wps
# 0.304 perplexity: 505.628 speed: 9472 wps
# 0.404 perplexity: 435.580 speed: 9551 wps
# 0.504 perplexity: 390.108 speed: 9555 wps
# 0.604 perplexity: 351.379 speed: 9546 wps
# 0.703 perplexity: 324.846 speed: 9579 wps
# 0.803 perplexity: 303.824 speed: 9574 wps
# 0.903 perplexity: 284.468 speed: 9551 wps
# Epoch: 1 Train Perplexity: 269.981
# Epoch: 1 Valid Perplexity: 178.561
# Epoch: 2 Learning rate: 1.000
# 0.004 perplexity: 211.632 speed: 7697 wps
# 0.104 perplexity: 151.509 speed: 9488 wps
# 0.204 perplexity: 158.947 speed: 9674 wps
# 0.304 perplexity: 153.963 speed: 9806 wps
# 0.404 perplexity: 150.938 speed: 9817 wps
# 0.504 perplexity: 148.413 speed: 9824 wps
# 0.604 perplexity: 143.763 speed: 9765 wps
# 0.703 perplexity: 141.616 speed: 9731 wps
# 0.803 perplexity: 139.618 speed: 9781 wps
# 0.903 perplexity: 135.880 speed: 9735 wps
# Epoch: 2 Train Perplexity: 133.771
# Epoch: 2 Valid Perplexity: 142.595
# Epoch: 3 Learning rate: 1.000
# 0.004 perplexity: 146.902 speed: 8345 wps
# 0.104 perplexity: 105.647 speed: 9572 wps
# 0.204 perplexity: 114.261 speed: 9585 wps
# 0.304 perplexity: 111.237 speed: 9586 wps
# 0.404 perplexity: 110.181 speed: 9605 wps
# 0.504 perplexity: 109.383 speed: 9601 wps
# 0.604 perplexity: 106.722 speed: 9635 wps
# 0.703 perplexity: 106.075 speed: 9597 wps
# 0.803 perplexity: 105.481 speed: 9624 wps
# 0.903 perplexity: 103.262 speed: 9618 wps
# Epoch: 3 Train Perplexity: 102.272
# Epoch: 3 Valid Perplexity: 131.884
# Epoch: 4 Learning rate: 1.000
# 0.004 perplexity: 118.127 speed: 7867 wps
# 0.104 perplexity: 85.530 speed: 9330 wps
# 0.204 perplexity: 93.559 speed: 9399 wps
# 0.304 perplexity: 91.141 speed: 9386 wps
# 0.404 perplexity: 90.668 speed: 9462 wps
# 0.504 perplexity: 90.366 speed: 9516 wps
# 0.604 perplexity: 88.479 speed: 9477 wps
# 0.703 perplexity: 88.275 speed: 9533 wps
# 0.803 perplexity: 88.091 speed: 9560 wps
# 0.903 perplexity: 86.430 speed: 9516 wps
# Epoch: 4 Train Perplexity: 85.839
# Epoch: 4 Valid Perplexity: 128.408
# Epoch: 5 Learning rate: 1.000
# 0.004 perplexity: 100.077 speed: 7682 wps
# 0.104 perplexity: 73.856 speed: 9197 wps
# 0.204 perplexity: 81.242 speed: 9266 wps
# 0.304 perplexity: 79.315 speed: 9375 wps
# 0.404 perplexity: 79.009 speed: 9439 wps
# 0.504 perplexity: 78.874 speed: 9377 wps
# 0.604 perplexity: 77.430 speed: 9436 wps
# 0.703 perplexity: 77.415 speed: 9417 wps
# 0.803 perplexity: 77.424 speed: 9407 wps
# 0.903 perplexity: 76.083 speed: 9407 wps
# Epoch: 5 Train Perplexity: 75.719
# Epoch: 5 Valid Perplexity: 127.057
# Epoch: 6 Learning rate: 0.500
# 0.004 perplexity: 87.561 speed: 7130 wps
# 0.104 perplexity: 64.202 speed: 9753 wps
# 0.204 perplexity: 69.518 speed: 9537 wps
# 0.304 perplexity: 66.868 speed: 9647 wps
# 0.404 perplexity: 65.766 speed: 9538 wps
# 0.504 perplexity: 64.967 speed: 9537 wps
# 0.604 perplexity: 63.090 speed: 9565 wps
# 0.703 perplexity: 62.415 speed: 9544 wps
# 0.803 perplexity: 61.751 speed: 9504 wps
# 0.903 perplexity: 60.027 speed: 9482 wps
# Epoch: 6 Train Perplexity: 59.127
# Epoch: 6 Valid Perplexity: 120.339
# Epoch: 7 Learning rate: 0.250
# 0.004 perplexity: 72.069 speed: 7683 wps
# 0.104 perplexity: 53.331 speed: 9526 wps
# 0.204 perplexity: 57.897 speed: 9572 wps
# 0.304 perplexity: 55.557 speed: 9491 wps
# 0.404 perplexity: 54.597 speed: 9483 wps
# 0.504 perplexity: 53.817 speed: 9471 wps
# 0.604 perplexity: 52.147 speed: 9511 wps
# 0.703 perplexity: 51.473 speed: 9497 wps
# 0.803 perplexity: 50.788 speed: 9521 wps
# 0.903 perplexity: 49.203 speed: 9515 wps
# Epoch: 7 Train Perplexity: 48.303
# Epoch: 7 Valid Perplexity: 120.782
# Epoch: 8 Learning rate: 0.125
# 0.004 perplexity: 63.503 speed: 8425 wps
# 0.104 perplexity: 47.324 speed: 9433 wps
# 0.204 perplexity: 51.525 speed: 9653 wps
# 0.304 perplexity: 49.405 speed: 9520 wps
# 0.404 perplexity: 48.532 speed: 9487 wps
# 0.504 perplexity: 47.800 speed: 9610 wps
# 0.604 perplexity: 46.282 speed: 9554 wps
# 0.703 perplexity: 45.637 speed: 9536 wps
# 0.803 perplexity: 44.972 speed: 9493 wps
# 0.903 perplexity: 43.506 speed: 9496 wps
# Epoch: 8 Train Perplexity: 42.653
# Epoch: 8 Valid Perplexity: 122.119
# Epoch: 9 Learning rate: 0.062
# 0.004 perplexity: 59.375 speed: 7158 wps
# 0.104 perplexity: 44.223 speed: 9275 wps
# 0.204 perplexity: 48.269 speed: 9459 wps
# 0.304 perplexity: 46.273 speed: 9564 wps
# 0.404 perplexity: 45.450 speed: 9604 wps
# 0.504 perplexity: 44.749 speed: 9604 wps
# 0.604 perplexity: 43.308 speed: 9619 wps
# 0.703 perplexity: 42.685 speed: 9647 wps
# 0.803 perplexity: 42.022 speed: 9673 wps
# 0.903 perplexity: 40.616 speed: 9678 wps
# Epoch: 9 Train Perplexity: 39.792
# Epoch: 9 Valid Perplexity: 123.170
# Epoch: 10 Learning rate: 0.031
# 0.004 perplexity: 57.333 speed: 7183 wps
# 0.104 perplexity: 42.631 speed: 9592 wps
# 0.204 perplexity: 46.580 speed: 9518 wps
# 0.304 perplexity: 44.625 speed: 9569 wps
# 0.404 perplexity: 43.832 speed: 9576 wps
# 0.504 perplexity: 43.153 speed: 9571 wps
# 0.604 perplexity: 41.761 speed: 9557 wps
# 0.703 perplexity: 41.159 speed: 9524 wps
# 0.803 perplexity: 40.494 speed: 9527 wps
# 0.903 perplexity: 39.111 speed: 9558 wps
# Epoch: 10 Train Perplexity: 38.298
# Epoch: 10 Valid Perplexity: 123.658
# Epoch: 11 Learning rate: 0.016
# 0.004 perplexity: 56.238 speed: 7190 wps
# 0.104 perplexity: 41.771 speed: 9171 wps
# 0.204 perplexity: 45.656 speed: 9415 wps
# 0.304 perplexity: 43.719 speed: 9472 wps
# 0.404 perplexity: 42.941 speed: 9483 wps
# 0.504 perplexity: 42.269 speed: 9494 wps
# 0.604 perplexity: 40.903 speed: 9530 wps
# 0.703 perplexity: 40.314 speed: 9545 wps
# 0.803 perplexity: 39.654 speed: 9580 wps
# 0.903 perplexity: 38.287 speed: 9597 wps
# Epoch: 11 Train Perplexity: 37.477
# Epoch: 11 Valid Perplexity: 123.523
# Epoch: 12 Learning rate: 0.008
# 0.004 perplexity: 55.552 speed: 7317 wps
# 0.104 perplexity: 41.267 speed: 9234 wps
# 0.204 perplexity: 45.119 speed: 9461 wps
# 0.304 perplexity: 43.204 speed: 9519 wps
# 0.404 perplexity: 42.441 speed: 9453 wps
# 0.504 perplexity: 41.773 speed: 9536 wps
# 0.604 perplexity: 40.423 speed: 9555 wps
# 0.703 perplexity: 39.836 speed: 9576 wps
# 0.803 perplexity: 39.181 speed: 9579 wps
# 0.903 perplexity: 37.827 speed: 9554 wps
# Epoch: 12 Train Perplexity: 37.020
# Epoch: 12 Valid Perplexity: 123.192
# Epoch: 13 Learning rate: 0.004
# 0.004 perplexity: 55.124 speed: 8234 wps
# 0.104 perplexity: 40.970 speed: 9391 wps
# 0.204 perplexity: 44.804 speed: 9525 wps
# 0.304 perplexity: 42.912 speed: 9512 wps
# 0.404 perplexity: 42.162 speed: 9536 wps
# 0.504 perplexity: 41.500 speed: 9630 wps
# 0.604 perplexity: 40.159 speed: 9591 wps
# 0.703 perplexity: 39.574 speed: 9575 wps
# 0.803 perplexity: 38.921 speed: 9613 wps
# 0.903 perplexity: 37.575 speed: 9629 wps
# Epoch: 13 Train Perplexity: 36.771
# Epoch: 13 Valid Perplexity: 122.917
# Evaluation
# Test Perplexity: 116.723 took 124.06s

# MediumConfig
# Epoch: 1 Learning rate: 1.000
# 0.008 perplexity: 5173.547 speed: 6469 wps
# 0.107 perplexity: 1219.527 speed: 6453 wps
# 0.206 perplexity: 866.163 speed: 6441 wps
# 0.306 perplexity: 695.163 speed: 6428 wps
# 0.405 perplexity: 598.464 speed: 6420 wps
# 0.505 perplexity: 531.875 speed: 6422 wps
# 0.604 perplexity: 477.079 speed: 6425 wps
# 0.704 perplexity: 438.297 speed: 6428 wps
# 0.803 perplexity: 407.928 speed: 6425 wps
# 0.903 perplexity: 381.264 speed: 6429 wps
# Epoch: 1 Train Perplexity: 360.795
# Epoch: 1 Valid Perplexity: 208.854
# ...
# Epoch: 39 Learning rate: 0.001
# 0.008 perplexity: 56.618 speed: 6357 wps
# 0.107 perplexity: 43.375 speed: 6341 wps
# 0.206 perplexity: 47.873 speed: 6336 wps
# 0.306 perplexity: 46.408 speed: 6337 wps
# 0.405 perplexity: 46.327 speed: 6337 wps
# 0.505 perplexity: 46.115 speed: 6335 wps
# 0.604 perplexity: 45.323 speed: 6336 wps
# 0.704 perplexity: 45.286 speed: 6337 wps
# 0.803 perplexity: 45.174 speed: 6336 wps
# 0.903 perplexity: 44.334 speed: 6336 wps
# Epoch: 39 Train Perplexity: 44.021
# Epoch: 39 Valid Perplexity: 87.516
# Evaluation
# Test Perplexity: 83.858 took 167.58s
