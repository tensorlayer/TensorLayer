#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
from tensorflow.models.rnn.ptb import reader
import numpy as np
import time

"""Example of Synced sequence input and output.
This is a reimpmentation of the TensorFlow official PTB example in :
tensorflow/models/rnn/ptb

Hao Dong: This tutorial can also be considered as pre-training the word embedding
matrix.

The training data will be generated as follow:\n
where batch_size can be seem as how many concurrent computation.\n
As the this example shows. The first batch learn the sequence info from 0 to 9.\n
The second batch learn the sequence info from 10 to 19.\n
So it ignores the info from 9 to 10 !\n
If only if we set the batch_size = 1, it will consider all info from 0 to 20.\n
However, if your dataset is "long" enough (a text corpus usually has billions words),
the ignored info would not effect the final result.

In PTB tutorial, we setted batch_size = 20, so we cut the dataset into 20 segments.
At the begining of each epoch, we initialize (reset) the 20 RNN states for 20
segments, then go through 20 segments separately.

>>> train_data = [i for i in range(20)]
>>> for batch in ptb_iterator(train_data, batch_size=2, num_steps=3):
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

About RNN
----------
$ Karpathy Blog : http://karpathy.github.io/2015/05/21/rnn-effectiveness/

More TensorFlow official RNN examples can be found here
---------------------------------------------------------
$ RNN for PTB : https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html#recurrent-neural-networks
$ Seq2seq : https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html#sequence-to-sequence-models
$ translation : tensorflow/models/rnn/translate
"""

"""Example / benchmark for building a PTB LSTM model.

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
"""

flags = tf.flags
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
FLAGS = flags.FLAGS

def ptb_iterator(raw_data, batch_size, num_steps):
    """Iterates on a list of words. Yields (Returns) the source contexts and
    the target context by the given batch_size and num_steps (sequence_length).

    e.g. x = [0, 1, 2]  y = [1, 2, 3] , when batch_size = 1, num_steps = 3,
    raw_data = [i for i in range(100)]

    In TensorFlow's tutorial, this generates batch_size pointers into the raw
    PTB data, and allows minibatch iteration along these pointers.

    Parameters
    ----------
    raw_data : a list
            the context in list format; note that context usually be
            represented by splitting by space, and then convert to unique
            word IDs.
    batch_size : int
            the batch size.
    num_steps : int
            the number of unrolls. i.e. sequence_length

    Yields
    ------
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

    Raises
    ------
    ValueError : if batch_size or num_steps are too high.

    Examples
    --------
    >>> train_data = [i for i in range(20)]
    >>> for batch in ptb_iterator(train_data, batch_size=2, num_steps=3):
    >>>     x, y = batch
    >>>     print(x, '\n',y)
    ... [[ 0  1  2] <---x   1st batch
    ...  [10 11 12]]
    ... [[ 1  2  3] <---y
    ...  [11 12 13]]
    ...
    ... [[ 3  4  5]         2nd batch
    ...  [13 14 15]]
    ... [[ 4  5  6]
    ...  [14 15 16]]
    ...
    ... [[ 6  7  8]         3rd batch
    ...  [16 17 18]]
    ... [[ 7  8  9]
    ...  [17 18 19]]

    Code Reference
    --------------
    tensorflow/models/rnn/ptb/reader.py
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)




def run_epoch_original(session, m, data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost
    iters += m.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

def run_epoch(sess, cost, final_state, input_data, targets, initial_state, num_steps, batch_size, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0; iters = 0
    state = initial_state.eval()
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
        cost, state, _ = sess.run([cost, final_state, eval_op],
                                 {input_data: x,
                                  targets: y,
                                  initial_state: state})
        costs += cost; iters += num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                (step * 1.0 / epoch_size, np.exp(costs / iters),
                iters * batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def main(_):
    """
    The core of the model consists of an LSTM cell that processes one word at
    a time and computes probabilities of the possible continuations of the
    sentence. The memory state of the network is initialized with a vector
    of zeros and gets updated after reading each word. Also, for computational
    reasons, we will process data in mini-batches of size batch_size.
    """
    train_data, valid_data, test_data, vocab_size = tl.files.load_ptb_dataset()
    # train_data = train_data[0:int(100000/2)]    # for fast testing
    print('len(train_data) {}'.format(len(train_data))) # 929589 a list of int
    print('len(valid_data) {}'.format(len(valid_data))) # 73760  a list of int
    print('len(test_data)  {}'.format(len(test_data)))  # 82430  a list of int
    print('vocab_size      {}'.format(vocab_size))      # 10000
    # exit()

    sess = tf.InteractiveSession()

    # config = SmallConfig()
    # eval_config = SmallConfig()
    # eval_config.batch_size = 1
    # eval_config.num_steps = 1

    if FLAGS.model == "small":
        init_scale = 0.1
        learning_rate = 1.0
        max_grad_norm = 5
        num_steps = 20
        hidden_size = 200
        max_epoch = 4
        max_max_epoch = 13
        keep_prob = 1.0
        lr_decay = 0.5
        batch_size = 20
        vocab_size = 10000
    elif FLAGS.model == "medium":
        init_scale = 0.05
        learning_rate = 1.0
        max_grad_norm = 5
        num_layers = 2
        num_steps = 35
        hidden_size = 650
        max_epoch = 6
        max_max_epoch = 39
        keep_prob = 0.5
        lr_decay = 0.8
        batch_size = 20
        vocab_size = 10000
    elif FLAGS.model == "large":
        init_scale = 0.04
        learning_rate = 1.0
        max_grad_norm = 10
        num_layers = 2
        num_steps = 35
        hidden_size = 1500
        max_epoch = 14
        max_max_epoch = 55
        keep_prob = 0.35
        lr_decay = 1 / 1.15
        batch_size = 20
        vocab_size = 10000
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

    # For words, one int represents one word, so num_steps == n_features.
    input_data = tf.placeholder(tf.int32, [batch_size, num_steps]) # Training, Validing
    input_data_test = tf.placeholder(tf.int32, [1, 1])     # Testing (Predicting)
    targets = tf.placeholder(tf.int32, [None, num_steps])

    def inference(x, is_training, num_steps, reuse=None):
        """If reuse is True, the inferences use the existing parameters,
        then different inferences share the same parameters.
        """
        print("\nnum_steps : %d, is_training : %s, reuse : %s" % (num_steps, is_training, reuse))
        initializer = tf.random_uniform_initializer(init_scale, init_scale)
        with tf.variable_scope("model", reuse=reuse):#, initializer=initializer):
            tl.layers.set_name_reuse(reuse)
            network = tl.layers.EmbeddingInputlayer(
                            inputs = x,
                            vocabulary_size = vocab_size,
                            embedding_size = hidden_size,
                            E_init=tf.random_uniform_initializer(-init_scale, init_scale),
                            name ='embedding_layer') # (20, 35, 650), Correct
            if is_training:
                network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop1')
            network = tl.layers.RNNLayer(network,
                            cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
                            cell_init_args={'forget_bias': 0.0},# 'state_is_tuple': True},
                            n_hidden=hidden_size,
                            initializer=tf.random_uniform_initializer(-init_scale, init_scale),
                            n_steps=num_steps,
                            return_last=False,
                            name='basic_lstm_layer1')
            lstm1 = network
            if is_training:
                network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop2')
            network = tl.layers.RNNLayer(network,
                            cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
                            cell_init_args={'forget_bias': 0.0}, # 'state_is_tuple': True},
                            n_hidden=hidden_size,
                            initializer=tf.random_uniform_initializer(-init_scale, init_scale),
                            n_steps=num_steps,
                            return_last=False,
                            return_seq_2d=True,
                            name='basic_lstm_layer2')
            lstm2 = network # (20, 35, 650)  Correct
            # if return_seq_2d=False, in the last RNN layer, you can reshape the
            # outputs as follow:
            # network = tl.layers.ReshapeLayer(network, shape=[-1, int(network.outputs._shape[-1])], name='reshape') # shape=(700, 650) Correct
            if is_training:
                network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop3')
            network = tl.layers.DenseLayer(network, n_units=vocab_size,
                            W_init=tf.random_uniform_initializer(-init_scale, init_scale),
                            b_init=tf.random_uniform_initializer(-init_scale, init_scale),
                            act = tl.activation.identity, name='output_layer')
        return network, lstm1, lstm2

    # Inference for Training
    network, lstm1, lstm2 = inference(input_data, is_training=True, num_steps=num_steps, reuse=None)
    # Inference for Validating
    network_val, lstm1_val, lstm2_val = inference(input_data, is_training=False, num_steps=num_steps, reuse=True)
    # Inference for Testing
    network_test, lstm1_test, lstm2_test = inference(input_data_test, is_training=False, num_steps=1, reuse=True)
    sess.run(tf.initialize_all_variables())

    # c, h = lstm1.final_state
    # print(c)    # Tensor(shape=(20, 650), dtype=float32)
    # print(h)    # Tensor(shape=(20, 650), dtype=float32)

    # Cost for Training
    logits = network.outputs
    # print(logits)   # shape=(700, 10000)  Correct, in ptb tutorial, shape=(700, 10000)
    # print(targets)  # shape=(?, 35)      in ptb tutorial, shape=(20, 35)
    # print(tf.reshape(targets, [-1]))    # (?,)
    # exit()

    def loss_fn(logits, targets, batch_size, num_steps):
        loss = tf.nn.seq2seq.sequence_loss_by_example(  # implement softmax inside
            [logits],
            [tf.reshape(targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        cost = tf.reduce_sum(loss) / batch_size
        return cost

    # Cost for Training
    cost = loss_fn(network.outputs, targets, batch_size=batch_size, num_steps=num_steps)
    # Cost for Validating
    cost_val = loss_fn(network_val.outputs, targets, batch_size=batch_size, num_steps=num_steps)
    # Cost for Testing
    cost_test = loss_fn(network_val.outputs, targets, batch_size=1, num_steps=1)

    # Truncated Backpropagation for training
    with tf.variable_scope('learning_rate'):
        lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    sess.run(tf.initialize_all_variables())

    network.print_params()
    network.print_layers()
    tl.layers.print_all_variables()


    print("\nStart learning the language model")
    for i in range(max_max_epoch):
        new_lr_decay = lr_decay ** max(i - max_epoch, 0.0)
        sess.run(tf.assign(lr, learning_rate * new_lr_decay))

        # Training
        print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(lr)))
        epoch_size = ((len(train_data) // batch_size) - 1) // num_steps
        start_time = time.time()
        costs = 0.0; iters = 0
        state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)   # reset the state at the begining of every epoch
        state2 = tl.layers.initialize_rnn_state(lstm2.initial_state)

        for step, (x, y) in enumerate(reader.ptb_iterator(train_data, batch_size,
                                                        num_steps)):
            feed_dict = {input_data: x, targets: y,
                        lstm1.initial_state: state1,
                        lstm2.initial_state: state2,
                        }
            feed_dict.update( network.all_drop )    # For training, enable dropout
            _cost, state1, state2, _ = sess.run([cost, lstm1.final_state, lstm2.final_state,  train_op],
                                            feed_dict=feed_dict) # BUG

            costs += _cost; iters += num_steps

            if step % (epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                    (step * 1.0 / epoch_size, np.exp(costs / iters),
                    iters * batch_size / (time.time() - start_time)))

        train_perplexity = np.exp(costs / iters)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))


        # Validing
        epoch_size = ((len(valid_data) // batch_size) - 1) // num_steps
        start_time = time.time()
        costs = 0.0; iters = 0
        state1 = tl.layers.initialize_rnn_state(lstm1_val.initial_state)   # reset the state at the begining of every epoch
        state2 = tl.layers.initialize_rnn_state(lstm2_val.initial_state)

        for step, (x, y) in enumerate(reader.ptb_iterator(valid_data, batch_size,
                                                        num_steps)):
            feed_dict = {input_data: x, targets: y,
                        lstm1_val.initial_state: state1,
                        lstm2_val.initial_state: state2,
                        }
            # feed_dict.update( network_val.all_drop )    # For training, enable dropout
            _cost, state1, state2, _ = sess.run([cost_val, lstm1_val.final_state, lstm2_val.final_state,  tf.no_op()],
                                            feed_dict=feed_dict) # BUG

            costs += _cost; iters += num_steps
        valid_perplexity = np.exp(costs / iters)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))


    # Testing
    epoch_size = ((len(test_data) // 1) - 1) // 1
    start_time = time.time()
    costs = 0.0; iters = 0
    state1 = tl.layers.initialize_rnn_state(lstm1_test.initial_state)   # reset the state at the begining of every epoch
    state2 = tl.layers.initialize_rnn_state(lstm2_test.initial_state)

    for step, (x, y) in enumerate(reader.ptb_iterator(test_data, batch_size=1,
                                                    num_steps=1)):
        feed_dict = {input_data_test: x, targets: y,
                    lstm1_test.initial_state: state1,
                    lstm2_test.initial_state: state2,
                    }
        # feed_dict.update( network_val.all_drop )    # For training, enable dropout
        _cost, state1, state2, _ = sess.run([cost_test, lstm1_test.final_state, lstm2_test.final_state,  tf.no_op()],
                                        feed_dict=feed_dict) # BUG

        costs += _cost; iters += num_steps
    test_perplexity = np.exp(costs / iters)
    print("Test Perplexity: %.3f" % (test_perplexity))
# def run_epoch(session, m, data, eval_op, verbose=False):
#   """Runs the model on the given data."""
#   epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
#   start_time = time.time()
#   costs = 0.0
#   iters = 0
#   state = m.initial_state.eval()
#   state2 = m.initial_state2.eval()
#   for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
#                                                     m.num_steps)):
#     cost, state, state2, _ = session.run([m.cost, m.final_state, m.final_state2, eval_op],
#                                  {m.input_data: x,
#                                   m.targets: y,
#                                   m.initial_state: state,
#                                   m.initial_state2: state2})
#     costs += cost
#     iters += m.num_steps
#
#     if verbose and step % (epoch_size // 10) == 10:
#       print("%.3f perplexity: %.3f speed: %.0f wps" %
#             (step * 1.0 / epoch_size, np.exp(costs / iters),
#              iters * m.batch_size / (time.time() - start_time)))
#
#   return np.exp(costs / iters)


    ## original
    #     print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
    #     train_perplexity = run_epoch(session, m, train_data, m.train_op,
    #                                verbose=True)
    #     print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
    #     valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
    #     print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    #
    # test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    # print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()

## medium
# Start learning the language model         INCORRECT
# Epoch: 1 Learning rate: 1.000
# 0.008 perplexity: 12055.703 speed: 481 wps
# 0.107 perplexity: 1435.919 speed: 443 wps
# 0.206 perplexity: 1053.871 speed: 453 wps
# 0.306 perplexity: 872.658 speed: 457 wps
# 0.405 perplexity: 773.015 speed: 457 wps
# 0.505 perplexity: 703.303 speed: 458 wps
# 0.604 perplexity: 642.311 speed: 459 wps
