#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time



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

About this tutorial
-------------------
Many to one LSTM example, Sequence input and single output (e.g. sentiment
analysis where a given sentence is classified as expressing positive or
negative sentiment). It is the 3rd example in Karpathy blog.
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/

The official PTB example in tensorflow/models/rnn/ptb

More TensorFlow official RNN tutorials can be found here
---------------------------------------------------------
# RNN for PTB : https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html#recurrent-neural-networks
# Seq2seq : https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html#sequence-to-sequence-models
# translation : tensorflow/models/rnn/translate

Code References
---------------
tf.nn.rnn_cell.BasicLSTMCell
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard7/tf.nn.rnn_cell.BasicLSTMCell.md
tf.nn.rnn_cell.BasicRNNCell
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard5/tf.nn.rnn_cell.BasicRNNCell.md
tf.nn.rnn_cell.GRUCell
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard5/tf.nn.rnn_cell.GRUCell.md
"""



def ptb_iterator(raw_data, batch_size, num_steps):
  """Iterates on a list of words. Yields (Returns) the source contexts and
    the target context by the given batch_size and num_steps (sequence_length).

  e.g. x = [0, 1, 2]  y = [1, 2, 3] , when batch_size = 1, num_steps = 3,
  raw_data = [i for i in range(100)]

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

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
  >>> train_data = [i for i in range(1000)]
  >>> for batch in ptb_iterator(train_data, batch_size=10, num_steps=20):
  ...       x, y = batch
  ...       print(len(x), len(x[0]), len(y), len(y[0]))
  ...       print('x:\n', x)
  ...       print('y:\n',y)
  ...       exit()

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



class PTBModel(object):
  """The PTB model.

  is_training : boolen
        True, enable DropoutWrapper.

  """
  def __init__(self, is_training, config):
    # print(config.num_steps)
    # exit()
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # from tensorflow.models.rnn import rnn
    # inputs = [tf.squeeze(input_, [1])
    #           for input_ in tf.split(1, num_steps, inputs)]
    # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, sess, lr_value):
    sess.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1        # fof tf.random_uniform_initializer()
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2          # for MultiRNNCell()
    num_steps = 20          # sequance length ?
    hidden_size = 200       # size of RNNs
    max_epoch = 4           # for lr_decay ?
    max_max_epoch = 13      # n_epoch
    keep_prob = 1.0         # for DropoutWrapper() - output_keep_prob
    lr_decay = 0.5
    batch_size = 20         # minibatche
    vocab_size = 10000      # number of words in vocabulary
    # def __init__(self):
    #     self.learning_rate = 1.0
    #     ...
    # When using self, we can use vars() to scope the attributes


class MediumConfig(object):
    """Medium config."""
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


class LargeConfig(object):
    """Large config."""
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


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(sess, m, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    # print(len(data), m.batch_size, m.num_steps) # 929589 20 20

    # for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
    #                                                     m.num_steps)):
    for step, (x, y) in enumerate(ptb_iterator(data, m.batch_size,
                                                        m.num_steps)):
        cost, state, _ = sess.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("  %.3f perplexity: %.3f speed: %.0f wps" %
                    (step * 1.0 / epoch_size, np.exp(costs / iters),
                     iters * m.batch_size / (time.time() - start_time)))

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
    train_data = train_data[0:int(100000/2)]    # for fast testing
    print('len(train_data) {}'.format(len(train_data))) # 929589
    print('len(valid_data) {}'.format(len(valid_data))) # 73760
    print('len(test_data)  {}'.format(len(test_data)))  # 82430
    print('vocab_size      {}'.format(vocab_size))      # 10000

    config = SmallConfig()
    eval_config = SmallConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1


    # train_data = [i for i in range(1000)]
    # for batch in ptb_iterator(train_data, batch_size=10, num_steps=20):
    #     x, y = batch
    #     print(len(x), len(x[0]), len(y), len(y[0]))
    #     print('x:\n', x)
    #     print('y:\n',y)
    #     exit()


    sess = tf.InteractiveSession()
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                  config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config)
        mtest = PTBModel(is_training=False, config=eval_config)

    sess.run(tf.initialize_all_variables())

    for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        m.assign_lr(sess, config.learning_rate * lr_decay)
        start_time = time.time()
        print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))
        train_perplexity = run_epoch(sess, m, train_data, m.train_op, verbose=True)
        print("  Train Perplexity: %.3f took %fs" % (train_perplexity, time.time() - start_time))
        valid_perplexity = run_epoch(sess, mvalid, valid_data, tf.no_op())
        print("  Valid Perplexity: %.3f" % (valid_perplexity))

    start_time = time.time()
    test_perplexity = run_epoch(sess, mtest, test_data, tf.no_op())
    print("Test Perplexity: %.3f took %fs" % (test_perplexity, time.time() - start_time))


if __name__ == "__main__":
    tf.app.run()
