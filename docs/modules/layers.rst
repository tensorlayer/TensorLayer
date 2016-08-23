API - Layers
=========================

To make TensorLayer simple, we minimize the number of layer classes as much as
we can. So we encourage you to use TensorFlow's function.
For example, we do not provide layer for local response normalization, we suggest
you to apply ``tf.nn.lrn`` on ``Layer.outputs``.
More functions can be found in `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_


Understand layer
-----------------

All TensorLayer layers have a number of properties in common:

 - ``layer.outputs`` : Tensor, the outputs of current layer.
 - ``layer.all_params`` : a list of Tensor, all network variables in order.
 - ``layer.all_layers`` : a list of Tensor, all network outputs in order.
 - ``layer.all_drop`` : a dictionary of {placeholder : float}, all keeping probabilities of noise layer.

All TensorLayer layers have a number of methods in common:

 - ``layer.print_params()`` : print the network variables information in order (after ``sess.run(tf.initialize_all_variables())``). alternatively, print all variables by ``tl.layers.print_all_variables()``.
 - ``layer.print_layers()`` : print the network layers information in order.
 - ``layer.count_params()`` : print the number of parameters in the network.



The initialization of a network is done by input layer, then we can stacked layers
as follow, then a network is a ``Layer`` class.
The most important properties of a network are ``network.all_params``, ``network.all_layers`` and ``network.all_drop``.
The ``all_params`` is a list which store all pointers of all network parameters in order,
the following script define a 3 layer network, then:

``all_params`` = [W1, b1, W2, b2, W_out, b_out]

The ``all_layers`` is a list which store all pointers of the outputs of all layers,
in the following network:

``all_layers`` = [drop(?,784), relu(?,800), drop(?,800), relu(?,800), drop(?,800)], identity(?,10)]

where ``?`` reflects any batch size. You can print the layer information and parameters information by
using ``network.print_layers()`` and ``network.print_params()``.
To count the number of parameters in a network, run ``network.count_params()``.



.. code-block:: python

  sess = tf.InteractiveSession()

  x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
  y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

  network = tl.layers.InputLayer(x, name='input_layer')
  network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
  network = tl.layers.DenseLayer(network, n_units=800,
                                  act = tf.nn.relu, name='relu1')
  network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
  network = tl.layers.DenseLayer(network, n_units=800,
                                  act = tf.nn.relu, name='relu2')
  network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
  network = tl.layers.DenseLayer(network, n_units=10,
                                  act = tl.activation.identity,
                                  name='output_layer')

  y = network.outputs
  y_op = tf.argmax(tf.nn.softmax(y), 1)

  cost = tl.cost.cross_entropy(y, y_)

  train_params = network.all_params

  train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                              epsilon=1e-08, use_locking=False).minimize(cost, var_list = train_params)

  sess.run(tf.initialize_all_variables())

  network.print_params()
  network.print_layers()

In addition, ``network.all_drop`` is a dictionary which stores the keeping probabilities of all
noise layer. In the above network, they are the keeping probabilities of dropout layers.

So for training, enable all dropout layers as follow.

.. code-block:: python

  feed_dict = {x: X_train_a, y_: y_train_a}
  feed_dict.update( network.all_drop )
  loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
  feed_dict.update( network.all_drop )

For evaluating and testing, disable all dropout layers as follow.

.. code-block:: python

  feed_dict = {x: X_val, y_: y_val}
  feed_dict.update(dp_dict)
  print("   val loss: %f" % sess.run(cost, feed_dict=feed_dict))
  print("   val acc: %f" % np.mean(y_val ==
                          sess.run(y_op, feed_dict=feed_dict)))

For more details, please read the MNIST examples.

Creating custom layers
------------------------

Understand Dense layer
^^^^^^^^^^^^^^^^^^^^^^^^^

Before creating your own TensorLayer layer, let's have a look at Dense layer.
It creates a weights matrix and biases vector if not exists, then implement
the output expression.
At the end, as a layer with parameter, we also need to append the parameters into ``all_params``.


.. code-block:: python

  class DenseLayer(Layer):
      """
      The :class:`DenseLayer` class is a fully connected layer.

      Parameters
      ----------
      layer : a :class:`Layer` instance
          The `Layer` class feeding into this layer.
      n_units : int
          The number of units of the layer.
      act : activation function
          The function that is applied to the layer activations.
      W_init : weights initializer
          The initializer for initializing the weight matrix.
      b_init : biases initializer
          The initializer for initializing the bias vector.
      W_init_args : dictionary
          The arguments for the weights tf.get_variable.
      b_init_args : dictionary
          The arguments for the biases tf.get_variable.
      name : a string or None
          An optional name to attach to this layer.
      """
      def __init__(
          self,
          layer = None,
          n_units = 100,
          act = tf.nn.relu,
          W_init = tf.truncated_normal_initializer(stddev=0.1),
          b_init = tf.constant_initializer(value=0.0),
          W_init_args = {},
          b_init_args = {},
          name ='dense_layer',
      ):
          Layer.__init__(self, name=name)
          self.inputs = layer.outputs
          if self.inputs.get_shape().ndims != 2:
              raise Exception("The input dimension must be rank 2")
          n_in = int(self.inputs._shape[-1])
          self.n_units = n_units
          print("  tensorlayer:Instantiate DenseLayer %s: %d, %s" % (self.name, self.n_units, act))
          with tf.variable_scope(name) as vs:
              W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, **W_init_args )
              b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, **b_init_args )
          self.outputs = act(tf.matmul(self.inputs, W) + b)

          # Hint : list(), dict() is pass by value (shallow).
          self.all_layers = list(layer.all_layers)
          self.all_params = list(layer.all_params)
          self.all_drop = dict(layer.all_drop)
          self.all_layers.extend( [self.outputs] )
          self.all_params.extend( [W, b] )


A simple layer
^^^^^^^^^^^^^^^

To implement a custom layer in TensorLayer, you will have to write a Python class
that subclasses Layer and implement the ``outputs`` expression.

The following is an example implementation of a layer that multiplies its input by 2:

.. code-block:: python

  class DoubleLayer(Layer):
      def __init__(
          self,
          layer = None,
          name ='dense_layer',
      ):
          Layer.__init__(self, name=name)
          self.inputs = layer.outputs
          self.outputs = self.inputs * 2

          self.all_layers = list(layer.all_layers)
          self.all_params = list(layer.all_params)
          self.all_drop = dict(layer.all_drop)
          self.all_layers.extend( [self.outputs] )



.. automodule:: tensorlayer.layers

.. autosummary::

   Layer
   InputLayer
   Word2vecEmbeddingInputlayer
   EmbeddingInputlayer
   DenseLayer
   ReconLayer
   DropoutLayer
   DropconnectDenseLayer
   Conv2dLayer
   PoolLayer
   RNNLayer
   FlattenLayer
   ConcatLayer
   ReshapeLayer
   EmbeddingAttentionSeq2seqWrapper
   flatten_reshape
   clear_layers_name
   set_name_reuse
   print_all_variables
   initialize_rnn_state


Basic layer
-----------

.. autoclass:: Layer

Input layer
------------

.. autoclass:: InputLayer
  :members:

Word Embedding Input layer
-----------------------------

Word2vec layer for training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Word2vecEmbeddingInputlayer

Embedding Input layer
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: EmbeddingInputlayer

Dense layer
------------

Dense layer
^^^^^^^^^^^^^

.. autoclass:: DenseLayer

Reconstruction layer for Autoencoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ReconLayer
   :members:

Noise layer
------------

Dropout layer
^^^^^^^^^^^^^^^^

.. autoclass:: DropoutLayer

Dropconnect + Dense layer
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: DropconnectDenseLayer

Convolutional layer
----------------------

2D Convolutional layer
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Conv2dLayer

Pooling layer
----------------

Max or Mean Pooling layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PoolLayer

Recurrent layer
------------------

Recurrent layer for any cell (LSTM, GRU etc)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RNNLayer

Shape layer
------------

Flatten layer
^^^^^^^^^^^^^^^

.. autoclass:: FlattenLayer

Concat layer
^^^^^^^^^^^^^^

.. autoclass:: ConcatLayer

Reshape layer
^^^^^^^^^^^^^^^

.. autoclass:: ReshapeLayer

Wrapper
---------

Embedding + Attention + Seq2seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: EmbeddingAttentionSeq2seqWrapper
  :members:

Developing or Untested
------------------------

We highly welcome contributions! Every bit helps and will be credited.

3D Convolutional layer
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Conv3dLayer

Maxout layer
^^^^^^^^^^^^^^^^

.. autoclass:: MaxoutLayer

Gaussian Noise layer
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GaussianNoiseLayer

Bidirectional Recurrent layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BidirectionalRNNLayer

Helper functions
------------------------

.. autofunction:: flatten_reshape
.. autofunction:: clear_layers_name
.. autofunction:: set_name_reuse
.. autofunction:: print_all_variables
.. autofunction:: initialize_rnn_state
