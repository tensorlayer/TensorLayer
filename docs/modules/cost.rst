API - Cost
================

To make TensorLayer simple, we minimize the number of cost functions as much as
we can. So we encourage you to use TensorFlow's function.
For example, you can implement L1, L2 and sum regularization by ``tf.nn.l2_loss``,
``tf.contrib.layers.l1_regularizer``, ``tf.contrib.layers.l2_regularizer`` and
``tf.contrib.layers.sum_regularizer``, see `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_.



Your cost function
-----------------------

TensorLayer provides a simple way to create you own cost function. Take a MLP below for example.

.. code-block:: python

  network = InputLayer(x, name='input')
  network = DropoutLayer(network, keep=0.8, name='drop1')
  network = DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu1')
  network = DropoutLayer(network, keep=0.5, name='drop2')
  network = DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu2')
  network = DropoutLayer(network, keep=0.5, name='drop3')
  network = DenseLayer(network, n_units=10, act=tf.identity, name='output')

The network parameters will be ``[W1, b1, W2, b2, W_out, b_out]``,
then you can apply L2 regularization on the weights matrix of first two layer as follow.

.. code-block:: python

  cost = tl.cost.cross_entropy(y, y_)
  cost = cost + tf.contrib.layers.l2_regularizer(0.001)(network.all_params[0]) + tf.contrib.layers.l2_regularizer(0.001)(network.all_params[2])

Besides, TensorLayer provides a easy way to get all variables by a given name, so you can also
apply L2 regularization on some weights as follow.

.. code-block:: python

  l2 = 0
  for w in tl.layers.get_variables_with_name('W_conv2d', train_only=True, printable=False):
      l2 += tf.contrib.layers.l2_regularizer(1e-4)(w)
  cost = tl.cost.cross_entropy(y, y_) + l2



Regularization of Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After initializing the variables, the informations of network parameters can be
observed by using ``network.print_params()``.

.. code-block:: python

  tl.layers.initialize_global_variables(sess)
  network.print_params()

.. code-block:: text

  param 0: (784, 800) (mean: -0.000000, median: 0.000004 std: 0.035524)
  param 1: (800,) (mean: 0.000000, median: 0.000000 std: 0.000000)
  param 2: (800, 800) (mean: 0.000029, median: 0.000031 std: 0.035378)
  param 3: (800,) (mean: 0.000000, median: 0.000000 std: 0.000000)
  param 4: (800, 10) (mean: 0.000673, median: 0.000763 std: 0.049373)
  param 5: (10,) (mean: 0.000000, median: 0.000000 std: 0.000000)
  num of params: 1276810


The output of network is ``network.outputs``, then the cross entropy can be
defined as follow. Besides, to regularize the weights,
the ``network.all_params`` contains all parameters of the network.
In this case, ``network.all_params = [W1, b1, W2, b2, Wout, bout]`` according
to param 0, 1 ... 5 shown by ``network.print_params()``.
Then max-norm regularization on W1 and W2 can be performed as follow.

.. code-block:: python

  y = network.outputs
  # Alternatively, you can use tl.cost.cross_entropy(y, y_) instead.
  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
  cost = cross_entropy
  cost = cost + tl.cost.maxnorm_regularizer(1.0)(network.all_params[0]) +
            tl.cost.maxnorm_regularizer(1.0)(network.all_params[2])

In addition, all TensorFlow's regularizers like
``tf.contrib.layers.l2_regularizer`` can be used with TensorLayer.


Regularization of Activation outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instance method ``network.print_layers()`` prints all outputs of different
layers in order. To achieve regularization on activation output, you can use
``network.all_layers`` which contains all outputs of different layers.
If you want to apply L1 penalty on the activations of first hidden layer,
just simply add ``tf.contrib.layers.l2_regularizer(lambda_l1)(network.all_layers[1])``
to the cost function.

.. code-block:: python

  network.print_layers()

.. code-block:: text

  layer 0: Tensor("dropout/mul_1:0", shape=(?, 784), dtype=float32)
  layer 1: Tensor("Relu:0", shape=(?, 800), dtype=float32)
  layer 2: Tensor("dropout_1/mul_1:0", shape=(?, 800), dtype=float32)
  layer 3: Tensor("Relu_1:0", shape=(?, 800), dtype=float32)
  layer 4: Tensor("dropout_2/mul_1:0", shape=(?, 800), dtype=float32)
  layer 5: Tensor("add_2:0", shape=(?, 10), dtype=float32)




.. automodule:: tensorlayer.cost

.. autosummary::

   cross_entropy
   sigmoid_cross_entropy
   binary_cross_entropy
   mean_squared_error
   normalized_mean_square_error
   absolute_difference_error
   dice_coe
   dice_hard_coe
   iou_coe
   cross_entropy_seq
   cross_entropy_seq_with_mask
   cosine_similarity
   li_regularizer
   lo_regularizer
   maxnorm_regularizer
   maxnorm_o_regularizer
   maxnorm_i_regularizer


Softmax cross entropy
----------------------
.. autofunction:: cross_entropy

Sigmoid cross entropy
----------------------
.. autofunction:: sigmoid_cross_entropy

Binary cross entropy
-------------------------
.. autofunction:: binary_cross_entropy

Mean squared error (L2)
-------------------------
.. autofunction:: mean_squared_error

Normalized mean square error
--------------------------------
.. autofunction:: normalized_mean_square_error

Absolute difference error (L1)
--------------------------------
.. autofunction:: absolute_difference_error

Dice coefficient
-------------------------
.. autofunction:: dice_coe

Hard Dice coefficient
-------------------------
.. autofunction:: dice_hard_coe

IOU coefficient
-------------------------
.. autofunction:: iou_coe

Cross entropy for sequence
-----------------------------
.. autofunction:: cross_entropy_seq

Cross entropy with mask for sequence
----------------------------------------
.. autofunction:: cross_entropy_seq_with_mask

Cosine similarity
-------------------
.. autofunction:: cosine_similarity

Regularization functions
--------------------------

For ``tf.nn.l2_loss``, ``tf.contrib.layers.l1_regularizer``, ``tf.contrib.layers.l2_regularizer`` and
``tf.contrib.layers.sum_regularizer``, see `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_.

Maxnorm
^^^^^^^^^^
.. autofunction:: maxnorm_regularizer

Special
^^^^^^^^^^
.. autofunction:: li_regularizer
.. autofunction:: lo_regularizer
.. autofunction:: maxnorm_o_regularizer
.. autofunction:: maxnorm_i_regularizer
