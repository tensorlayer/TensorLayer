API - Layers
=========================

To make TensorLayer simple, we minimize the number of layer classes as much as
we can. So we encourage you to use TensorFlow's function.
For example, we provide layer for local response normalization, but user can still apply ``tf.nn.lrn`` on ``network.outputs``.
More functions can be found in `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_.


Understand Basic layer
-------------------------

All TensorLayer layers have a number of properties in common:

 - ``layer.outputs`` : a Tensor, the outputs of current layer.
 - ``layer.all_params`` : a list of Tensor, all network variables in order.
 - ``layer.all_layers`` : a list of Tensor, all network outputs in order.
 - ``layer.all_drop`` : a dictionary of {placeholder : float}, all keeping probabilities of noise layer.

All TensorLayer layers have a number of methods in common:

 - ``layer.print_params()`` : print the network variables information in order (after ``tl.layers.initialize_global_variables(sess)``). alternatively, print all variables by ``tl.layers.print_all_variables()``.
 - ``layer.print_layers()`` : print the network layers information in order.
 - ``layer.count_params()`` : print the number of parameters in the network.

The initialization of a network is done by input layer, then we can stacked layers
as follow, a network is a ``Layer`` class.
The most important properties of a network are ``network.all_params``, ``network.all_layers`` and ``network.all_drop``.
The ``all_params`` is a list which store all pointers of all network parameters in order,
the following script define a 3 layer network, then:

``all_params`` = [W1, b1, W2, b2, W_out, b_out]

To get specified variables, you can use ``network.all_params[2:3]`` or ``get_variables_with_name()``.
As the ``all_layers`` is a list which store all pointers of the outputs of all layers,
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

  tl.layers.initialize_global_variables(sess)

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

For more details, please read the MNIST examples on Github.


Customized layer
-----------------

A Simple layer
^^^^^^^^^^^^^^^

To implement a custom layer in TensorLayer, you will have to write a Python class
that subclasses Layer and implement the ``outputs`` expression.

The following is an example implementation of a layer that multiplies its input by 2:

.. code-block:: python

  class DoubleLayer(Layer):
      def __init__(
          self,
          layer = None,
          name ='double_layer',
      ):
          # check layer name (fixed)
          Layer.__init__(self, name=name)

          # the input of this layer is the output of previous layer (fixed)
          self.inputs = layer.outputs

          # operation (customized)
          self.outputs = self.inputs * 2

          # get stuff from previous layer (fixed)
          self.all_layers = list(layer.all_layers)
          self.all_params = list(layer.all_params)
          self.all_drop = dict(layer.all_drop)

          # update layer (customized)
          self.all_layers.extend( [self.outputs] )


Your Dense layer
^^^^^^^^^^^^^^^^^^^

Before creating your own TensorLayer layer, let's have a look at Dense layer.
It creates a weights matrix and biases vector if not exists, then implement
the output expression.
At the end, as a layer with parameter, we also need to append the parameters into ``all_params``.

.. code-block:: python

  class MyDenseLayer(Layer):
    def __init__(
        self,
        layer = None,
        n_units = 100,
        act = tf.nn.relu,
        name ='simple_dense',
    ):
        # check layer name (fixed)
        Layer.__init__(self, name=name)

        # the input of this layer is the output of previous layer (fixed)
        self.inputs = layer.outputs

        # print out info (customized)
        print("  MyDenseLayer %s: %d, %s" % (self.name, n_units, act))

        # operation (customized)
        n_in = int(self.inputs._shape[-1])
        with tf.variable_scope(name) as vs:
            # create new parameters
            W = tf.get_variable(name='W', shape=(n_in, n_units))
            b = tf.get_variable(name='b', shape=(n_units))
            # tensor operation
            self.outputs = act(tf.matmul(self.inputs, W) + b)

        # get stuff from previous layer (fixed)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        # update layer (customized)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )


Modifying Pre-train Behaviour
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Greedy layer-wise pretraining is an important task for deep neural network
initialization, while there are many kinds of pre-training methods according
to different network architectures and applications.

For example, the pre-train process of `Vanilla Sparse Autoencoder <http://deeplearning.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity>`_
can be implemented by using KL divergence (for sigmoid) as the following code,
but for `Deep Rectifier Network <http://www.jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf>`_,
the sparsity can be implemented by using the L1 regularization of activation output.

.. code-block:: python

  # Vanilla Sparse Autoencoder
  beta = 4
  rho = 0.15
  p_hat = tf.reduce_mean(activation_out, reduction_indices = 0)
  KLD = beta * tf.reduce_sum( rho * tf.log(tf.div(rho, p_hat))
          + (1- rho) * tf.log((1- rho)/ (tf.sub(float(1), p_hat))) )


There are many pre-train methods, for this reason, TensorLayer provides a simple way to modify or design your
own pre-train method. For Autoencoder, TensorLayer uses ``ReconLayer.__init__()``
to define the reconstruction layer and cost function, to define your own cost
function, just simply modify the ``self.cost`` in ``ReconLayer.__init__()``.
To creat your own cost expression please read `Tensorflow Math <https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html>`_.
By default, ``ReconLayer`` only updates the weights and biases of previous 1
layer by using ``self.train_params = self.all _params[-4:]``, where the 4
parameters are ``[W_encoder, b_encoder, W_decoder, b_decoder]``, where
``W_encoder, b_encoder`` belong to previous DenseLayer, ``W_decoder, b_decoder``
belong to this ReconLayer.
In addition, if you want to update the parameters of previous 2 layers at the same time, simply modify ``[-4:]`` to ``[-6:]``.


.. code-block:: python

  ReconLayer.__init__(...):
      ...
      self.train_params = self.all_params[-4:]
      ...
  	self.cost = mse + L1_a + L2_w








Layer list
------------

.. automodule:: tensorlayer.layers

.. autosummary::

   get_variables_with_name
   get_layers_with_name
   set_name_reuse
   print_all_variables
   initialize_global_variables

   Layer

   InputLayer
   OneHotInputLayer
   Word2vecEmbeddingInputlayer
   EmbeddingInputlayer
   AverageEmbeddingInputlayer

   DenseLayer
   ReconLayer
   DropoutLayer
   GaussianNoiseLayer
   DropconnectDenseLayer

   Conv1dLayer
   Conv2dLayer
   DeConv2dLayer
   Conv3dLayer
   DeConv3dLayer
   PoolLayer
   PadLayer
   UpSampling2dLayer
   DownSampling2dLayer
   DeformableConv2dLayer
   AtrousConv1dLayer
   AtrousConv2dLayer

   Conv1d
   Conv2d
   DeConv2d

   MaxPool1d
   MeanPool1d
   MaxPool2d
   MeanPool2d
   MaxPool3d
   MeanPool3d

   DepthwiseConv2d

   SubpixelConv1d
   SubpixelConv2d

   SpatialTransformer2dAffineLayer
   transformer
   batch_transformer

   BatchNormLayer
   LocalResponseNormLayer
   InstanceNormLayer
   LayerNormLayer

   ROIPoolingLayer

   TimeDistributedLayer

   RNNLayer
   BiRNNLayer

   ConvRNNCell
   BasicConvLSTMCell
   ConvLSTMLayer

   advanced_indexing_op
   retrieve_seq_length_op
   retrieve_seq_length_op2
   DynamicRNNLayer
   BiDynamicRNNLayer

   Seq2Seq
   PeekySeq2Seq
   AttentionSeq2Seq

   FlattenLayer
   ReshapeLayer
   TransposeLayer

   LambdaLayer

   ConcatLayer
   ElementwiseLayer

   ExpandDimsLayer
   TileLayer

   StackLayer
   UnStackLayer

   EstimatorLayer
   SlimNetsLayer
   KerasLayer

   PReluLayer

   MultiplexerLayer

   EmbeddingAttentionSeq2seqWrapper

   flatten_reshape
   clear_layers_name
   initialize_rnn_state
   list_remove_repeat
   merge_networks


Name Scope and Sharing Parameters
---------------------------------

These functions help you to reuse parameters for different inference (graph), and get a
list of parameters by given name. About TensorFlow parameters sharing click `here <https://www.tensorflow.org/versions/master/how_tos/variable_scope/index.html>`_.

Get variables with name
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: get_variables_with_name

Get layers with name
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: get_layers_with_name

Enable layer name reuse
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: set_name_reuse

Print variables
^^^^^^^^^^^^^^^^^^
.. autofunction:: print_all_variables

Initialize variables
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: initialize_global_variables

Basic layer
-----------
.. autoclass:: Layer


Input layer
------------
.. autoclass:: InputLayer
  :members:

One-hot layer
----------------
.. autoclass:: OneHotInputLayer

Word Embedding Input layer
-----------------------------

Word2vec layer for training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Word2vecEmbeddingInputlayer

Embedding Input layer
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: EmbeddingInputlayer

Average Embedding Input layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AverageEmbeddingInputlayer

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

Gaussian noise layer
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GaussianNoiseLayer

Dropconnect + Dense layer
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DropconnectDenseLayer

Convolutional layer (Pro)
--------------------------

1D Convolution
^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Conv1dLayer

2D Convolution
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Conv2dLayer

2D Deconvolution
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DeConv2dLayer

3D Convolution
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Conv3dLayer

3D Deconvolution
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DeConv3dLayer

2D UpSampling
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UpSampling2dLayer

2D DownSampling
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DownSampling2dLayer

2D Deformable Conv
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DeformableConv2dLayer

1D Atrous convolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: AtrousConv1dLayer

2D Atrous convolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AtrousConv2dLayer


Convolutional layer (Simplified)
-----------------------------------

For users don't familiar with TensorFlow, the following simplified functions may easier for you.
We will provide more simplified functions later, but if you are good at TensorFlow, the professional
APIs may better for you.

1D Convolution
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: Conv1d

2D Convolution
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: Conv2d

2D Deconvolution
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: DeConv2d

1D Max pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: MaxPool1d

1D Mean pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: MeanPool1d

2D Max pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: MaxPool2d

2D Mean pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: MeanPool2d

3D Max pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: MaxPool3d

3D Mean pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: MeanPool3d

2D Depthwise/Separable Conv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DepthwiseConv2d

Super-Resolution layer
------------------------

1D Subpixel Convolution
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: SubpixelConv1d

2D Subpixel Convolution
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: SubpixelConv2d


Spatial Transformer
-----------------------

2D Affine Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpatialTransformer2dAffineLayer

2D Affine Transformation function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: transformer

Batch 2D Affine Transformation function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: batch_transformer


Pooling layer
----------------

Pooling layer for any dimensions and any pooling functions.

.. autoclass:: PoolLayer


Padding
----------------

Padding layer for any modes.

.. autoclass:: PadLayer


Normalization layer
--------------------

For local response normalization as it does not have any weights and arguments,
you can also apply ``tf.nn.lrn`` on ``network.outputs``.

Batch Normalization
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchNormLayer

Local Response Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: LocalResponseNormLayer

Instance Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: InstanceNormLayer

Layer Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: LayerNormLayer

Object Detection
-------------------

ROI layer
^^^^^^^^^^^
.. autoclass:: ROIPoolingLayer


Time distributed layer
------------------------

.. autoclass:: TimeDistributedLayer



Fixed Length Recurrent layer
-------------------------------
All recurrent layers can implement any type of RNN cell by feeding different cell function (LSTM, GRU etc).

RNN layer
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: RNNLayer

Bidirectional layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BiRNNLayer



Recurrent Convolutional layer
-------------------------------

Conv RNN Cell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ConvRNNCell

Basic Conv LSTM Cell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BasicConvLSTMCell

Conv LSTM layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ConvLSTMLayer



Advanced Ops for Dynamic RNN
-------------------------------
These operations usually be used inside Dynamic RNN layer, they can
compute the sequence lengths for different situation and get the last RNN outputs by indexing.

Output indexing
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: advanced_indexing_op

Compute Sequence length 1
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: retrieve_seq_length_op

Compute Sequence length 2
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: retrieve_seq_length_op2


Dynamic RNN layer
----------------------

RNN layer
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DynamicRNNLayer

Bidirectional layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BiDynamicRNNLayer



Sequence to Sequence
----------------------

Simple Seq2Seq
^^^^^^^^^^^^^^^^^
.. autoclass:: Seq2Seq

PeekySeq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: PeekySeq2Seq

AttentionSeq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AttentionSeq2Seq




Shape layer
------------

Flatten layer
^^^^^^^^^^^^^^^
.. autoclass:: FlattenLayer

Reshape layer
^^^^^^^^^^^^^^^
.. autoclass:: ReshapeLayer

Transpose layer
^^^^^^^^^^^^^^^^^
.. autoclass:: TransposeLayer


Lambda layer
---------------

.. autoclass:: LambdaLayer

Merge layer
-------------

Concat layer
^^^^^^^^^^^^^^
.. autoclass:: ConcatLayer


Element-wise layer
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ElementwiseLayer


Extend layer
-------------

Expand dims layer
^^^^^^^^^^^^^^^^^^^
.. autoclass:: ExpandDimsLayer

Tile layer
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: TileLayer


Stack layer
-------------

Stack layer
^^^^^^^^^^^^^^
.. autoclass:: StackLayer

Unstack layer
^^^^^^^^^^^^^^^
.. autofunction:: UnStackLayer


Estimator layer
------------------
.. autoclass:: EstimatorLayer



Connect TF-Slim
------------------

Yes ! TF-Slim models can be connected into TensorLayer, all Google's Pre-trained model can be used easily ,
see `Slim-model <https://github.com/tensorflow/models/tree/master/slim#Install>`_ .

.. autoclass:: SlimNetsLayer

Connect Keras
------------------

Yes ! Keras models can be connected into TensorLayer! see `tutorial_keras.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py>`_ .

.. autoclass:: KerasLayer


Parametric activation layer
---------------------------

.. autoclass:: PReluLayer

Flow control layer
----------------------

.. autoclass:: MultiplexerLayer

Wrapper
---------

Embedding + Attention + Seq2seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: EmbeddingAttentionSeq2seqWrapper
  :members:



Helper functions
------------------------

Flatten tensor
^^^^^^^^^^^^^^^^^
.. autofunction:: flatten_reshape

Permanent clear existing layer names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: clear_layers_name

Initialize RNN state
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: initialize_rnn_state

Remove repeated items in a list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: list_remove_repeat

Merge networks attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: merge_networks
