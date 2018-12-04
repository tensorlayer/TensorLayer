API - Layers
============

.. automodule:: tensorlayer.layers


Name Scope and Sharing Parameters
---------------------------------

These functions help you to reuse parameters for different inference (graph), and get a
list of parameters by given name. About TensorFlow parameters sharing click `here <https://www.tensorflow.org/versions/master/how_tos/variable_scope/index.html>`__.

Get variables with name
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: get_variables_with_name

Get layers with name
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: get_layers_with_name

Print variables
^^^^^^^^^^^^^^^^^^
.. autofunction:: print_all_variables


Understanding the Basic Layer
-----------------------------

All TensorLayer layers have a number of properties in common:

 - ``layer.outputs`` : a Tensor, the outputs of current layer.
 - ``layer.all_weights`` : a list of Tensor, all network variables in order.
 - ``layer.all_outputs`` : a list of Tensor, all network outputs in order.
 - ``layer.all_drop`` : a dictionary of {placeholder : float}, all keeping probabilities of noise layers.

All TensorLayer layers have a number of methods in common:

 - ``layer.print_weights()`` : print network variable information in order (after ``sess.run(tf.global_variables_initializer())``). alternatively, print all variables by ``tl.layers.print_all_variables()``.
 - ``layer.print_outputs()`` : print network layer information in order.
 - ``layer.count_all_weights()`` : print the number of parameters in the network.

A network starts with the input layer and is followed by layers stacked in order.
A network is essentially a ``Layer`` class.
The key properties of a network are ``network.all_weights``, ``network.all_outputs`` and ``network.all_drop``.
The ``all_weights`` is a list which store pointers to all network parameters in order. For example,
the following script define a 3 layer network, then:

``all_weights`` = [W1, b1, W2, b2, W_out, b_out]

To get specified variable information, you can use ``network.all_weights[2:3]`` or ``get_variables_with_name()``.
``all_outputs`` is a list which stores the pointers to the outputs of all layers, see the example as follow:

``all_outputs`` = [drop(?,784), relu(?,800), drop(?,800), relu(?,800), drop(?,800)], identity(?,10)]

where ``?`` reflects a given batch size. You can print the layer and parameters information by
using ``network.print_outputs()`` and ``network.print_weights()``.
To count the number of parameters in a network, run ``network.count_params()``.

.. code-block:: python

  sess = tf.InteractiveSession()

  x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
  y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

  network = tl.layers.Input(name='input')(x)
  network = tl.layers.Dropout(keep=0.8, name='drop1')(network)
  network = tl.layers.Dense(n_units=800, act=tf.nn.relu, name='relu1')(network)
  network = tl.layers.Dropout(keep=0.5, name='drop2')(network)
  network = tl.layers.Dense(n_units=800, act=tf.nn.relu, name='relu2')(network)
  network = tl.layers.Dropout(keep=0.5, name='drop3')(network)
  network = tl.layers.Dense(n_units=10, act=None, name='output')(network)


  y = network.outputs
  y_op = tf.argmax(tf.nn.softmax(y), 1)

  cost = tl.cost.cross_entropy(y, y_)

  train_params = network.all_weights

  train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                              epsilon=1e-08, use_locking=False).minimize(cost, var_list = train_params)

  sess.run(tf.global_variables_initializer())

  network.print_weights()
  network.print_outputs()

In addition, ``network.all_drop`` is a dictionary which stores the keeping probabilities of all
noise layers. In the above network, they represent the keeping probabilities of dropout layers.

In case for training, you can enable all dropout layers as follow:

.. code-block:: python

  feed_dict = {x: X_train_a, y_: y_train_a}
  feed_dict.update( network.all_drop )
  loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
  feed_dict.update( network.all_drop )

In case for evaluating and testing, you can disable all dropout layers as follow.

.. code-block:: python

  feed_dict = {x: X_val, y_: y_val}
  feed_dict.update(dp_dict)
  print("   val loss: %f" % sess.run(cost, feed_dict=feed_dict))
  print("   val acc: %f" % np.mean(y_val ==
                          sess.run(y_op, feed_dict=feed_dict)))

For more details, please read the MNIST examples in the example folder.

.. -----------------------------------------------------------
..                        Layer List
.. -----------------------------------------------------------

Layer list
----------

.. autosummary::

   get_variables_with_name
   get_layers_with_name
   print_all_variables

   Layer

   Input
   OneHotInput
   Word2vecEmbeddingInput
   EmbeddingInput
   AverageEmbeddingInput

   Dense
   Dropout
   GaussianNoise
   DropconnectDense

   UpSampling2d
   DownSampling2d
   AtrousDeConv2d

   Conv1d
   Conv2d
   DeConv2d
   DeConv3d
   DepthwiseConv2d
   SeparableConv1d
   SeparableConv2d
   DeformableConv2d
   GroupConv2d

   PadLayer
   PoolLayer
   ZeroPad1d
   ZeroPad2d
   ZeroPad3d
   MaxPool1d
   MeanPool1d
   MaxPool2d
   MeanPool2d
   MaxPool3d
   MeanPool3d
   GlobalMaxPool1d
   GlobalMeanPool1d
   GlobalMaxPool2d
   GlobalMeanPool2d
   GlobalMaxPool3d
   GlobalMeanPool3d

   SubpixelConv1d
   SubpixelConv2d

   SpatialTransformer2dAffine
   transformer
   batch_transformer

   BatchNorm
   LocalResponseNorm
   InstanceNorm
   LayerNorm
   GroupNorm
   SwitchNorm

   ROIPoolingLayer

   TimeDistributed

   RNN
   BiRNN

   ConvRNNCell
   BasicConvLSTMCell
   ConvLSTM

   advanced_indexing_op
   retrieve_seq_length_op
   retrieve_seq_length_op2
   retrieve_seq_length_op3
   target_mask_op
   DynamicRNN
   BiDynamicRNN

   Seq2Seq

   Flatten
   Reshape
   Transpose

   Lambda

   Concat
   Elementwise
   ElementwiseLambda

   ExpandDims
   Tile

   Stack
   UnStack

   SlimNets

   Sign
   Scale
   BinaryDense
   BinaryConv2d
   TernaryDense
   TernaryConv2d
   DorefaDense
   DorefaConv2d
   QuantizedDense
   QuantizedDenseWithBN
   QuantizedConv2d
   QuantizedConv2dWithBN

   PRelu
   PRelu6
   PTRelu6

   flatten_reshape
   clear_layers_name
   initialize_rnn_state
   list_remove_repeat
   merge_networks

.. -----------------------------------------------------------
..                    Customizing Layers
.. -----------------------------------------------------------

Customizing Layers
------------------

A Simple Layer
^^^^^^^^^^^^^^

To implement a custom layer in TensorLayer, you will have to write a Python class
that subclasses Layer and implement the ``outputs`` expression.

The following is an example implementation of a layer that multiplies its input by 2:

.. code-block:: python

  class Double(Layer):
      def __init__(
          self,
          layer = None,
          name ='double_layer',
      ):
          # manage layer (fixed)
          super(Double, self).__init__(prev_layer=prev_layer, name=name)

          # the input of this layer is the output of previous layer (fixed)
          self.inputs = layer.outputs

          # operation (customized)
          self.outputs = self.inputs * 2

          # update layer (customized)


Your Dense Layer
^^^^^^^^^^^^^^^^

Before creating your own TensorLayer layer, let's have a look at the Dense layer.
It creates a weight matrix and a bias vector if not exists, and then implements
the output expression.
At the end, for a layer with parameters, we also append the parameters into ``all_weights``.

.. code-block:: python

  class MyDense(Layer):
    def __init__(
        self,
        layer = None,
        n_units = 100,
        act = tf.nn.relu,
        name ='simple_dense',
    ):
        # manage layer (fixed)
        super(MyDense, self).__init__(prev_layer=prev_layer, act=act, name=name)

        # the input of this layer is the output of previous layer (fixed)
        self.inputs = layer.outputs

        # print out info (customized)
        print("  MyDense %s: %d, %s" % (self.name, n_units, act))

        # operation (customized)
        n_in = int(self.inputs._shape[-1])
        with tf.variable_scope(name) as vs:
            # create new parameters
            W = tf.get_variable(name='W', shape=(n_in, n_units))
            b = tf.get_variable(name='b', shape=(n_units))
            # tensor operation
            self.outputs = self._apply_activation(tf.matmul(self.inputs, W) + b)

        # update layer (customized)



.. -----------------------------------------------------------
..                        Basic Layers
.. -----------------------------------------------------------

Basic Layer
-----------

.. autoclass:: Layer

.. -----------------------------------------------------------
..                        Input Layers
.. -----------------------------------------------------------

Input Layers
---------------

Input Layer
^^^^^^^^^^^^^^^^
.. autoclass:: Input

One-hot Input Layer
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: OneHotInput

Word2Vec Embedding Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Word2vecEmbeddingInput

Embedding Input Layer
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: EmbeddingInput

Average Embedding Input Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AverageEmbeddingInput

.. -----------------------------------------------------------
..                     Activation Layers
.. -----------------------------------------------------------


Activation Layers
---------------------------

PReLU Layer
^^^^^^^^^^^^^^^^^
.. autoclass:: PRelu


PReLU6 Layer
^^^^^^^^^^^^^^^^^^
.. autoclass:: PRelu6


PTReLU6 Layer
^^^^^^^^^^^^^^^^^^^
.. autoclass:: PTRelu6


.. -----------------------------------------------------------
..                  Convolutional Layers
.. -----------------------------------------------------------

Convolutional Layers
---------------------

Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Conv1d
"""""""""""""""""""""
.. autoclass:: Conv1d

Conv2d
"""""""""""""""""""""
.. autoclass:: Conv2d


Deconvolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

DeConv2d
"""""""""""""""""""""
.. autoclass:: DeConv2d

DeConv3d
"""""""""""""""""""""
.. autoclass:: DeConv3d


Atrous De-Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

AtrousDeConv2d
"""""""""""""""""""""
.. autoclass:: AtrousDeConv2d

Deformable Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

DeformableConv2d
"""""""""""""""""""""
.. autoclass:: DeformableConv2d


Depthwise Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

DepthwiseConv2d
"""""""""""""""""""""
.. autoclass:: DepthwiseConv2d


Group Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

GroupConv2d
"""""""""""""""""""""
.. autoclass:: GroupConv2d


Separable Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

SeparableConv1d
"""""""""""""""""""""
.. autoclass:: SeparableConv1d

SeparableConv2d
"""""""""""""""""""""
.. autoclass:: SeparableConv2d


SubPixel Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

SubpixelConv1d
"""""""""""""""""""""
.. autoclass:: SubpixelConv1d

SubpixelConv2d
"""""""""""""""""""""
.. autoclass:: SubpixelConv2d


.. -----------------------------------------------------------
..                        Dense Layers
.. -----------------------------------------------------------

Dense Layers
-------------

Dense Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Dense

Drop Connect Dense Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DropconnectDense


.. -----------------------------------------------------------
..                       Dropout Layer
.. -----------------------------------------------------------

Dropout Layers
-------------------
.. autoclass:: Dropout

.. -----------------------------------------------------------
..                        Extend Layers
.. -----------------------------------------------------------

Extend Layers
-------------------

Expand Dims Layer
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ExpandDims


Tile layer
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Tile


.. -----------------------------------------------------------
..                 External Libraries Layers
.. -----------------------------------------------------------

External Libraries Layers
------------------------------

TF-Slim Layer
^^^^^^^^^^^^^^^^^^^
TF-Slim models can be connected into TensorLayer. All Google's Pre-trained model can be used easily ,
see `Slim-model <https://github.com/tensorflow/models/tree/master/research/slim>`__.

.. autoclass:: SlimNets




.. -----------------------------------------------------------
..                  Image Resampling Layers
.. -----------------------------------------------------------

Image Resampling Layers
-------------------------

2D UpSampling
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UpSampling2d

2D DownSampling
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DownSampling2d

.. -----------------------------------------------------------
..                      Lambda Layer
.. -----------------------------------------------------------

Lambda Layers
---------------

Lambda Layer
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Lambda

ElementWise Lambda Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ElementwiseLambda

.. -----------------------------------------------------------
..                      Merge Layer
.. -----------------------------------------------------------

Merge Layers
---------------

Concat Layer
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Concat

ElementWise Layer
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Elementwise

.. -----------------------------------------------------------
..                      Noise Layers
.. -----------------------------------------------------------

Noise Layer
---------------
.. autoclass:: GaussianNoise

.. -----------------------------------------------------------
..                  Normalization Layers
.. -----------------------------------------------------------

Normalization Layers
--------------------

For local response normalization as it does not have any weights and arguments,
you can also apply ``tf.nn.lrn`` on ``network.outputs``.

Batch Normalization
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BatchNorm

Local Response Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: LocalResponseNorm

Instance Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: InstanceNorm

Layer Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: LayerNorm

Group Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GroupNorm

Switch Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SwitchNorm

.. -----------------------------------------------------------
..                Object Detection Layers
.. -----------------------------------------------------------

Object Detection Layer
------------------------
.. autoclass:: ROIPoolingLayer

.. -----------------------------------------------------------
..                     Padding Layers
.. -----------------------------------------------------------

Padding Layers
------------------------

Pad Layer (Expert API)
^^^^^^^^^^^^^^^^^^^^^^^^^
Padding layer for any modes.

.. autoclass:: PadLayer

1D Zero padding
^^^^^^^^^^^^^^^^^^^
.. autoclass:: ZeroPad1d

2D Zero padding
^^^^^^^^^^^^^^^^^^^
.. autoclass:: ZeroPad2d

3D Zero padding
^^^^^^^^^^^^^^^^^^^
.. autoclass:: ZeroPad3d

.. -----------------------------------------------------------
..                     Pooling Layers
.. -----------------------------------------------------------

Pooling Layers
------------------------

Pool Layer (Expert API)
^^^^^^^^^^^^^^^^^^^^^^^^^
Pooling layer for any dimensions and any pooling functions.

.. autoclass:: PoolLayer

1D Max pooling
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MaxPool1d

1D Mean pooling
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MeanPool1d

2D Max pooling
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MaxPool2d

2D Mean pooling
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MeanPool2d

3D Max pooling
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MaxPool3d

3D Mean pooling
^^^^^^^^^^^^^^^^^^^
.. autoclass:: MeanPool3d

1D Global Max pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMaxPool1d

1D Global Mean pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMeanPool1d

2D Global Max pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMaxPool2d

2D Global Mean pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMeanPool2d

3D Global Max pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMaxPool3d

3D Global Mean pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: GlobalMeanPool3d

.. -----------------------------------------------------------
..                    Quantized Layers
.. -----------------------------------------------------------

Quantized Nets
------------------

This is an experimental API package for building Quantized Neural Networks. We are using matrix multiplication rather than add-minus and bit-count operation at the moment. Therefore, these APIs would not speed up the inferencing, for production, you can train model via TensorLayer and deploy the model into other customized C/C++ implementation (We probably provide users an extra C/C++ binary net framework that can load model from TensorLayer).

Note that, these experimental APIs can be changed in the future


Sign
^^^^^^^^^^^^^^
.. autoclass:: Sign

Scale
^^^^^^^^^^^^^^
.. autoclass:: Scale

Binary Dense Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: BinaryDense

Binary (De)Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

BinaryConv2d
"""""""""""""""""""""
.. autoclass:: BinaryConv2d

Ternary Dense Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^

TernaryDense
"""""""""""""""""""""
.. autoclass:: TernaryDense

Ternary Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

TernaryConv2d
"""""""""""""""""""""
.. autoclass:: TernaryConv2d

DoReFa Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

DorefaConv2d
"""""""""""""""""""""
.. autoclass:: DorefaConv2d

DoReFa Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

DorefaConv2d
"""""""""""""""""""""
.. autoclass:: DorefaConv2d

Quantization Dense Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^

QuantizedDense
"""""""""""""""""""""
.. autoclass:: QuantizedDense

QuantizedDenseWithBN
""""""""""""""""""""""""""""""""""""
.. autoclass:: QuantizedDenseWithBN

Quantization Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantization
"""""""""""""""""""""
.. autoclass:: QuantizedConv2d

QuantizedConv2dWithBN
"""""""""""""""""""""
.. autoclass:: QuantizedConv2dWithBN


.. -----------------------------------------------------------
..                  Recurrent Layers
.. -----------------------------------------------------------

Recurrent Layers
---------------------

Fixed Length Recurrent layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
All recurrent layers can implement any type of RNN cell by feeding different cell function (LSTM, GRU etc).

RNN layer
""""""""""""""""""""""""""
.. autoclass:: RNN

Bidirectional layer
"""""""""""""""""""""""""""""""""
.. autoclass:: BiRNN


Recurrent Convolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conv RNN Cell
"""""""""""""""""""""""""""""""""
.. autoclass:: ConvRNNCell

Basic Conv LSTM Cell
"""""""""""""""""""""""""""""""""
.. autoclass:: BasicConvLSTMCell

Conv LSTM layer
"""""""""""""""""""""""""""""""""
.. autoclass:: ConvLSTM


Advanced Ops for Dynamic RNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These operations usually be used inside Dynamic RNN layer, they can
compute the sequence lengths for different situation and get the last RNN outputs by indexing.

Output indexing
"""""""""""""""""""""""""
.. autofunction:: advanced_indexing_op

Compute Sequence length 1
""""""""""""""""""""""""""
.. autofunction:: retrieve_seq_length_op

Compute Sequence length 2
""""""""""""""""""""""""""
.. autofunction:: retrieve_seq_length_op2

Compute Sequence length 3
""""""""""""""""""""""""""
.. autofunction:: retrieve_seq_length_op3

Get Mask
""""""""""""""""""""""""""
.. autofunction:: target_mask_op


Dynamic RNN Layer
^^^^^^^^^^^^^^^^^^^^^^

RNN Layer
""""""""""""""""""""""""""
.. autoclass:: DynamicRNN

Bidirectional Layer
"""""""""""""""""""""""""""""""""
.. autoclass:: BiDynamicRNN


Sequence to Sequence
^^^^^^^^^^^^^^^^^^^^^^

Simple Seq2Seq
"""""""""""""""""
.. autoclass:: Seq2Seq


.. -----------------------------------------------------------
..                      Shape Layers
.. -----------------------------------------------------------

Shape Layers
------------

Flatten Layer
^^^^^^^^^^^^^^^
.. autoclass:: Flatten

Reshape Layer
^^^^^^^^^^^^^^^
.. autoclass:: Reshape

Transpose Layer
^^^^^^^^^^^^^^^^^
.. autoclass:: Transpose

.. -----------------------------------------------------------
..               Spatial Transformer Layers
.. -----------------------------------------------------------

Spatial Transformer
-----------------------

2D Affine Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: SpatialTransformer2dAffine

2D Affine Transformation function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: transformer

Batch 2D Affine Transformation function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: batch_transformer

.. -----------------------------------------------------------
..                      Stack Layers
.. -----------------------------------------------------------

Stack Layer
-------------

Stack Layer
^^^^^^^^^^^^^^
.. autoclass:: Stack

Unstack Layer
^^^^^^^^^^^^^^^
.. autoclass:: UnStack

.. -----------------------------------------------------------
..                 Time Distributed Layer
.. -----------------------------------------------------------

Time Distributed Layer
------------------------
.. autoclass:: TimeDistributed


.. -----------------------------------------------------------
..                      Helper Functions
.. -----------------------------------------------------------

Helper Functions
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
