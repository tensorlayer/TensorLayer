API - Layers
============

.. automodule:: tensorlayer.layers

.. -----------------------------------------------------------
..                        Layer List
.. -----------------------------------------------------------

Layer list
----------

.. autosummary::

   Layer

   Input

   OneHot
   Word2vecEmbedding
   Embedding
   AverageEmbedding

   Dense
   Dropout
   GaussianNoise
   DropconnectDense

   UpSampling2d
   DownSampling2d

   Conv1d
   Conv2d
   Conv3d
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
   CornerPool2d

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

   RNN
   BiRNN

   ConvRNNCell
   BasicConvLSTMCell
   ConvLSTM

   retrieve_seq_length_op
   retrieve_seq_length_op2
   retrieve_seq_length_op3
   target_mask_op

   Seq2Seq

   Flatten
   Reshape
   Transpose
   Shuffle

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
   initialize_rnn_state
   list_remove_repeat

.. -----------------------------------------------------------
..                        Basic Layers
.. -----------------------------------------------------------

Base Layer
-----------

.. autoclass:: Layer

.. -----------------------------------------------------------
..                        Input Layer
.. -----------------------------------------------------------

Input Layers
---------------

Input Layer
^^^^^^^^^^^^^^^^
.. autofunction:: Input

.. -----------------------------------------------------------
..                        Embedding Layers
.. -----------------------------------------------------------


One-hot Layer
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: OneHot

Word2Vec Embedding Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Word2vecEmbedding

Embedding Layer
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Embedding

Average Embedding Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: AverageEmbedding

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

Conv3d
"""""""""""""""""""""
.. autoclass:: Conv3d

Deconvolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^

DeConv2d
"""""""""""""""""""""
.. autoclass:: DeConv2d

DeConv3d
"""""""""""""""""""""
.. autoclass:: DeConv3d


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

2D Corner pooling
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: CornerPool2d

.. -----------------------------------------------------------
..                    Quantized Layers
.. -----------------------------------------------------------

Quantized Nets
------------------

This is an experimental API package for building Quantized Neural Networks. We are using matrix multiplication rather than add-minus and bit-count operation at the moment. Therefore, these APIs would not speed up the inferencing, for production, you can train model via TensorLayer and deploy the model into other customized C/C++ implementation (We probably provide users an extra C/C++ binary net framework that can load model from TensorLayer).

Note that, these experimental APIs can be changed in the future.


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

Common Recurrent layer
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

Simple Seq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Seq2Seq

Advanced Ops for Dynamic RNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These operations usually be used inside Dynamic RNN layer, they can
compute the sequence lengths for different situation and get the last RNN outputs by indexing.

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

Shuffle Layer
^^^^^^^^^^^^^^^^^
.. autoclass:: Shuffle

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
..                      Helper Functions
.. -----------------------------------------------------------

Helper Functions
------------------------

Flatten tensor
^^^^^^^^^^^^^^^^^
.. autofunction:: flatten_reshape

Initialize RNN state
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: initialize_rnn_state

Remove repeated items in a list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: list_remove_repeat

