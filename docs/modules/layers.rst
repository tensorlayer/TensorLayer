:mod:`tensorlayer.layers`
=========================

To make TensorLayer simple, we minimize the number of layer classes as much as
we can. So we encourage you to use TensorFlow's function.
For example, we do not provide layer for local response normalization, we suggest
you to apply ``tf.nn.lrn`` on ``Layer.outputs``.
More functions can be found in `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_

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
