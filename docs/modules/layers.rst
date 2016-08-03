:mod:`tunelayer.layers`
=========================

To make TuneLayer simple, we minimize the number of layer classes as much as
we can. So we encourage you to use TensorFlow's function.
For example, we do not provide layer for local response normalization, we suggest
you to apply ``tf.nn.lrn`` on ``Layer.outputs``.
More functions can be found in `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_

.. automodule:: tunelayer.layers

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
   flatten_reshape
   clear_layers_name
   set_name_reuse
   print_all_variables
   initialize_rnn_state


Basic layer
----------------

.. autoclass:: Layer

Input layer
----------------

.. autoclass:: InputLayer
  :members:

Word Embedding Input layer
----------------

.. autoclass:: Word2vecEmbeddingInputlayer
.. autoclass:: EmbeddingInputlayer

Dense layer
----------------

.. autoclass:: DenseLayer
.. autoclass:: ReconLayer
   :members:

Noise layer
----------------

.. autoclass:: DropoutLayer
.. autoclass:: DropconnectDenseLayer

Convolutional layer
--------------------

.. autoclass:: Conv2dLayer
.. autoclass:: PoolLayer

Recurrent layer
----------------

.. autoclass:: RNNLayer

Shape layer
----------------

.. autoclass:: FlattenLayer
.. autoclass:: ConcatLayer
.. autoclass:: ReshapeLayer

Developing or Untested
-------------------------

.. autoclass:: Conv3dLayer
.. autoclass:: MaxoutLayer
.. autoclass:: GaussianNoiseLayer
.. autoclass:: BidirectionalRNNLayer

Helper functions
----------------

.. autofunction:: flatten_reshape
.. autofunction:: clear_layers_name
.. autofunction:: set_name_reuse
.. autofunction:: print_all_variables
.. autofunction:: initialize_rnn_state
