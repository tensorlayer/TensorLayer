:mod:`tensorlayer.layers`
=========================


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
   flatten_reshape


Basic layer
----------------

.. autoclass:: Layer

Input layer
----------------

.. autoclass:: InputLayer

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

Coming soon
----------------

.. autoclass:: MaxoutLayer
.. autoclass:: GaussianNoiseLayer
.. autoclass:: ReshapeLayer
.. autoclass:: BidirectionalRNNLayer


Helper functions
----------------

.. autofunction:: flatten_reshape
