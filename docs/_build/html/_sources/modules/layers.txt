:mod:`tensorlayer.layers`
=========================


.. automodule:: tensorlayer.layers

.. autosummary::

   Layer
   InputLayer
   Word2vecEmbeddingInputlayer
   DenseLayer
   ReconLayer
   DropoutLayer
   DropconnectDenseLayer
   Conv2dLayer
   PoolLayer
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

Shape layer
----------------

.. autoclass:: FlattenLayer
.. autoclass:: ConcatLayer

Coming soon
----------------

.. autoclass:: MaxoutLayer
.. autoclass:: ResnetLayer
.. autoclass:: GaussianNoiseLayer
.. autoclass:: ReshapeLayer


Helper functions
----------------

.. autofunction:: flatten_reshape
