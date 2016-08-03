:mod:`tunelayer.cost`
=======================

To make TuneLayer simple, we minimize the number of cost functions as much as
we can. So we encourage you to use TensorFlow's function.
For example, you can implement L1, L2 and sum regularization by
``tf.contrib.layers.l1_regularizer``, ``tf.contrib.layers.l2_regularizer`` and
``tf.contrib.layers.sum_regularizer``, see `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_.


.. automodule:: tunelayer.cost

.. autosummary::

   cross_entropy
   mean_squared_error
   li_regularizer
   lo_regularizer
   maxnorm_regularizer
   maxnorm_o_regularizer
   maxnorm_i_regularizer

Cost functions
----------------

.. autofunction:: cross_entropy
.. autofunction:: mean_squre_error


Regularization functions
--------------------------

.. autofunction:: li_regularizer
.. autofunction:: lo_regularizer
.. autofunction:: maxnorm_regularizer
.. autofunction:: maxnorm_o_regularizer
.. autofunction:: maxnorm_i_regularizer
