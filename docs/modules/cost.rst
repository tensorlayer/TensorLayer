API - Cost
==================

To make TensorLayer simple, we minimize the number of cost functions as much as
we can. So we encourage you to use TensorFlow's function, , see `TensorFlow API <https://www.tensorflow.org/versions/r2.0/api_docs/python/tf>`_.

.. note::
    Please refer to `Getting Started <https://github.com/tensorlayer/tensorlayer/tree/master/docs/user>`_ for getting specific weights for weight regularization.

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
   huber_loss


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
``tf.contrib.layers.sum_regularizer``, see tensorflow API.
Maxnorm
^^^^^^^^^^
.. autofunction:: maxnorm_regularizer

Special
^^^^^^^^^^
.. autofunction:: li_regularizer
.. autofunction:: lo_regularizer
.. autofunction:: maxnorm_o_regularizer
.. autofunction:: maxnorm_i_regularizer

Huber Loss
^^^^^^^^^^
.. autofunction:: huber_loss