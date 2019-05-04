API - Initializers
=========================

To make TensorLayer simple, TensorLayer only warps some basic initializers. For more advanced initializer,
e.g. ``tf.initializers.he_normal``, please refer to TensorFlow provided initializers
`here <https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/initializers>`_.

.. automodule:: tensorlayer.initializers

.. autosummary::

   Initializer
   Zeros
   Ones
   Constant
   RandomUniform
   RandomNormal
   TruncatedNormal
   deconv2d_bilinear_upsampling_initializer

Initializer
------------
.. autoclass:: Initializer

Zeros
------------
.. autoclass:: Zeros

Ones
------------
.. autoclass:: Ones

Constant
-----------------
.. autoclass:: Constant

RandomUniform
--------------
.. autoclass:: RandomUniform

RandomNormal
---------------------
.. autoclass:: RandomNormal

TruncatedNormal
---------------------
.. autoclass:: TruncatedNormal

deconv2d_bilinear_upsampling_initializer
------------------------------------------
.. autofunction:: deconv2d_bilinear_upsampling_initializer
