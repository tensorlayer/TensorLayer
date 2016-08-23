API - Activations
=========================

To make TensorLayer simple, we minimize the number of activation functions as much as
we can. So we encourage you to use TensorFlow's function. TensorFlow provides
``tf.nn.relu``, ``tf.nn.relu6``, ``tf.nn.elu``, ``tf.nn.softplus``,
``tf.nn.softsign`` and so on. More TensorFlow official activation functions can be found
`here <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#activation-functions>`_.


Creating custom activation
---------------------------

To implement a custom activation function in TensorLayer is very easy.

The following is an example implementation of an activation that multiplies its input by 2.
For more complex activation, TensorFlow API will be required.

.. code-block:: python

  def double_activation(x):
      return x * 2



.. automodule:: tensorlayer.activation

.. autosummary::

   identity
   ramp


Activation functions
---------------------

.. autofunction:: identity
.. autofunction:: ramp
