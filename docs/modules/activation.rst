API - Activations
=========================

To make TensorLayer simple, we minimize the number of activation functions as much as
we can. So we encourage you to use TensorFlow's function. TensorFlow provides
``tf.nn.relu``, ``tf.nn.relu6``, ``tf.nn.elu``, ``tf.nn.softplus``,
``tf.nn.softsign`` and so on. More TensorFlow official activation functions can be found
`here <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#activation-functions>`_.
For parametric activation, please read the layer APIs.

The shortcut of ``tensorlayer.activation`` is ``tensorlayer.act``.

Your activation
-------------------

Customizes activation function in TensorLayer is very easy.
The following example implements an activation that multiplies its input by 2.
For more complex activation, TensorFlow API will be required.

.. code-block:: python

  def double_activation(x):
      return x * 2

.. automodule:: tensorlayer.activation

.. autosummary::

   identity
   ramp
   leaky_relu
   swish
   pixel_wise_softmax

Identity
-------------
.. autofunction:: identity

Ramp
------
.. autofunction:: ramp

Leaky Relu
------------
.. autofunction:: leaky_relu

Swish
------------
.. autofunction:: swish

Pixel-wise softmax
--------------------
.. autofunction:: pixel_wise_softmax

Parametric activation
------------------------------
See ``tensorlayer.layers``.
