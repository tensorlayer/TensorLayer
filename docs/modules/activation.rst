API - Activations
=========================

To make TensorLayer simple, we minimize the number of activation functions as much as
we can. So we encourage you to use TensorFlow's function. TensorFlow provides
``tf.nn.relu``, ``tf.nn.relu6``, ``tf.nn.elu``, ``tf.nn.softplus``,
``tf.nn.softsign`` and so on.
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
      
  double_activation = lambda x: x * 2

.. automodule:: tensorlayer.activation

.. autosummary::

   leaky_relu
   leaky_relu6
   leaky_twice_relu6
   ramp
   swish
   sign
   hard_tanh
   pixel_wise_softmax

Ramp
------
.. autofunction:: ramp

Leaky ReLU
------------
.. autofunction:: leaky_relu

Leaky ReLU6
------------
.. autofunction:: leaky_relu6

Twice Leaky ReLU6
-----------------
.. autofunction:: leaky_twice_relu6

Swish
------------
.. autofunction:: swish

Sign
---------------------
.. autofunction:: sign

Hard Tanh
---------------------
.. autofunction:: hard_tanh

Pixel-wise softmax
--------------------
.. autofunction:: pixel_wise_softmax

Parametric activation
------------------------------
See ``tensorlayer.layers``.
