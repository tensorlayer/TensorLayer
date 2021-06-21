API - Activations
=========================

To make TensorLayer simple, we minimize the number of activation functions as much as
we can. So we encourage you to use Customizes activation function.
For parametric activation, please read the layer APIs.

The shortcut of ``tensorlayer.activation`` is ``tensorlayer.act``.

Your activation
-------------------

Customizes activation function in TensorLayer is very easy.
The following example implements an activation that multiplies its input by 2.
For more complex activation, TensorFlow(MindSpore/PaddlePaddle) API will be required.

.. code-block:: python

  class DoubleActivation(object):
    def __init__(self):
        pass
    def __call__(self, x):
        return x * 2
  double_activation = DoubleActivation()

.. automodule:: tensorlayer.layers.activation

.. autosummary::

   PRelu
   PRelu6
   PTRelu6
   LeakyReLU
   LeakyReLU6
   LeakyTwiceRelu6
   Ramp
   Swish
   HardTanh
   Mish

PRelu
------
.. autofunction:: PRelu

PRelu6
------------
.. autofunction:: PRelu6

PTRelu6
------------
.. autofunction:: PTRelu6

LeakyReLU
-----------------
.. autofunction:: LeakyReLU

LeakyReLU6
------------
.. autofunction:: LeakyReLU6

LeakyTwiceRelu6
---------------------
.. autofunction:: LeakyTwiceRelu6

Ramp
---------------------
.. autofunction:: Ramp

Swish
--------------------
.. autofunction:: Swish

HardTanh
----------------
.. autofunction:: HardTanh

Mish
---------
.. autofunction:: Mish

Parametric activation
------------------------------
See ``tensorlayer.layers``.
