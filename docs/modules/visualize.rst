API - Visualize Model and Data
================================

TensorFlow provides `TensorBoard <https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html>`_
to visualize the model, activations etc. Here we provide more functions for data visualization.

.. automodule:: tensorlayer.visualize

.. autosummary::

   W
   CNN2d
   frame
   images2d
   tsne_embedding

Visualize model parameters
------------------------------

Visualize weight matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: W

Visualize CNN 2d filter
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: CNN2d

Visualize images
-----------------

Image by matplotlib
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: frame

Images by matplotlib
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: images2d

Visualize embeddings
--------------------

.. autofunction:: tsne_embedding
