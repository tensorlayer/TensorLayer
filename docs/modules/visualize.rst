API - Visualization
================================

TensorFlow provides `TensorBoard <https://www.tensorflow.org/get_started/summaries_and_tensorboard>`_
to visualize the model, activations etc. Here we provide more functions for data visualization.

.. automodule:: tensorlayer.visualize

.. autosummary::

   read_image
   read_images
   save_image
   save_images
   draw_boxes_and_labels_to_image
   W
   CNN2d
   frame
   images2d
   tsne_embedding


Save and read images
----------------------

Read one image
^^^^^^^^^^^^^^^^^
.. autofunction:: read_image

Read multiple images
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: read_images

Save one image
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_image

Save multiple images
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_images

Save image for object detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: draw_boxes_and_labels_to_image



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
