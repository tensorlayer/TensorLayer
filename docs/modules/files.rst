API - Load, Save Model and Data
===================================

Load benchmark dataset, save and restore model, save and load variables.
TensorFlow provides ``.ckpt`` file format to save and restore the models, while
we suggest to use standard python file format ``.npz`` to save models for the
sake of cross-platform.


.. code-block:: python

  # save model as .ckpt
  saver = tf.train.Saver()
  save_path = saver.save(sess, "model.ckpt")
  # restore model from .ckpt
  saver = tf.train.Saver()
  saver.restore(sess, "model.ckpt")

  # save model as .npz
  tl.files.save_npz(network.all_params , name='model.npz')

  # restore model from .npz
  load_params = tl.files.load_npz(path='', name='model.npz')
  tl.files.assign_params(sess, load_params, network)

  # you can assign the pre-trained parameters as follow
  # 1st parameter
  tl.files.assign_params(sess, [load_params[0]], network)
  # the first three parameters
  tl.files.assign_params(sess, load_params[:3], network)

.. automodule:: tensorlayer.files

.. autosummary::

   load_mnist_dataset
   load_cifar10_dataset
   load_ptb_dataset
   load_matt_mahoney_text8_dataset
   load_imbd_dataset
   load_nietzsche_dataset
   load_wmt_en_fr_dataset

   save_npz
   load_npz
   assign_params

   save_any_to_npy
   load_npy_to_any

   npz_to_W_pdf

   load_file_list

Load dataset functions
------------------------

MNIST
^^^^^^^
.. autofunction:: load_mnist_dataset

CIFAR-10
^^^^^^^^^^^^
.. autofunction:: load_cifar10_dataset

Penn TreeBank (PTB)
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_ptb_dataset

Matt Mahoney's text8
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_matt_mahoney_text8_dataset

IMBD
^^^^^^^^^^^
.. autofunction:: load_imbd_dataset

Nietzsche
^^^^^^^^^^^^^^
.. autofunction:: load_nietzsche_dataset


English-to-French translation data from the WMT'15 Website
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_wmt_en_fr_dataset


Load and save network
----------------------

Save network as .npz
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_npz

Load network from .npz
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_npz

Assign parameters to network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: assign_params

Load and save variables
------------------------

Save variables as .npy
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_any_to_npy

Load variables from .npy
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_npy_to_any


Visualizing npz file
----------------------
.. autofunction:: npz_to_W_pdf


Helper functions
------------------

.. autofunction:: load_file_list
