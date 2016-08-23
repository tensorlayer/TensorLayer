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

.. autofunction:: load_mnist_dataset
.. autofunction:: load_cifar10_dataset
.. autofunction:: load_ptb_dataset
.. autofunction:: load_matt_mahoney_text8_dataset
.. autofunction:: load_imbd_dataset
.. autofunction:: load_nietzsche_dataset
.. autofunction:: load_wmt_en_fr_dataset


Load and save network
----------------------

.. autofunction:: save_npz
.. autofunction:: load_npz
.. autofunction:: assign_params

Load and save variables
------------------------
.. autofunction:: save_any_to_npy
.. autofunction:: load_npy_to_any


Visualizing npz file
----------------------
.. autofunction:: npz_to_W_pdf


Helper functions
------------------

.. autofunction:: load_file_list
