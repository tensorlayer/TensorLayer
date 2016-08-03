:mod:`tunelayer.files`
========================

Load benchmark dataset, save and restore model, save and load variables.


.. automodule:: tunelayer.files

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
   save_any_to_npy

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
.. autofunction:: save_any_to_npy


Visualizing npz file
----------------------
.. autofunction:: npz_to_W_pdf


Helper functions
------------------

.. autofunction:: load_file_list
