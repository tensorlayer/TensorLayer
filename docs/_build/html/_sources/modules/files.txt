:mod:`tensorlayer.files`
========================

.. automodule:: tensorlayer.files

.. autosummary::

   load_mnist_dataset
   load_cifar10_dataset
   load_ptb_dataset
   load_matt_mahoney_text8_dataset
   load_imbd_dataset

   read_words
   read_analogies_file
   build_vocab
   build_reverse_dictionary
   build_words_dataset
   words_to_word_ids
   word_ids_to_words
   save_vocab

   save_npz
   load_npz
   assign_params

   npz_to_W_pdf

   save_any_to_npy
   save_any_to_npy

   load_file_list

Load dataset functions
------------------------

.. autofunction:: load_mnist_dataset
.. autofunction:: load_cifar10_dataset
.. autofunction:: load_ptb_dataset
.. autofunction:: load_matt_mahoney_text8_dataset
.. autofunction:: load_imbd_dataset

Vector representations of words
-------------------------------

.. autofunction:: read_words
.. autofunction:: read_analogies_file
.. autofunction:: build_vocab
.. autofunction:: build_reverse_dictionary
.. autofunction:: build_words_dataset
.. autofunction:: words_to_word_ids
.. autofunction:: word_ids_to_words
.. autofunction:: save_vocab

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
---------------------------------------
.. autofunction:: npz_to_W_pdf


Helper functions
----------------------

.. autofunction:: load_file_list
