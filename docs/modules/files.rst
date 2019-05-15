API - Files
===================================

A collections of helper functions to work with dataset.
Load benchmark dataset, save and restore model, save and load variables.

.. automodule:: tensorlayer.files

.. autosummary::

   load_mnist_dataset
   load_fashion_mnist_dataset
   load_cifar10_dataset
   load_cropped_svhn
   load_ptb_dataset
   load_matt_mahoney_text8_dataset
   load_imdb_dataset
   load_nietzsche_dataset
   load_wmt_en_fr_dataset
   load_flickr25k_dataset
   load_flickr1M_dataset
   load_cyclegan_dataset
   load_celebA_dataset
   load_voc_dataset
   load_mpii_pose_dataset
   download_file_from_google_drive

   save_npz
   load_npz
   assign_weights
   load_and_assign_npz
   save_npz_dict
   load_and_assign_npz_dict
   save_weights_to_hdf5
   load_hdf5_to_weights_in_order
   load_hdf5_to_weights

   save_any_to_npy
   load_npy_to_any

   file_exists
   folder_exists
   del_file
   del_folder
   read_file
   load_file_list
   load_folder_list
   exists_or_mkdir
   maybe_download_and_extract

   natural_keys

..
   save_ckpt
   load_ckpt
   save_graph
   load_graph
   save_graph_and_params
   load_graph_and_params

   npz_to_W_pdf


Load dataset functions
------------------------

MNIST
^^^^^^^
.. autofunction:: load_mnist_dataset

Fashion-MNIST
^^^^^^^^^^^^^^^^
.. autofunction:: load_fashion_mnist_dataset

CIFAR-10
^^^^^^^^^^^^
.. autofunction:: load_cifar10_dataset

SVHN
^^^^^^^
.. autofunction:: load_cropped_svhn

Penn TreeBank (PTB)
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_ptb_dataset

Matt Mahoney's text8
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_matt_mahoney_text8_dataset

IMBD
^^^^^^^^^^^
.. autofunction:: load_imdb_dataset

Nietzsche
^^^^^^^^^^^^^^
.. autofunction:: load_nietzsche_dataset

English-to-French translation data from the WMT'15 Website
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_wmt_en_fr_dataset

Flickr25k
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_flickr25k_dataset

Flickr1M
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_flickr1M_dataset

CycleGAN
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_cyclegan_dataset

CelebA
^^^^^^^^^
.. autofunction:: load_celebA_dataset

VOC 2007/2012
^^^^^^^^^^^^^^^^
.. autofunction:: load_voc_dataset

MPII
^^^^^^^^^^^^^^^^
.. autofunction:: load_mpii_pose_dataset

Google Drive
^^^^^^^^^^^^^^^^
.. autofunction:: download_file_from_google_drive





Load and save network
----------------------

TensorFlow provides ``.ckpt`` file format to save and restore the models, while
we suggest to use standard python file format ``hdf5`` to save models for the
sake of cross-platform. Other file formats such as ``.npz`` are also available.

.. code-block:: python

  ## save model as .h5
  tl.files.save_weights_to_hdf5('model.h5', network.all_weights)
  # restore model from .h5 (in order)
  tl.files.load_hdf5_to_weights_in_order('model.h5', network.all_weights)
  # restore model from .h5 (by name)
  tl.files.load_hdf5_to_weights('model.h5', network.all_weights)

  ## save model as .npz
  tl.files.save_npz(network.all_weights , name='model.npz')
  # restore model from .npz (method 1)
  load_params = tl.files.load_npz(name='model.npz')
  tl.files.assign_weights(sess, load_params, network)
  # restore model from .npz (method 2)
  tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)

  ## you can assign the pre-trained parameters as follow
  # 1st parameter
  tl.files.assign_weights(sess, [load_params[0]], network)
  # the first three parameters
  tl.files.assign_weights(sess, load_params[:3], network)

Save network into list (npz)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_npz

Load network from list (npz)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_npz

Assign a list of parameters to network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: assign_weights

Load and assign a list of parameters to network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_and_assign_npz


Save network into dict (npz)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_npz_dict

Load network from dict (npz)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_and_assign_npz_dict

Save network into OrderedDict (hdf5)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_weights_to_hdf5

Load network from hdf5 in order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_hdf5_to_weights_in_order

Load network from hdf5 by name
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_hdf5_to_weights

..
  Save network architecture as a graph
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  .. autofunction:: save_graph

  Load network architecture from a graph
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  .. autofunction:: load_graph

  Save network architecture and parameters
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  .. autofunction:: save_graph_and_params

  Load network architecture and parameters
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  .. autofunction:: load_graph_and_params

..
  Save network into ckpt
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  .. autofunction:: save_ckpt

  Load network from ckpt
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  .. autofunction:: load_ckpt




Load and save variables
------------------------

Save variables as .npy
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: save_any_to_npy

Load variables from .npy
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_npy_to_any




Folder/File functions
------------------------

Check file exists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: file_exists

Check folder exists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: folder_exists

Delete file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: del_file

Delete folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: del_folder

Read file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: read_file

Load file list from folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_file_list

Load folder list from folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: load_folder_list

Check and Create folder
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: exists_or_mkdir

Download or extract
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: maybe_download_and_extract



Sort
-------

List of string with number in human order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: natural_keys

Visualizing npz file
----------------------
.. autofunction:: npz_to_W_pdf
