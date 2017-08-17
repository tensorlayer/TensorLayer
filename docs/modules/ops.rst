API - Operation System
=======================

Operation system, more functions can be found in `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`_.

.. automodule:: tensorlayer.ops

.. autosummary::

   exit_tf
   open_tb
   clear_all
   set_gpu_fraction
   disable_print
   enable_print
   suppress_stdout
   get_site_packages_directory
   empty_trash

TensorFlow functions
---------------------------

Close TF session and associated processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: exit_tf

Open TensorBoard
^^^^^^^^^^^^^^^^^^^
.. autofunction:: open_tb

Delete placeholder
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: clear_all

GPU functions
---------------------------
.. autofunction:: set_gpu_fraction


Console display
------------------

Disable print
^^^^^^^^^^^^^^^
.. autofunction:: disable_print

Enable print
^^^^^^^^^^^^^^^
.. autofunction:: enable_print

Temporary disable print
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: suppress_stdout

Site packages information
----------------------------
.. autofunction:: get_site_packages_directory

Trash
-------
.. autofunction:: empty_trash
