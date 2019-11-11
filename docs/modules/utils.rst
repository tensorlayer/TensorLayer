API - Utility
========================

.. automodule:: tensorlayer.utils

.. autosummary::

   fit
   test
   predict
   evaluation
   class_balancing_oversample
   get_random_int
   dict_to_one
   list_string_to_dict
   flatten_list
   exit_tensorflow
   open_tensorboard
   set_gpu_fraction

Training, testing and predicting
----------------------------------

Training
^^^^^^^^^^^
.. autofunction:: fit

Evaluation
^^^^^^^^^^^^^
.. autofunction:: test

Prediction
^^^^^^^^^^^^
.. autofunction:: predict

Evaluation functions
---------------------
.. autofunction:: evaluation

Class balancing functions
----------------------------
.. autofunction:: class_balancing_oversample

Random functions
----------------------------
.. autofunction:: get_random_int

Dictionary and list
--------------------

Set all items in dictionary to one
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: dict_to_one

Convert list of string to dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: list_string_to_dict

Flatten a list
^^^^^^^^^^^^^^^^^^^
.. autofunction:: flatten_list

Close TF session and associated processes
-----------------------------------------
.. autofunction:: exit_tensorflow

Open TensorBoard
----------------
.. autofunction:: open_tensorboard

Set GPU functions
-----------------
.. autofunction:: set_gpu_fraction
