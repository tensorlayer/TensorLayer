API - Preprocessing
=========================


We provide abundant data augmentation and processing functions by using Numpy, Scipy, Threading and Queue.
However, we recommend you to use TensorFlow operation function like ``tf.image.central_crop``,
more TensorFlow data augmentation method can be found
`here <https://www.tensorflow.org/api_guides/python/image.html>`_ and ``tutorial_cifar10_tfrecord.py``.
Some of the code in this package are borrowed from Keras.

.. automodule:: tensorlayer.prepro

.. autosummary::

   threading_data

   rotation
   rotation_multi
   crop
   crop_multi
   flip_axis
   flip_axis_multi
   shift
   shift_multi

   shear
   shear_multi
   swirl
   swirl_multi
   elastic_transform
   elastic_transform_multi

   zoom
   zoom_multi
   brightness
   brightness_multi

   imresize

   samplewise_norm
   featurewise_norm

   channel_shift
   channel_shift_multi

   drop

   transform_matrix_offset_center
   apply_transform
   projective_transform_by_points

   array_to_img

   find_contours
   pt2map
   binary_dilation
   dilation

   pad_sequences
   process_sequences
   sequences_add_start_id
   sequences_get_mask

   distorted_images
   crop_central_whiten_images


Threading
------------
.. autofunction:: threading_data

Images
-----------

- These functions only apply on a single image, use ``threading_data`` to apply multiple threading see ``tutorial_image_preprocess.py``.
- All functions have argument ``is_random``.
- All functions end with `multi` , usually be used for image segmentation i.e. the input and output image should be matched.

Rotation
^^^^^^^^^
.. autofunction:: rotation
.. autofunction:: rotation_multi

Crop
^^^^^^^^^
.. autofunction:: crop
.. autofunction:: crop_multi

Flip
^^^^^^^^^
.. autofunction:: flip_axis
.. autofunction:: flip_axis_multi

Shift
^^^^^^^^^
.. autofunction:: shift
.. autofunction:: shift_multi

Shear
^^^^^^^^^
.. autofunction:: shear
.. autofunction:: shear_multi

Swirl
^^^^^^^^^
.. autofunction:: swirl
.. autofunction:: swirl_multi

Elastic transform
^^^^^^^^^^^^^^^^^^
.. autofunction:: elastic_transform
.. autofunction:: elastic_transform_multi

Zoom
^^^^^^^^^
.. autofunction:: zoom
.. autofunction:: zoom_multi

Brightness
^^^^^^^^^^^^
.. autofunction:: brightness
.. autofunction:: brightness_multi

Resize
^^^^^^^^^^^^
.. autofunction:: imresize

Normalization
^^^^^^^^^^^^^^^
.. autofunction:: samplewise_norm
.. autofunction:: featurewise_norm

Channel shift
^^^^^^^^^^^^^^
.. autofunction:: channel_shift
.. autofunction:: channel_shift_multi

Noise
^^^^^^^^^^^^^^
.. autofunction:: drop

Manual transform
^^^^^^^^^^^^^^^^^
.. autofunction:: transform_matrix_offset_center
.. autofunction:: apply_transform
.. autofunction:: projective_transform_by_points

Numpy and PIL
^^^^^^^^^^^^^^
.. autofunction:: array_to_img

Find contours
^^^^^^^^^^^^^^
.. autofunction:: find_contours

Points to Image
^^^^^^^^^^^^^^^^^
.. autofunction:: pt2map

Binary dilation
^^^^^^^^^^^^^^^^^
.. autofunction:: binary_dilation

Greyscale dilation
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: dilation

Sequence
---------

More related functions can be found in ``tensorlayer.nlp``.

Padding
^^^^^^^^^
.. autofunction:: pad_sequences

Process
^^^^^^^^^
.. autofunction:: process_sequences

Add Start ID
^^^^^^^^^^^^^^^
.. autofunction:: sequences_add_start_id

Get Mask
^^^^^^^^^
.. autofunction:: sequences_get_mask


Tensor Opt
------------

.. note::
  These functions will be deprecated, see ``tutorial_cifar10_tfrecord.py`` for new information.

.. autofunction:: distorted_images
.. autofunction:: crop_central_whiten_images
