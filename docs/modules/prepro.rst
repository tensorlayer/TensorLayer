respective_API - Data Pre-processing
====================================

TensorLayer high-performance visual data augmentation API
-----------------------------------------------------------

Data pre-processing is a critical step in deep learning.
Though TensorFlow has provided rich high-performance
data processing modules (e.g., ``tf.data``),
visual data augmentation often remains as a key bottleneck.
To address this bottleneck, TensorFlow suggest
to use optimized ``tf.image`` operators within user's
augmentation functions written in Python. This approach, however, has two limitations:

- Real-world visual tasks such as object detection, segmentation, and pose estimation
need to cope with unstructured image meta-data (e.g., coordinates).
These data are beyond the offering of ``tf.image``
which processes images as tensors.
Also, the use of ``tf.image`` operators
breaks the pure Python programing experience (i.e., users have to
use ``tf.py_func`` in order to keep using Python functions), incurring extra cross-language performance overheads.

- ``tf.image`` API is inflexible. Image operations must be
applied independently and are hard to jointly optimize. Mostly importantly,
a pipeline of independent image operations can significantly
reduces the quality of input images, thus affecting training accuracy.

TensorLayer address these two limitations by providing a flexible
and high-performance image augmentation Python API, compatible with
the ``tf.image``. This API is based on affine transformation.
It allows you to combine multiple image operations into
a single fast operation, offering up to 100x performance improvement (for example in
`openpose-plus <https://github.com/tensorlayer/openpose-plus>`_).
The following example illustrates the rationale
behind this tremendous speed up.





(XXX) We provide abundant data augmentation and processing functions by using Numpy, Scipy, Threading and Queue.
However, we recommend you to use TensorFlow operation function like ``tf.image.central_crop``,
more TensorFlow data augmentation method can be found
`here <https://www.tensorflow.org/api_guides/python/image.html>`_ and ``tutorial_cifar10_tfrecord.py``.


Magic for Fast Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The direct way to apply rotation, shifting, flippling, zooming and shearing to an image in pure Python is as follows,

.. code-block:: python

    image = tl.vis.read_image('tiger.jpeg')

    xx = tl.prepro.rotation(image, rg=20, is_random=False)
    xx = tl.prepro.shift(xx, wrg=0.2, hrg=0.2, is_random=False)
    xx = tl.prepro.flip_axis(xx, axis=1, is_random=False)
    xx = tl.prepro.zoom(xx, zoom_range=(0.8, 1.5), is_random=False)
    xx = tl.prepro.shear(xx, intensity=0.2, is_random=False)

    tl.vis.save_image(xx, '_result_slow.png')

Let do some math

<insert image here>




Therefore, all transformations can be combined into one:

.. code-block:: python

    # 1. get all affine transform matrices
    M_rotate = tl.prepro.affine_rotation_matrix(rg=20, is_random=False)
    M_flip = tl.prepro.affine_horizontal_flip_matrix(is_random=False)
    M_shift = tl.prepro.affine_shift_matrix(wrg=0.2, hrg=0.2, h=h, w=w, is_random=False)
    M_shear = tl.prepro.affine_shear_matrix(intensity=0.2, is_random=False)
    M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.8, 1.5), is_random=False)

    # 2. combine all affine transform matrices to one matrix, the rotation is the first transformation
    M_combined = M_rotate.dot(M_shift).dot(M_flip).dot(M_zoom).dot(M_shear)

    # 2. transfrom the matrix from Cartesian coordinate (the origin in the middle of image)
    # to Image coordinate (the origin on the top-left of image)
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, h, w)

    # 3. then we can transfrom the image once for all transformations
    result = tl.prepro.affine_transfrom(image, transform_matrix)

    tl.vis.save_image(result, '_result_fast.png')

Our experiments show that using Python for data augmentation would not be the training bottleneck

The code of this tutorial can be found `here <https://github.com/tensorlayer/tensorlayer/tree/master/examples/data_process/tutorial_fast_affine_transform.py>`__.



.. automodule:: tensorlayer.prepro

.. autosummary::

   affine_rotation_matrix
   affine_horizontal_flip_matrix
   affine_shift_matrix
   affine_shear_matrix
   affine_shear_matrix2
   affine_zoom_matrix
   affine_respective_zoom_matrix

   transform_matrix_offset_center
   affine_transfrom
   projective_transform_by_points

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
   shear2
   shear_multi2
   swirl
   swirl_multi
   elastic_transform
   elastic_transform_multi

   zoom
   respective_zoom
   zoom_multi

   brightness
   brightness_multi

   illumination

   rgb_to_hsv
   hsv_to_rgb
   adjust_hue

   imresize

   pixel_value_scale

   samplewise_norm
   featurewise_norm

   channel_shift
   channel_shift_multi

   drop

   array_to_img

   find_contours
   pt2map
   binary_dilation
   dilation
   binary_erosion
   erosion


   obj_box_coord_rescale
   obj_box_coords_rescale
   obj_box_coord_scale_to_pixelunit
   obj_box_coord_centroid_to_upleft_butright
   obj_box_coord_upleft_butright_to_centroid
   obj_box_coord_centroid_to_upleft
   obj_box_coord_upleft_to_centroid

   parse_darknet_ann_str_to_list
   parse_darknet_ann_list_to_cls_box

   obj_box_horizontal_flip
   obj_box_imresize
   obj_box_crop
   obj_box_shift
   obj_box_zoom

   keypoint_random_crop
   keypoint_random_crop2
   keypoint_random_rotate
   keypoint_random_flip
   keypoint_random_resize
   keypoint_random_resize_shortestedge

   pad_sequences
   remove_pad_sequences
   process_sequences
   sequences_add_start_id
   sequences_add_end_id
   sequences_add_end_id_after_pad
   sequences_get_mask


..
  Threading
  ------------
  .. autofunction:: threading_data

Affine Transform
------------------

Get rotation matrix
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: affine_rotation_matrix

Get flippling matrix
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: affine_horizontal_flip_matrix

Get shifting matrix
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: affine_shift_matrix

Get shearing matrix
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: affine_shear_matrix

Get shearing matrix V2
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: affine_shear_matrix2

Get zooming matrix
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: affine_zoom_matrix

Get seperated zooming matrix
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: affine_zoom_matrix

Transform matrix offset
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: transform_matrix_offset_center

Apply affine transform by matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: affine_transfrom

Projective transform by points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: projective_transform_by_points


Images
-----------

- These functions only apply on a single image, use ``threading_data`` to apply multiple threading see ``tutorial_image_preprocess.py``.
- All functions have argument ``is_random``.
- All functions end with ``*_multi`` process all images together, usually be used for image segmentation i.e. the input and output image should be matched.

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

Shear V2
^^^^^^^^^^^
.. autofunction:: shear2
.. autofunction:: shear_multi2

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

Respective Zoom
^^^^^^^^^^^^^^^^^
.. autofunction:: respective_zoom

Brightness
^^^^^^^^^^^^
.. autofunction:: brightness
.. autofunction:: brightness_multi

Brightness, contrast and saturation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: illumination

RGB to HSV
^^^^^^^^^^^^^^
.. autofunction:: rgb_to_hsv

HSV to RGB
^^^^^^^^^^^^^^
.. autofunction:: hsv_to_rgb

Adjust Hue
^^^^^^^^^^^^^^
.. autofunction:: adjust_hue

Resize
^^^^^^^^^^^^
.. autofunction:: imresize

Pixel value scale
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: pixel_value_scale

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

Binary erosion
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: binary_erosion

Greyscale erosion
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: erosion



Object detection
-------------------

Tutorial for Image Aug
^^^^^^^^^^^^^^^^^^^^^^^

Hi, here is an example for image augmentation on VOC dataset.

.. code-block:: python

  import tensorlayer as tl

  ## download VOC 2012 dataset
  imgs_file_list, _, _, _, classes, _, _,\
      _, objs_info_list, _ = tl.files.load_voc_dataset(dataset="2012")

  ## parse annotation and convert it into list format
  ann_list = []
  for info in objs_info_list:
      ann = tl.prepro.parse_darknet_ann_str_to_list(info)
      c, b = tl.prepro.parse_darknet_ann_list_to_cls_box(ann)
      ann_list.append([c, b])

  # read and save one image
  idx = 2  # you can select your own image
  image = tl.vis.read_image(imgs_file_list[idx])
  tl.vis.draw_boxes_and_labels_to_image(image, ann_list[idx][0],
       ann_list[idx][1], [], classes, True, save_name='_im_original.png')

  # left right flip
  im_flip, coords = tl.prepro.obj_box_horizontal_flip(image,
          ann_list[idx][1], is_rescale=True, is_center=True, is_random=False)
  tl.vis.draw_boxes_and_labels_to_image(im_flip, ann_list[idx][0],
          coords, [], classes, True, save_name='_im_flip.png')

  # resize
  im_resize, coords = tl.prepro.obj_box_imresize(image,
          coords=ann_list[idx][1], size=[300, 200], is_rescale=True)
  tl.vis.draw_boxes_and_labels_to_image(im_resize, ann_list[idx][0],
          coords, [], classes, True, save_name='_im_resize.png')

  # crop
  im_crop, clas, coords = tl.prepro.obj_box_crop(image, ann_list[idx][0],
           ann_list[idx][1], wrg=200, hrg=200,
           is_rescale=True, is_center=True, is_random=False)
  tl.vis.draw_boxes_and_labels_to_image(im_crop, clas, coords, [],
           classes, True, save_name='_im_crop.png')

  # shift
  im_shfit, clas, coords = tl.prepro.obj_box_shift(image, ann_list[idx][0],
          ann_list[idx][1], wrg=0.1, hrg=0.1,
          is_rescale=True, is_center=True, is_random=False)
  tl.vis.draw_boxes_and_labels_to_image(im_shfit, clas, coords, [],
          classes, True, save_name='_im_shift.png')

  # zoom
  im_zoom, clas, coords = tl.prepro.obj_box_zoom(image, ann_list[idx][0],
          ann_list[idx][1], zoom_range=(1.3, 0.7),
          is_rescale=True, is_center=True, is_random=False)
  tl.vis.draw_boxes_and_labels_to_image(im_zoom, clas, coords, [],
          classes, True, save_name='_im_zoom.png')


In practice, you may want to use threading method to process a batch of images as follows.

.. code-block:: python

  import tensorlayer as tl
  import random

  batch_size = 64
  im_size = [416, 416]
  n_data = len(imgs_file_list)
  jitter = 0.2
  def _data_pre_aug_fn(data):
      im, ann = data
      clas, coords = ann
      ## change image brightness, contrast and saturation randomly
      im = tl.prepro.illumination(im, gamma=(0.5, 1.5),
               contrast=(0.5, 1.5), saturation=(0.5, 1.5), is_random=True)
      ## flip randomly
      im, coords = tl.prepro.obj_box_horizontal_flip(im, coords,
               is_rescale=True, is_center=True, is_random=True)
      ## randomly resize and crop image, it can have same effect as random zoom
      tmp0 = random.randint(1, int(im_size[0]*jitter))
      tmp1 = random.randint(1, int(im_size[1]*jitter))
      im, coords = tl.prepro.obj_box_imresize(im, coords,
              [im_size[0]+tmp0, im_size[1]+tmp1], is_rescale=True,
               interp='bicubic')
      im, clas, coords = tl.prepro.obj_box_crop(im, clas, coords,
               wrg=im_size[1], hrg=im_size[0], is_rescale=True,
               is_center=True, is_random=True)
      ## rescale value from [0, 255] to [-1, 1] (optional)
      im = im / 127.5 - 1
      return im, [clas, coords]

  # randomly read a batch of image and the corresponding annotations
  idexs = tl.utils.get_random_int(min=0, max=n_data-1, number=batch_size)
  b_im_path = [imgs_file_list[i] for i in idexs]
  b_images = tl.prepro.threading_data(b_im_path, fn=tl.vis.read_image)
  b_ann = [ann_list[i] for i in idexs]

  # threading process
  data = tl.prepro.threading_data([_ for _ in zip(b_images, b_ann)],
                _data_pre_aug_fn)
  b_images2 = [d[0] for d in data]
  b_ann = [d[1] for d in data]

  # save all images
  for i in range(len(b_images)):
      tl.vis.draw_boxes_and_labels_to_image(b_images[i],
               ann_list[idexs[i]][0], ann_list[idexs[i]][1], [],
               classes, True, save_name='_bbox_vis_%d_original.png' % i)
      tl.vis.draw_boxes_and_labels_to_image((b_images2[i]+1)*127.5,
               b_ann[i][0], b_ann[i][1], [], classes, True,
               save_name='_bbox_vis_%d.png' % i)

Image Aug with TF Dataset API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Example code for VOC `here <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_tf_dataset_voc.py>`__.

Coordinate pixel unit to percentage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coord_rescale

Coordinates pixel unit to percentage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coords_rescale

Coordinate percentage to pixel unit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coord_scale_to_pixelunit

Coordinate [x_center, x_center, w, h] to up-left button-right
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coord_centroid_to_upleft_butright

Coordinate up-left button-right to [x_center, x_center, w, h]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coord_upleft_butright_to_centroid

Coordinate [x_center, x_center, w, h] to up-left-width-high
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coord_centroid_to_upleft

Coordinate up-left-width-high to [x_center, x_center, w, h]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_coord_upleft_to_centroid

Darknet format string to list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: parse_darknet_ann_str_to_list

Darknet format split class and coordinate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: parse_darknet_ann_list_to_cls_box

Image Aug - Flip
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_horizontal_flip

Image Aug - Resize
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_imresize

Image Aug - Crop
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_crop

Image Aug - Shift
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction::  obj_box_shift

Image Aug - Zoom
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: obj_box_zoom

Keypoints
------------

Image Aug - Crop
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: keypoint_random_crop

.. autofunction:: keypoint_random_crop2

Image Aug - Rotate
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: keypoint_random_rotate

Image Aug - Flip
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: keypoint_random_flip

Image Aug - Resize
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: keypoint_random_resize

Image Aug - Resize Shortest Edge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: keypoint_random_resize_shortestedge


Sequence
---------

More related functions can be found in ``tensorlayer.nlp``.

Padding
^^^^^^^^^
.. autofunction:: pad_sequences

Remove Padding
^^^^^^^^^^^^^^^^^
.. autofunction:: remove_pad_sequences


Process
^^^^^^^^^
.. autofunction:: process_sequences

Add Start ID
^^^^^^^^^^^^^^^
.. autofunction:: sequences_add_start_id


Add End ID
^^^^^^^^^^^^^^^
.. autofunction:: sequences_add_end_id

Add End ID after pad
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: sequences_add_end_id_after_pad

Get Mask
^^^^^^^^^
.. autofunction:: sequences_get_mask
