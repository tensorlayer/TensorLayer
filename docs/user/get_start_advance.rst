.. _getstartadvance:

==================
Advanced features
==================


Pre-trained CNN
================

Get entire CNN
---------------

.. code-block:: python

  import tensorflow as tf
  import tensorlayer as tl
  import numpy as np
  from tensorlayer.models.imagenet_classes import class_names

  vgg = tl.models.vgg16(pretrained=True)
  img = tl.vis.read_image('data/tiger.jpeg')
  img = tl.prepro.imresize(img, (224, 224)).astype(np.float32) / 255
  output = vgg(img, is_train=False)

Get a part of CNN
------------------

.. code-block:: python

  # get VGG without the last layer
  cnn = tl.models.vgg16(end_with='fc2_relu', mode='static').as_layer()
  # add one more layer and build a new model
  ni = Input([None, 224, 224, 3], name="inputs")
  nn = cnn(ni)
  nn = tl.layers.Dense(n_units=100, name='out')(nn)
  model = tl.models.Model(inputs=ni, outputs=nn)
  # train your own classifier (only update the last layer)
  train_params = model.get_layer('out').weights

Reuse CNN
------------------

.. code-block:: python

  # in dynamic model, we can directly use the same model
  # in static model
  vgg_layer = tl.models.vgg16().as_layer()
  ni_1 = tl.layers.Input([None, 224, 224, 3])
  ni_2 = tl.layers.Input([None, 224, 224, 3])
  a_1 = vgg_layer(ni_1)
  a_2 = vgg_layer(ni_2)
  M = Model(inputs=[ni_1, ni_2], outputs=[a_1, a_2])

