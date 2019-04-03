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
  img = tl.prepro.imresize(tl.vis.read_image('img.jpeg'), (224, 224)).astype(np.float32)
  mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape([1, 1, 1, 3])
  img = img - mean
  vgg.eval()
  output = vgg(img1)
  probs = tf.nn.softmax(output)[0].numpy()
  preds = (np.argsort(probs)[::-1])[0:5]
  for p in preds:
      print(class_names[p], probs[p])

Get a part of CNN
------------------

.. code-block:: python

  import tensorflow as tf
  import tensorlayer as tl

  # get VGG without the last layer
  vgg = tl.models.vgg.vgg16(end_with='fc2_relu')
  # add one more layer
  net = tl.layers.Dense(n_units=100, name='out')(vgg)
  # restore pre-trained VGG parameters
  vgg.restore_weights()
  # train your own classifier (only update the last layer)
  train_params = tl.layers.get_variables_with_name('out')

Reuse CNN
------------------

.. code-block:: python

  # in dynamic mode, we can directly use the same model
  # in static mode
  vgg_layer = tl.models.vgg.vgg16.as_layer()
  ni_1 = tl.layers.Input([None, 224, 244, 3])
  ni_2 = tl.layers.Input([None, 224, 244, 3])
  a_1 = vgg_layer(ni_1)
  a_2 = vgg_layer(ni_2)
  M = Model(inputs=[ni_1, ni_2], outputs=[a_1, a_2])
