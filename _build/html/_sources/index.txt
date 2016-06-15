.. tensorlayer documentation master file, created by
   sphinx-quickstart on Wed Jun 15 16:10:29 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TensorLayer
=======================================

Author: Hao Dong

  network = tl.InputLayer(x, name='input_layer')
  network = tl.DropoutLayer(network, keep=0.8, name='drop1')
  network = tl.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
  network = tl.DropoutLayer(network, keep=0.5, name='drop2')
  network = tl.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')
  network = tl.DropoutLayer(network, keep=0.5, name='drop3')
  network = tl.DenseLayer(network, n_units=10, act = identity, name='output_layer')


Contents:

.. toctree::
   :maxdepth: 2
   LICENSE




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
