Welcome to TensorLayer
=======================================


.. image:: user/my_figs/img_tensorlayer.png
  :scale: 25 %
  :align: center
  :target: https://github.com/zsdonghao/tensorlayer

`TensorLayer`_ is a deep learning and reinforcement learning library for researchers and practitioners. It is an extension library for `Google TensorFlow <https://www.tensorflow.org>`_. It providers high-level APIs and pre-built training blocks that can largely simplify the development of complex learning models. TensorLayer is easy to be extended and customized for your needs. In addition, we provide a rich set of examples and tutorials to help you to build up your own deep learning and reinforcement learning algorithms.

The `documentation <http://tensorlayer.readthedocs.io/en/latest/user/tutorial.html>`_ describes the usages of TensorLayer APIs. It is also a self-contained document that includes a tutorial to walk through different types of neural networks, deep reinforcement learning and Natural Language Processing (NLP) etc. We have included the corresponding modularized implementations of Google TensorFlow Deep Learning tutorial, so you could read TensorFlow tutorial as the same time  as the same time
`[en] <https://www.tensorflow.org/versions/master/tutorials/index.html>`_ `[cn] <http://wiki.jikexueyuan.com/project/tensorflow-zh/>`_ .


.. _TensorLayer-philosopy:

Design goals
----------

`TensorLayer`_ grow out from a need to combine the power of TensorFlow with the availability of the right building blocks for training neural networks. Its development is guided by a number of design goals:

 * **Transparency**: Developing advanced learning algorithms requires low-level tunning of the underlying training engine. TensorLayer exposes the implementation details of the TensorFlow in a structured way, and allows users to do low-level engine manupulations, such as the configurations of training process, iteration, initialization as well as the access to Tensor components and TPUs.
 * **Extensibility**: Be easy to use, extend and modify, to facilitate use in research and practition activities. A network is abstracted to regularization, cost and outputs of each layer. Other wraping libraries for TensorFlow are easy to be merged into TensorLayer, suitable for researchers.
 * **Performance**: The running speed under GPU support is the same with TensorFlow. TensorLayer can also run in a distributed  mode.
 * **Low learning curve**: To facilitate bootstrapping, we provide mass format-consistent examples covering Dropout, DropConnect, Denoising Autoencoder, LSTM, CNN etc, speed up your development.

.. note::
   If you got problem to read the docs online, you could download the repository
   on `GitHub`_, then go to ``/docs/_build/html/index.html`` to read the docs
   offline. The ``_build`` folder can be generated in ``docs`` using ``make html``.



User Guide
------------

The TensorLayer user guide explains how to install TensorFlow, CUDA and cuDNN,
how to build and train neural networks using TensorLayer, and how to contribute
to the library as a developer.

.. toctree::
  :maxdepth: 2

  user/installation
  user/tutorial
  user/example
  user/development
  user/more

API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
  :maxdepth: 2

  modules/layers
  modules/activation
  modules/nlp
  modules/rein
  modules/iterate
  modules/cost
  modules/visualize
  modules/files
  modules/utils
  modules/preprocess
  modules/ops


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _GitHub: https://github.com/zsdonghao/tensorlayer
.. _TensorLayer: https://github.com/zsdonghao/tensorlayer/
