Welcome to TensorLayer
=======================================


.. image:: user/my_figs/img_tensorlayer.png
  :scale: 25 %
  :align: center
  :target: https://github.com/zsdonghao/tensorlayer

`TensorLayer`_ is a Deep Learning (DL) and Reinforcement Learning (RL) library extended from `Google TensorFlow <https://www.tensorflow.org>`_.  It provides popular DL and RL modules that can be easily customized and assembled for tackling real-world machine learning problems.

.. _TensorLayer-philosopy:

Design Philosophy
----------
 

`TensorLayer`_ grow out from a need to combine the power of TensorFlow with the right building modules for deep neural networks. According to our years of research and pratical experiences of tackling real-world machine learning problems, we come up with three design goals for TensorLayer:

* **Simplicity**: we make TensorLayer easy to work with by providing mass tutorials that can be deployed and run through in minutes. A TensorFlow user may find it easier to bootstrap with the simple, high-level APIs provided by TensorLayer, and then deep dive into their implementation details if need. 
* **Flexibility**: developing an effective DL algorithm for a specific domain typically requires careful tunings from many aspects. Without the loss of simplicity, TensorLayer allows users to customize their modules by manipulating the native APIs of TensorFlow (e.g., training parameters, iteration control and tensor components).
* **Performance**: TensorLayer aims to provide zero-cost abstraction for TensorFlow. With its first-class support for TensorFlow, it can easily run on either heterogeneous platforms or multiple computation nodes without compromise in performance.

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
  modules/cost
  modules/iterate
  modules/utils
  modules/nlp
  modules/rein
  modules/files
  modules/visualize
  modules/preprocess
  modules/ops
  modules/activation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _GitHub: https://github.com/zsdonghao/tensorlayer
.. _TensorLayer: https://github.com/zsdonghao/tensorlayer/
