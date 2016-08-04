Welcome to TensorLayer
=======================================


.. image:: user/my_figs/img_tensorlayer.png
  :scale: 25 %
  :align: center
  :target: https://github.com/zsdonghao/tensorlayer

`TensorLayer`_ was designed for both Researchers
and Engineers, it is a transparent library built on the top of Google TensorFlow.
It was designed to provide a higher-level
API to TensorFlow in order to speed-up experimentations and developments.
`TensorLayer`_ is easy to extended and modified.
In addition, we provides mass examples and tutorials
to help you to find the one you need in your project.

The `documentation <http://tensorlayer.readthedocs.io/en/latest/user/tutorial.html>`_
is not only for describing how to use `TensorLayer`_ but also
a tutorial to walk through different type of Neural Networks, Deep
Reinforcement Learning and Natural Language Processing etc.

However, different with other inflexible TensorFlow wrappers,
`TensorLayer`_ assumes that you are somewhat familiar with Neural Networks and TensorFlow.
A basic understanding of how TensorFlow works is required to be
able to use `TensorLayer`_ skillfully.


.. _TensorLayer-philosopy:

Philosophy
----------

`TensorLayer`_ grew out of a need to combine the flexibility of TensorFlow with the
availability of the right building blocks for training neural networks.
Its development is guided by a number of design goals:


* **Transparency**: Do not hide TensorFlow behind abstractions. Try to rely on
  TensorFlow's functionality where possible, and follow TensorFlow's conventions.
  Do not hide training process, all iteration, initialization can be managed
  by user.

* **Tensor**: Neural networks perform on multidimensional data arrays which are
  referred to as "tensors".

* **TPU**: Tensor Processing Unit is a custom ASIC built specifically for
  machine learning and tailored for TensorFlow.

* **Distribution**: Distributed Machine Learning is the default function of TensorFlow.

* **Compatibility**: A network is abstracted to regularization, cost and outputs
  of each layer. Easy to work with other TensorFlow libraries.

* **Simplicity**: Be easy to use, easy to extend and modify, to facilitate use
  in Research and Engineering.

* **High-Speed**: The running speed under GPU support is the same with
  TensorFlow only. The simplicity do not sacrifice the performance.



.. note::
   If you got problem to read the docs online, you could download the project
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
