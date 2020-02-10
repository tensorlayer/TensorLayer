Welcome to TensorLayer
=======================================


.. image:: user/my_figs/tl_transparent_logo.png
  :width: 30 %
  :align: center
  :target: https://github.com/tensorlayer/tensorlayer

**Documentation Version:** |release|

**Jun 2019** `Deep Reinforcement Learning Model ZOO Release !! <https://github.com/tensorlayer/tensorlayer/tree/master/examples/reinforcement_learning>`__.

**Good News:** We won the **Best Open Source Software Award** `@ACM Multimedia (MM) 2017 <http://www.acmmm.org/2017/mm-2017-awardees/>`_.

`TensorLayer`_ is a Deep Learning (DL) and Reinforcement Learning (RL) library extended from `Google TensorFlow <https://www.tensorflow.org>`_.  It provides popular DL and RL modules that can be easily customized and assembled for tackling real-world machine learning problems.
More details can be found `here <https://github.com/tensorlayer/tensorlayer>`_.

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
  user/examples
  user/contributing
  user/get_involved
  user/faq

.. toctree::
  :maxdepth: 2
  :caption: Getting started

  user/get_start_model
  user/get_start_advance

API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
  :maxdepth: 2
  :caption: Stable Functionalities

  modules/activation
  modules/array_ops
  modules/cost
  modules/prepro
  modules/files
  modules/iterate
  modules/layers
  modules/models
  modules/nlp
  modules/initializers
  modules/rein
  modules/utils
  modules/visualize

.. toctree::
  :maxdepth: 2
  :caption: Alpha Version Functionalities

  modules/db
  modules/optimizers
  modules/distributed
  
Command-line Reference
----------------------

TensorLayer provides a handy command-line tool `tl` to perform some common tasks.

.. toctree::
  :maxdepth: 2
  :caption: Command Line Interface

  modules/cli


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _GitHub: https://github.com/tensorlayer/tensorlayer
.. _TensorLayer: https://github.com/tensorlayer/tensorlayer/
