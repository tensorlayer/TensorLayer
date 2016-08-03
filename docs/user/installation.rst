.. _installation:

============
Installation
============

TuneLayer has some prerequisites that need to be installed first, including
`TensorFlow <https://www.tensorflow.org>`_, numpy and matplotlib. For GPU
support CUDA and cuDNN are required.

If you run into any trouble, please check the `TensorFlow installation
instructions <https://www.tensorflow.org/versions/master/get_started/os_setup.html>`_
which cover installing the TensorFlow for a range of operating systems including
Mac OX and Linux, or ask for help on `hao.dong11@imperial.ac.uk <hao.dong11@imperial.ac.uk>`_.



Prerequisites
=============

Python + pip
-------------

TuneLayer is build on the top of Python-version TensorFlow. Please install
Python first.

.. note::
  We highly recommend python3 instead of python2 for the sake of future.

Python includes ``pip`` command for installing additional modules is recommended.
Besides, a `virtual environment
<http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/>`_
via ``virtualenv`` can help you to manage your python packages.

Take Python3 for example, to install Python includes ``pip``, run the following commands:

.. code-block:: bash

  sudo apt-get install python3
  sudo apt-get install python3-pip
  sudo pip3 install virtualenv

To build a virtual environment and install matplotlib and numpy into it, run the following commands:

.. code-block:: bash

  virtualenv env
  env/bin/pip install matplotlib
  env/bin/pip install numpy

Check the installed packages, run the following command:

.. code-block:: bash

  env/bin/pip list


TensorFlow
------------

The installation instructions of TensorFlow are written to be very detailed on TensorFlow website.
However, there are something need to be considered.

TensorFlow release
====================

TensorFlow
-----------

`TensorFlow <https://www.tensorflow.org/versions/master/get_started/os_setup.html>`_ only officially
supports GPU acceleration for Linux at present.
If you want to use GPU with Mac OX, you need to compile TensorFlow from source.

.. warning::
  For ARM processor architecture, you also need to install TensorFlow from source.

TuneLayer
---------

Hao Dong highly recommend you to install TuneLayer as follow.

.. code-block:: bash

  cd to the root of the git tree
  pip3 install . -e

This command will run the ``setup.py`` to install TuneLayer. The ``-e`` allows
you to edit the scripts in ``tunelayer`` folder, this help you to extend and modify
TuneLayer easily.


GPU support
===========

Thanks to NVIDIA supports, training a fully connected network on a
GPU, which may be 10 to 20 times faster than training them on a CPU.
For convolutional network, may have 50 times faster. This requires an NVIDIA GPU with CUDA and cuDNN support.

CUDA
----

The TensorFlow website also teach how to install the CUDA and cuDNN, please click:
`TensorFlow: CUDA install <https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux>`_.

Install the latest CUDA and cuDNN available from NVIDIA:

`CUDA install <https://developer.nvidia.com/cuda-downloads>`_

`cuDNN install <https://developer.nvidia.com/cuda-downloads>`_

After installation, make sure ``/usr/local/cuda/bin`` is in your ``PATH`` (use ``echo #PATH`` to check), and
``nvcc --version`` works. Also ensure ``/usr/local/cuda/lib64`` is in your
``LD_LIBRARY_PATH``, so the CUDA libraries can be found.

If CUDA is set up correctly, the following command should print some GPU information on
the terminal:

.. code-block:: bash

  python -c "import tensorflow"


cuDNN
-----

NVIDIA provides a library for common neural network operations that especially
speeds up Convolutional Neural Networks (CNNs). Again, it can be obtained from
NVIDIA after registering as a developer (it take a while):
`cuDNN install <https://developer.nvidia.com/cuda-downloads>`_

To install it, copy the ``*.h`` files to ``/usr/local/cuda/include`` and the
``lib*`` files to ``/usr/local/cuda/lib64``.
