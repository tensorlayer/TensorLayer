.. _installation:

============
Installation
============

TensorLayer has some prerequisites that need to be installed first, including
`TensorFlow`_ , numpy and matplotlib. For GPU
support CUDA and cuDNN are required.

If you run into any trouble, please check the `TensorFlow installation
instructions <https://www.tensorflow.org/versions/master/get_started/os_setup.html>`_
which cover installing the TensorFlow for a range of operating systems including
Mac OX, Linux and Windows, or ask for help on `tensorlayer@gmail.com <tensorlayer@gmail.com>`_
or `FAQ <http://tensorlayer.readthedocs.io/en/latest/user/more.html>`_.


Install TensorFlow
=========================

.. code-block:: bash

  pip3 install tensorflow-gpu==2.0.0-beta1 # specific version  (YOU SHOULD INSTALL THIS ONE NOW)
  pip3 install tensorflow-gpu # GPU version
  pip3 install tensorflow # CPU version

The installation instructions of TensorFlow are written to be very detailed on `TensorFlow`_  website.
However, there are something need to be considered. For example, `TensorFlow`_ officially supports GPU acceleration for Linux, Mac OX and Windows at present. For ARM processor architecture, you need to install TensorFlow from source.

Install TensorLayer
=========================

For stable version:

.. code-block:: bash

  pip3 install tensorlayer
  
  pip install tensorlayer -i https://pypi.tuna.tsinghua.edu.cn/simple  (faster in China)

For latest version, please install from Github.

.. code-block:: bash

  pip3 install git+https://github.com/tensorlayer/tensorlayer.git
  or
  pip3 install https://github.com/tensorlayer/tensorlayer/archive/master.zip

For developers, you should clone the folder to your local machine and put it along with your project scripts.

.. code-block:: bash

  git clone https://github.com/tensorlayer/tensorlayer.git


Alternatively, you can build from the source.

.. code-block:: bash

  # First clone the repository and change the current directory to the newly cloned repository
  git clone https://github.com/tensorlayer/tensorlayer.git
  cd tensorlayer

  # Install virtualenv if necessary
  sudo pip3 install virtualenv
  # Then create a virtualenv called `venv`
  virtualenv venv

  # Activate the virtualenv

  ## Linux:
  source venv/bin/activate

  ## Windows:
  venv\Scripts\activate.bat

  # basic installation
  pip3 install .

  # ============= IF TENSORFLOW IS NOT ALREADY INSTALLED ============= #

  # for a machine **without** an NVIDIA GPU
  pip3 install -e ".[all_cpu_dev]"

  # for a machine **with** an NVIDIA GPU
  pip3 install -e ".[all_gpu_dev]"

If you want install TensorLayer 1.X, the simplest way to install TensorLayer 1.X is as follow. It will also install the numpy and matplotlib automatically.

.. code-block:: bash

  [stable version] pip3 install tensorlayer==1.x.x

However, if you want to modify or extend TensorLayer 1.X, you can download the repository from
`Github`_ and install it as follow.

.. code-block:: bash

  cd to the root of the git tree
  pip3 install -e .

This command will run the ``setup.py`` to install TensorLayer. The ``-e`` reflects
editable, then you can edit the source code in ``tensorlayer`` folder, and ``import`` the edited
TensorLayer.


GPU support
==========================

Thanks to NVIDIA supports, training a fully connected network on a
GPU, which may be 10 to 20 times faster than training them on a CPU.
For convolutional network, may have 50 times faster.
This requires an NVIDIA GPU with CUDA and cuDNN support.


CUDA
----

The TensorFlow website also teach how to install the CUDA and cuDNN, please see
`TensorFlow GPU Support <https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux>`_.

Download and install the latest CUDA is available from NVIDIA website:

 - `CUDA download and install <https://developer.nvidia.com/cuda-downloads>`_


..
  After installation, make sure ``/usr/local/cuda/bin`` is in your ``PATH`` (use ``echo #PATH`` to check), and
  ``nvcc --version`` works. Also ensure ``/usr/local/cuda/lib64`` is in your
  ``LD_LIBRARY_PATH``, so the CUDA libraries can be found.

If CUDA is set up correctly, the following command should print some GPU information on
the terminal:

.. code-block:: bash

  python -c "import tensorflow"


cuDNN
--------

Apart from CUDA, NVIDIA also provides a library for common neural network operations that especially
speeds up Convolutional Neural Networks (CNNs). Again, it can be obtained from
NVIDIA after registering as a developer (it take a while):

Download and install the latest cuDNN is available from NVIDIA website:

 - `cuDNN download and install <https://developer.nvidia.com/cudnn>`_


To install it, copy the ``*.h`` files to ``/usr/local/cuda/include`` and the
``lib*`` files to ``/usr/local/cuda/lib64``.

.. _TensorFlow: https://www.tensorflow.org/versions/master/get_started/os_setup.html
.. _GitHub: https://github.com/tensorlayer/tensorlayer
.. _TensorLayer: https://github.com/tensorlayer/tensorlayer/



Windows User
==============

TensorLayer is built on the top of Python-version TensorFlow, so please install Python first.
Note：We highly recommend installing Anaconda. The lowest version requirements of Python is py35.

`Anaconda download <https://www.continuum.io/downloads>`_

GPU support
------------
Thanks to NVIDIA supports, training a fully connected network on a GPU, which may be 10 to 20 times faster than training them on a CPU. For convolutional network, may have 50 times faster. This requires an NVIDIA GPU with CUDA and cuDNN support.

1. Installing Microsoft Visual Studio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You should preinstall Microsoft Visual Studio (VS) before installing CUDA. The lowest version requirements is VS2010. We recommend installing VS2015 or VS2013. CUDA7.5 supports VS2010, VS2012 and VS2013. CUDA8.0 also supports VS2015.

2. Installing CUDA
^^^^^^^^^^^^^^^^^^^^^^^
Download and install the latest CUDA is available from NVIDIA website:

`CUDA download <https://developer.nvidia.com/CUDA-downloads>`_

We do not recommend modifying the default installation directory.

3. Installing cuDNN
^^^^^^^^^^^^^^^^^^^^^^
The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. Download and extract the latest cuDNN is available from NVIDIA website:

`cuDNN download <https://developer.nvidia.com/cuDNN>`_

After extracting cuDNN, you will get three folders (bin, lib, include). Then these folders should be copied to CUDA installation. (The default installation directory is `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0`)

Installing TensorLayer
------------------------
For TensorLayer, please refer to the steps mentioned above.

.. code-block:: bash

  pip3 install tensorflow        #CPU version
  pip3 install tensorflow-gpu    #GPU version (GPU version and CPU version just choose one)
  pip3 install tensorlayer       #Install tensorlayer



Issue
=======

If you get the following output when import tensorlayer, please read `FQA <http://tensorlayer.readthedocs.io/en/latest/user/more.html>`_.

.. code-block:: bash

  _tkinter.TclError: no display name and no $DISPLAY environment variable
