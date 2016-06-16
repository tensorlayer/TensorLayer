.. _installation:

============
Installation
============

Lasagne has a couple of prerequisites that need to be installed first, but it
is not very picky about versions. The single exception is Theano: Due to its
tight coupling to Theano, you will have to install a recent version of Theano
(usually more recent than the latest official release!) fitting the version of
Lasagne you choose to install.

Most of the instructions below assume you are running a Linux or Mac system,
but are otherwise very generic. For detailed step-by-step instructions for
specific platforms including Windows, check our `From Zero to Lasagne
<https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne>`_ guides.

If you run into any trouble, please check the `Theano installation instructions
<http://deeplearning.net/software/theano/install.html>`_ which cover installing
the prerequisites for a range of operating systems, or ask for help on `our
mailing list <https://groups.google.com/d/forum/lasagne-users>`_.


Prerequisites
=============

Python + pip
------------

Lasagne currently requires Python 2.7 or 3.4 to run. Please install Python via
the package manager of your operating system if it is not included already.

Python includes ``pip`` for installing additional modules that are not shipped
with your operating system, or shipped in an old version, and we will make use
of it below. We recommend installing these modules into your home directory
via ``--user``, or into a `virtual environment
<http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/>`_
via ``virtualenv``.

C compiler
----------

Theano requires a working C compiler, and numpy/scipy require a compiler as
well if you install them via ``pip``. On Linux, the default compiler is usually
``gcc``, and on Mac OS, it's ``clang``. Again, please install them via the
package manager of your operating system.

numpy/scipy + BLAS
------------------

Lasagne requires numpy of version 1.6.2 or above, and Theano also requires
scipy 0.11 or above. Numpy/scipy rely on a BLAS library to provide fast linear
algebra routines. They will work fine without one, but a lot slower, so it is
worth getting this right (but this is less important if you plan to use a GPU).

If you install numpy and scipy via your operating system's package manager,
they should link to the BLAS library installed in your system. If you install
numpy and scipy via ``pip install numpy`` and ``pip install scipy``, make sure
to have development headers for your BLAS library installed (e.g., the
``libopenblas-dev`` package on Debian/Ubuntu) while running the installation
command. Please refer to the `numpy/scipy build instructions
<http://www.scipy.org/scipylib/building/index.html>`_ if in doubt.

Theano
------

The version to install depends on the Lasagne version you choose, so this will
be handled below.


Stable Lasagne release
======================

Lasagne 0.1 requires a more recent version of Theano than the one available
on PyPI. To install a version that is known to work, run the following command:

.. code-block:: bash

  pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt

.. warning::
  An even more recent version of Theano will often work as well, but at the
  time of writing, a simple ``pip install Theano`` will give you a version that
  is too old.

To install release 0.1 of Lasagne from PyPI, run the following command:

.. code-block:: bash

  pip install Lasagne==0.1

If you do not use ``virtualenv``, add ``--user`` to both commands to install
into your home directory instead. To upgrade from an earlier installation, add
``--upgrade``.


Bleeding-edge version
=====================

The latest development version of Lasagne usually works fine with the latest
development version of Theano. To install both, run the following commands:

.. code-block:: bash

  pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
  pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

Again, add ``--user`` if you want to install to your home directory instead.


.. _lasagne-development-install:

Development installation
========================

Alternatively, you can install Lasagne (and optionally Theano) from source,
in a way that any changes to your local copy of the source tree take effect
without requiring a reinstall. This is often referred to as *editable* or
*development* mode. Firstly, you will need to obtain a copy of the source tree:

.. code-block:: bash

  git clone https://github.com/Lasagne/Lasagne.git

It will be cloned to a subdirectory called ``Lasagne``. Make sure to place it
in some permanent location, as for an *editable* installation, Python will
import the module directly from this directory and not copy over the files.
Enter the directory and install the known good version of Theano:

.. code-block:: bash

  cd Lasagne
  pip install -r requirements.txt

Alternatively, install the bleeding-edge version of Theano as described in the
previous section.

To install the Lasagne package itself, in editable mode, run:

.. code-block:: bash

  pip install --editable .

As always, add ``--user`` to install it to your home directory instead.

**Optional**: If you plan to contribute to Lasagne, you will need to fork the
Lasagne repository on GitHub. This will create a repository under your user
account. Update your local clone to refer to the official repository as
``upstream``, and your personal fork as ``origin``:

.. code-block:: bash

  git remote rename origin upstream
  git remote add origin https://github.com/<your-github-name>/Lasagne.git

If you set up an `SSH key <https://help.github.com/categories/ssh/>`_, use the
SSH clone URL instead: ``git@github.com:<your-github-name>/Lasagne.git``.

You can now use this installation to develop features and send us pull requests
on GitHub, see :doc:`development`!


GPU support
===========

Thanks to Theano, Lasagne transparently supports training your networks on a
GPU, which may be 10 to 50 times faster than training them on a CPU. Currently,
this requires an NVIDIA GPU with CUDA support, and some additional software for
Theano to use it.

CUDA
----

Install the latest CUDA Toolkit and possibly the corresponding driver available
from NVIDIA: https://developer.nvidia.com/cuda-downloads

Closely follow the *Getting Started Guide* linked underneath the download table
to be sure you don't mess up your system by installing conflicting drivers.

After installation, make sure ``/usr/local/cuda/bin`` is in your ``PATH``, so
``nvcc --version`` works. Also make sure ``/usr/local/cuda/lib64`` is in your
``LD_LIBRARY_PATH``, so the toolkit libraries can be found.

Theano
------

If CUDA is set up correctly, the following should print some information on
your GPU (the first CUDA-capable GPU in your system if you have multiple ones):

.. code-block:: bash

  THEANO_FLAGS=device=gpu python -c "import theano; print(theano.sandbox.cuda.device_properties(0))"

To configure Theano to use the GPU by default, create a file ``.theanorc``
directly in your home directory, with the following contents:

.. code-block:: none

  [global]
  floatX = float32
  device = gpu

Optionally add ``allow_gc = False`` for some extra performance at the expense
of (sometimes substantially) higher GPU memory usage.

If you run into problems, please check Theano's instructions for `Using the GPU
<http://deeplearning.net/software/theano/tutorial/using_gpu.html>`_.

cuDNN
-----

NVIDIA provides a library for common neural network operations that especially
speeds up Convolutional Neural Networks (CNNs). Again, it can be obtained from
NVIDIA (after registering as a developer): https://developer.nvidia.com/cudnn

Note that it requires a reasonably modern GPU with Compute Capability 3.0 or higher;
see `NVIDIA's list of CUDA GPUs <https://developer.nvidia.com/cuda-gpus>`_.

To install it, copy the ``*.h`` files to ``/usr/local/cuda/include`` and the
``lib*`` files to ``/usr/local/cuda/lib64``.

To check whether it is found by Theano, run the following command:

.. code-block:: bash

  python -c "from theano.sandbox.cuda.dnn import dnn_available as d; print(d() or d.msg)"

It will print ``True`` if everything is fine, or an error message otherwise.
There are no additional steps required for Theano to make use of cuDNN.

Docker
======

Instead of manually installing Theano and Lasagne on your machines as described above,
you may want to use a pre-made `Docker <https://www.docker.com/what-docker>`_
image: `Lasagne Docker (CPU) <https://hub.docker.com/r/kaixhin/lasagne/>`_ or
`Lasagne Docker (CUDA) <https://hub.docker.com/r/kaixhin/cuda-lasagne/>`_. These
are updated on a weekly basis with bleeding-edge builds of Theano and Lasagne.
Examples of running bash in a Docker container are as follows:

.. code-block:: bash

  sudo docker run -it kaixhin/lasagne
  sudo nvidia-docker run -it kaixhin/cuda-lasagne:7.0

For a guide to Docker, see the `official docs <https://docs.docker.com>`_.
CUDA support requires `NVIDIA Docker <https://github.com/NVIDIA/nvidia-docker>`_.
For more details on how to use the Lasagne Docker images,
consult the `source project <https://github.com/Kaixhin/dockerfiles>`_.
