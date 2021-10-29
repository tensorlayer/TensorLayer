|TENSORLAYER-LOGO|


|Awesome| |Documentation-EN| |Documentation-CN| |Book-CN| |Downloads|

|PyPI| |PyPI-Prerelease| |Commits-Since| |Python| |TensorFlow|

|Travis| |Docker| |RTD-EN| |RTD-CN| |PyUP| |Docker-Pulls| |Code-Quality|


|JOIN-SLACK-LOGO|

TensorLayer is a novel TensorFlow-based deep learning and reinforcement
learning library designed for researchers and engineers. It provides a
large collection of customizable neural layers / functions that are key
to build real-world AI applications. TensorLayer is awarded the 2017
Best Open Source Software by the `ACM Multimedia
Society <http://www.acmmm.org/2017/mm-2017-awardees/>`__.

Design Features
=================

TensorLayer is a new deep learning library designed with simplicity, flexibility and high-performance in mind.

- **Simplicity** : TensorLayer has a high-level layer/model abstraction which is effortless to learn. You can learn how deep learning can benefit your AI tasks in minutes through the massive [examples](https://github.com/tensorlayer/awesome-tensorlayer).
- **Flexibility** : TensorLayer APIs are transparent and flexible, inspired by the emerging PyTorch library. Compared to the Keras abstraction, TensorLayer makes it much easier to build and train complex AI models.
- **Zero-cost Abstraction** : Though simple to use, TensorLayer does not require you to make any compromise in the performance of TensorFlow (Check the following benchmark section for more details).

TensorLayer stands at a unique spot in the TensorFlow wrappers. Other wrappers like Keras and TFLearn
hide many powerful features of TensorFlow and provide little support for writing custom AI models. Inspired by PyTorch, TensorLayer APIs are simple, flexible and Pythonic,
making it easy to learn while being flexible enough to cope with complex AI tasks.
TensorLayer has a fast-growing community. It has been used by researchers and engineers all over the world, including those from  Peking University,
Imperial College London, UC Berkeley, Carnegie Mellon University, Stanford University, and companies like Google, Microsoft, Alibaba, Tencent, Xiaomi, and Bloomberg.

Install
=======

TensorLayer has pre-requisites including TensorFlow, numpy, and others. For GPU support, CUDA and cuDNN are required.
The simplest way to install TensorLayer is to use the Python Package Index (PyPI):

.. code:: bash

    # for last stable version
    pip install --upgrade tensorlayer

    # for latest release candidate
    pip install --upgrade --pre tensorlayer

    # if you want to install the additional dependencies, you can also run
    pip install --upgrade tensorlayer[all]              # all additional dependencies
    pip install --upgrade tensorlayer[extra]            # only the `extra` dependencies
    pip install --upgrade tensorlayer[contrib_loggers]  # only the `contrib_loggers` dependencies

Alternatively, you can install the latest or development version by directly pulling from github:

.. code:: bash

    pip install https://github.com/tensorlayer/tensorlayer/archive/master.zip
    # or
    # pip install https://github.com/tensorlayer/tensorlayer/archive/<branch-name>.zip

Using Docker - a ready-to-use environment
-----------------------------------------

The `TensorLayer
containers <https://hub.docker.com/r/tensorlayer/tensorlayer/>`__ are
built on top of the official `TensorFlow
containers <https://hub.docker.com/r/tensorflow/tensorflow/>`__:

Containers with CPU support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    # for CPU version and Python 2
    docker pull tensorlayer/tensorlayer:latest
    docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest

    # for CPU version and Python 3
    docker pull tensorlayer/tensorlayer:latest-py3
    docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-py3

Containers with GPU support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

NVIDIA-Docker is required for these containers to work: `Project
Link <https://github.com/NVIDIA/nvidia-docker>`__

.. code:: bash

    # for GPU version and Python 2
    docker pull tensorlayer/tensorlayer:latest-gpu
    nvidia-docker run -it --rm -p 8888:88888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu

    # for GPU version and Python 3
    docker pull tensorlayer/tensorlayer:latest-gpu-py3
    nvidia-docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu-py3

Contribute
==========

Please read the `Contributor
Guideline <https://github.com/tensorlayer/tensorlayer/blob/master/CONTRIBUTING.md>`__
before submitting your PRs.

Cite
====

If you find this project useful, we would be grateful if you cite the
TensorLayer papers.

::

    @article{tensorlayer2017,
        author  = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
        journal = {ACM Multimedia},
        title   = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
        url     = {http://tensorlayer.org},
        year    = {2017}
    }

    @inproceedings{tensorlayer2021,
      title={Tensorlayer 3.0: A Deep Learning Library Compatible With Multiple Backends},
      author={Lai, Cheng and Han, Jiarong and Dong, Hao},
      booktitle={2021 IEEE International Conference on Multimedia \& Expo Workshops (ICMEW)},
      pages={1--3},
      year={2021},
      organization={IEEE}

License
=======

TensorLayer is released under the Apache 2.0 license.


.. |TENSORLAYER-LOGO| image:: https://raw.githubusercontent.com/tensorlayer/tensorlayer/master/img/tl_transparent_logo.png
   :target: https://tensorlayer.readthedocs.io/
.. |JOIN-SLACK-LOGO| image:: https://raw.githubusercontent.com/tensorlayer/tensorlayer/master/img/join_slack.png
   :target: https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc

.. |Awesome| image:: https://awesome.re/mentioned-badge.svg
   :target: https://github.com/tensorlayer/awesome-tensorlayer
.. |Documentation-EN| image:: https://img.shields.io/badge/documentation-english-blue.svg
   :target: https://tensorlayer.readthedocs.io/
.. |Documentation-CN| image:: https://img.shields.io/badge/documentation-%E4%B8%AD%E6%96%87-blue.svg
   :target: https://tensorlayercn.readthedocs.io/
.. |Book-CN| image:: https://img.shields.io/badge/book-%E4%B8%AD%E6%96%87-blue.svg
   :target: http://www.broadview.com.cn/book/5059/
.. |Downloads| image:: http://pepy.tech/badge/tensorlayer
   :target: http://pepy.tech/project/tensorlayer


.. |PyPI| image:: http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/github/release/tensorlayer/tensorlayer.svg?label=PyPI%20-%20Release
   :target: https://pypi.org/project/tensorlayer/
.. |PyPI-Prerelease| image:: http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/github/release/tensorlayer/tensorlayer/all.svg?label=PyPI%20-%20Pre-Release
   :target: https://pypi.org/project/tensorlayer/
.. |Commits-Since| image:: http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/github/commits-since/tensorlayer/tensorlayer/latest.svg
   :target: https://github.com/tensorlayer/tensorlayer/compare/1.10.1...master
.. |Python| image:: http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/pypi/pyversions/tensorlayer.svg
   :target: https://pypi.org/project/tensorlayer/
.. |TensorFlow| image:: https://img.shields.io/badge/tensorflow-1.6.0+-blue.svg
   :target: https://github.com/tensorflow/tensorflow/releases

.. |Travis| image:: http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/travis/tensorlayer/tensorlayer/master.svg?label=Travis
   :target: https://travis-ci.org/tensorlayer/tensorlayer
.. |Docker| image:: http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/circleci/project/github/tensorlayer/tensorlayer/master.svg?label=Docker%20Build
   :target: https://circleci.com/gh/tensorlayer/tensorlayer/tree/master
.. |RTD-EN| image:: http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/readthedocs/tensorlayer/latest.svg?label=ReadTheDocs-EN
   :target: https://tensorlayer.readthedocs.io/
.. |RTD-CN| image:: http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/readthedocs/tensorlayercn/latest.svg?label=ReadTheDocs-CN
   :target: https://tensorlayercn.readthedocs.io/
.. |PyUP| image:: https://pyup.io/repos/github/tensorlayer/tensorlayer/shield.svg
   :target: https://pyup.io/repos/github/tensorlayer/tensorlayer/
.. |Docker-Pulls| image:: http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/docker/pulls/tensorlayer/tensorlayer.svg
   :target: https://hub.docker.com/r/tensorlayer/tensorlayer/
.. |Code-Quality| image:: https://api.codacy.com/project/badge/Grade/d6b118784e25435498e7310745adb848
   :target: https://www.codacy.com/app/tensorlayer/tensorlayer
