<a href="https://tensorlayer.readthedocs.io/">
    <div align="center">
        <img src="img/tl_transparent_logo.png" width="50%" height="30%"/>
    </div>
</a>

<!--- [![PyPI Version](https://badge.fury.io/py/tensorlayer.svg)](https://badge.fury.io/py/tensorlayer) --->
<!--- ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorlayer.svg)) --->

![GitHub last commit (branch)](https://img.shields.io/github/last-commit/tensorlayer/tensorlayer/master.svg)
[![Supported TF Version](https://img.shields.io/badge/TensorFlow-2.0.0%2B-brightgreen.svg)](https://github.com/tensorflow/tensorflow/releases)
[![Documentation Status](https://readthedocs.org/projects/tensorlayer/badge/)](https://tensorlayer.readthedocs.io/)
[![Build Status](https://travis-ci.org/tensorlayer/tensorlayer.svg?branch=master)](https://travis-ci.org/tensorlayer/tensorlayer)
[![Downloads](http://pepy.tech/badge/tensorlayer)](http://pepy.tech/project/tensorlayer)
[![Docker Pulls](https://img.shields.io/docker/pulls/tensorlayer/tensorlayer.svg)](https://hub.docker.com/r/tensorlayer/tensorlayer/)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d6b118784e25435498e7310745adb848)](https://www.codacy.com/app/tensorlayer/tensorlayer)

<!---  [![CircleCI](https://circleci.com/gh/tensorlayer/tensorlayer/tree/master.svg?style=svg)](https://circleci.com/gh/tensorlayer/tensorlayer/tree/master) --->

<!---  [![Documentation Status](https://readthedocs.org/projects/tensorlayercn/badge/)](https://tensorlayercn.readthedocs.io/)
<!---  [![PyUP Updates](https://pyup.io/repos/github/tensorlayer/tensorlayer/shield.svg)](https://pyup.io/repos/github/tensorlayer/tensorlayer/) --->

<br/>

<a href="https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc" target="\_blank">
	<div align="center">
		<img src="img/join_slack.png" width="40%"/>
	</div>
</a>

<br/>

TensorLayer is a novel TensorFlow-based deep learning and reinforcement learning library designed for researchers and engineers. It provides a large collection of customizable neural layers / functions that are key to build real-world AI applications. TensorLayer is awarded the 2017 Best Open Source Software by the [ACM Multimedia Society](https://twitter.com/ImperialDSI/status/923928895325442049).

🔥📰🔥 Reinforcement Learning Model Zoos: [Low-level APIs for Research](https://github.com/tensorlayer/tensorlayer/tree/master/examples/reinforcement_learning) and [High-level APIs for Production](https://github.com/tensorlayer/RLzoo)

🔥📰🔥 [Sipeed Maxi-EMC](https://github.com/sipeed/Maix-EMC): Run TensorLayer models on the **low-cost AI chip** (e.g., K210) (Alpha Version)

🔥📰🔥 [NNoM](https://github.com/majianjia/nnom): Run TensorLayer quantized models on the **MCU** (e.g., STM32) (Coming Soon)

🔥📰🔥 [Free GPU and Data Storage from SurgicalAI](https://github.com/fangde/FreeGPU): SurgicalAI is sponsoring the TensorLayer Community with Cloud Computing Resources such as Free GPUs and Data Storage.

# Features

As deep learning practitioners, we have been looking for a library that can address various development
purposes. This library is easy to adopt by providing diverse examples, tutorials and pre-trained models.
Also, it allow users to easily fine-tune TensorFlow; while being suitable for production deployment. TensorLayer aims to satisfy all these purposes. It has three key features:

- **_Simplicity_** : TensorLayer lifts the low-level dataflow interface of TensorFlow to _high-level_ layers / models. It is very easy to learn through the rich [example codes](https://github.com/tensorlayer/awesome-tensorlayer) contributed by a wide community.
- **_Flexibility_** : TensorLayer APIs are transparent: it does not mask TensorFlow from users; but leaving massive hooks that help _low-level tuning_ and _deep customization_.
- **_Zero-cost Abstraction_** : TensorLayer can achieve the _full power_ of TensorFlow. The following table shows the training speeds of [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) using TensorLayer and native TensorFlow on a TITAN Xp.

  |   Mode    |       Lib       | Data Format  | Max GPU Memory Usage(MB) | Max CPU Memory Usage(MB) | Avg CPU Memory Usage(MB) | Runtime (sec) |
  | :-------: | :-------------: | :----------: | :----------------------: | :----------------------: | :----------------------: | :-----------: |
  | AutoGraph | TensorFlow 2.0  | channel last |          11833           |           2161           |           2136           |      74       |
  |           | Tensorlayer 2.0 | channel last |          11833           |           2187           |           2169           |      76       |
  |   Graph   |      Keras      | channel last |           8677           |           2580           |           2576           |      101      |
  |   Eager   | TensorFlow 2.0  | channel last |           8723           |           2052           |           2024           |      97       |
  |           | TensorLayer 2.0 | channel last |           8723           |           2010           |           2007           |      95       |

TensorLayer stands at a unique spot in the library landscape. Other wrapper libraries like Keras and TFLearn also provide high-level abstractions. They, however, often
hide the underlying engine from users, which make them hard to customize
and fine-tune. On the contrary, TensorLayer APIs are generally lightweight, flexible and transparent.
Users often find it easy to start with the examples and tutorials, and then dive
into TensorFlow seamlessly. In addition, TensorLayer does not create library lock-in through native supports for importing components from Keras.

TensorLayer has a fast growing usage among top researchers and engineers, from universities like Peking University,
Imperial College London, UC Berkeley, Carnegie Mellon University, Stanford University, and
University of Technology of Compiegne (UTC), and companies like Google, Microsoft, Alibaba, Tencent, Xiaomi, and Bloomberg.

# Tutorials and Real-World Applications

You can find a large collection of tutorials, examples and real-world applications using TensorLayer within [examples](examples/) or through the following space:

<a href="https://github.com/tensorlayer/awesome-tensorlayer/blob/master/readme.md" target="\_blank">
	<div align="center">
		<img src="img/awesome-mentioned.png" width="40%"/>
	</div>
</a>

# Documentation

TensorLayer has extensive documentation for both beginners and professionals. The documentation is available in
both English and Chinese. Please click the following icons to find the documents you need:

[![English Documentation](https://img.shields.io/badge/documentation-english-blue.svg)](https://tensorlayer.readthedocs.io/)
[![Chinese Documentation](https://img.shields.io/badge/documentation-%E4%B8%AD%E6%96%87-blue.svg)](https://tensorlayercn.readthedocs.io/)
[![Chinese Book](https://img.shields.io/badge/book-%E4%B8%AD%E6%96%87-blue.svg)](http://www.broadview.com.cn/book/5059/)

If you want to try the experimental features on the the master branch, you can find the latest document
[here](https://tensorlayer.readthedocs.io/en/latest/).

# Install

For latest code for TensorLayer 2.0, please build from the source. TensorLayer 2.0 has pre-requisites including TensorFlow 2, numpy, and others. For GPU support, CUDA and cuDNN are required.

Install TensorFlow:

```bash
pip3 install tensorflow-gpu==2.0.0-beta1 # specific version  (YOU SHOULD INSTALL THIS ONE NOW)
pip3 install tensorflow-gpu # GPU version
pip3 install tensorflow # CPU version
```

Install the stable version of TensorLayer:

```bash
pip3 install tensorlayer
```

Install the latest version of TensorLayer:

```bash
pip3 install git+https://github.com/tensorlayer/tensorlayer.git
or
pip3 install https://github.com/tensorlayer/tensorlayer/archive/master.zip
```

For developers, you should clone the folder to your local machine and put it along with your project scripts.

```bash
git clone https://github.com/tensorlayer/tensorlayer.git
```

If you want install TensorLayer 1.X, the simplest way to install TensorLayer 1.X is to use the **Py**thon **P**ackage **I**ndex (PyPI):

```bash
# for last stable version of TensorLayer 1.X
pip3 install --upgrade tensorlayer==1.X

# for latest release candidate of TensorLayer 1.X
pip3 install --upgrade --pre tensorlayer

# if you want to install the additional dependencies, you can also run
pip3 install --upgrade tensorlayer[all]              # all additional dependencies
pip3 install --upgrade tensorlayer[extra]            # only the `extra` dependencies
pip3 install --upgrade tensorlayer[contrib_loggers]  # only the `contrib_loggers` dependencies
```

<!---
Alternatively, you can install the latest or development version by directly pulling from github:

```bash
pip3 install https://github.com/tensorlayer/tensorlayer/archive/master.zip
# or
# pip3 install https://github.com/tensorlayer/tensorlayer/archive/<branch-name>.zip
```
--->

<!---
## Using Docker

The [TensorLayer containers](https://hub.docker.com/r/tensorlayer/tensorlayer/) are built on top of the official [TensorFlow containers](https://hub.docker.com/r/tensorflow/tensorflow/):

### Containers with CPU support

```bash
# for CPU version and Python 2
docker pull tensorlayer/tensorlayer:latest
docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest

# for CPU version and Python 3
docker pull tensorlayer/tensorlayer:latest-py3
docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-py3
```

### Containers with GPU support

NVIDIA-Docker is required for these containers to work: [Project Link](https://github.com/NVIDIA/nvidia-docker)

```bash
# for GPU version and Python 2
docker pull tensorlayer/tensorlayer:latest-gpu
nvidia-docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu

# for GPU version and Python 3
docker pull tensorlayer/tensorlayer:latest-gpu-py3
nvidia-docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu-py3
```
--->

# Contribute

Please read the [Contributor Guideline](CONTRIBUTING.md) before submitting your PRs.

# Cite

If you use TensorLayer for any projects, please cite this paper：

```
@article{tensorlayer2017,
    author  = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
    journal = {ACM Multimedia},
    title   = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
    url     = {http://tensorlayer.org},
    year    = {2017}
}
```

# License

TensorLayer is released under the Apache 2.0 license. We also host TensorLayer on [iHub](https://code.ihub.org.cn/projects/328) and [Gitee](https://gitee.com/organizations/TensorLayer).
