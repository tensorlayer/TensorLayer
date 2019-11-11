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

[TensorLayer](https://tensorlayer.readthedocs.io) is a novel TensorFlow-based deep learning and reinforcement learning library designed for researchers and engineers. It provides an extensive collection of customizable neural layers to build complex AI models. TensorLayer is awarded the 2017 Best Open Source Software by the [ACM Multimedia Society](https://twitter.com/ImperialDSI/status/923928895325442049). 
TensorLayer can also be found at [iHub](https://code.ihub.org.cn/projects/328) and [Gitee](https://gitee.com/organizations/TensorLayer).

# News

🔥 Reinforcement Learning Model Zoo: [Low-level APIs for Research](https://github.com/tensorlayer/tensorlayer/tree/master/examples/reinforcement_learning) and [High-level APIs for Production](https://github.com/tensorlayer/RLzoo)

🔥 [Sipeed Maxi-EMC](https://github.com/sipeed/Maix-EMC): Run TensorLayer models on the **low-cost AI chip** (e.g., K210) (Alpha Version)

<!-- 🔥 [NNoM](https://github.com/majianjia/nnom): Run TensorLayer quantized models on the **MCU** (e.g., STM32) (Coming Soon) -->

🔥 [Free GPU and storage resources](https://github.com/fangde/FreeGPU): TensorLayer users can access to free GPU and storage resources donated by SurgicalAI. Thank you SurgicalAI!

# Design Features

TensorLayer is a new deep learning library designed with simplicity, flexibility and high-performance in mind.

- ***Simplicity*** : TensorLayer has a high-level layer/model abstraction which is effortless to learn. You can learn how deep learning can benefit your AI tasks in minutes through the massive [examples](https://github.com/tensorlayer/awesome-tensorlayer).
- ***Flexibility*** : TensorLayer APIs are transparent and flexible, inspired by the emerging PyTorch library. Compared to the Keras abstraction, TensorLayer makes it much easier to build and train complex AI models.
- ***Zero-cost Abstraction*** : Though simple to use, TensorLayer does not require you to make any compromise in the performance of TensorFlow (Check the following benchmark section for more details).

TensorLayer stands at a unique spot in the TensorFlow wrappers. Other wrappers like Keras and TFLearn
hide many powerful features of TensorFlow and provide little support for writing custom AI models. Inspired by PyTorch, TensorLayer APIs are simple, flexible and Pythonic,
making it easy to learn while being flexible enough to cope with complex AI tasks.
TensorLayer has a fast-growing community. It has been used by researchers and engineers all over the world, including those from  Peking University,
Imperial College London, UC Berkeley, Carnegie Mellon University, Stanford University, and companies like Google, Microsoft, Alibaba, Tencent, Xiaomi, and Bloomberg.

# Multilingual Documents

TensorLayer has extensive documentation for both beginners and professionals. The documentation is available in
both English and Chinese.

[![English Documentation](https://img.shields.io/badge/documentation-english-blue.svg)](https://tensorlayer.readthedocs.io/)
[![Chinese Documentation](https://img.shields.io/badge/documentation-%E4%B8%AD%E6%96%87-blue.svg)](https://tensorlayercn.readthedocs.io/)
[![Chinese Book](https://img.shields.io/badge/book-%E4%B8%AD%E6%96%87-blue.svg)](http://www.broadview.com.cn/book/5059/)

If you want to try the experimental features on the the master branch, you can find the latest document
[here](https://tensorlayer.readthedocs.io/en/latest/).

# Extensive Examples

You can find a large collection of examples that use TensorLayer in [here](examples/) and the following space:

<a href="https://github.com/tensorlayer/awesome-tensorlayer/blob/master/readme.md" target="\_blank">
	<div align="center">
		<img src="img/awesome-mentioned.png" width="40%"/>
	</div>
</a>

# Getting Start

TensorLayer 2.0 relies on TensorFlow, numpy, and others. To use GPUs, CUDA and cuDNN are required.

Install TensorFlow:

```bash
pip3 install tensorflow-gpu==2.0.0-rc1 # TensorFlow GPU (version 2.0 RC1)
pip3 install tensorflow # CPU version
```

Install the stable release of TensorLayer:

```bash
pip3 install tensorlayer
```

Install the unstable development version of TensorLayer:

```bash
pip3 install git+https://github.com/tensorlayer/tensorlayer.git
```

If you want to install the additional dependencies, you can also run
```bash
pip3 install --upgrade tensorlayer[all]              # all additional dependencies
pip3 install --upgrade tensorlayer[extra]            # only the `extra` dependencies
pip3 install --upgrade tensorlayer[contrib_loggers]  # only the `contrib_loggers` dependencies
```

If you are TensorFlow 1.X users, you can use TensorLayer 1.11.0:

```bash
# For last stable version of TensorLayer 1.X
pip3 install --upgrade tensorlayer==1.11.0
```

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

# Performance Benchmark

The following table shows the training speeds of [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) using TensorLayer and native TensorFlow on a TITAN Xp.

|   Mode    |       Lib       |  Data Format  | Max GPU Memory Usage(MB)  |Max CPU Memory Usage(MB) | Avg CPU Memory Usage(MB) | Runtime (sec) |
| :-------: | :-------------: | :-----------: | :-----------------: | :-----------------: | :-----------------: | :-----------: |
| AutoGraph | TensorFlow 2.0  | channel last  | 11833 |      2161         |        2136         |      74       |
|           | Tensorlayer 2.0 | channel last  | 11833 |      2187         |        2169         |      76       |
|   Graph   |      Keras      | channel last  | 8677 |      2580         |        2576         |      101       |
|   Eager   | TensorFlow 2.0  | channel last  | 8723 |      2052         |        2024         |      97       |
|           | TensorLayer 2.0 | channel last  | 8723 |      2010         |        2007         |      95       |

# Getting Involved

Please read the [Contributor Guideline](CONTRIBUTING.md) before submitting your PRs.

We suggest users to report bugs using Github issues. Users can also discuss how to use TensorLayer in the following slack channel.

<br/>

<a href="https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc" target="\_blank">
	<div align="center">
		<img src="img/join_slack.png" width="40%"/>
	</div>
</a>

<br/>

# Citing TensorLayer

If you find TensorLayer useful for your project, please cite the following paper：

```
@article{tensorlayer2017,
    author  = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
    journal = {ACM Multimedia},
    title   = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
    url     = {http://tensorlayer.org},
    year    = {2017}
}
```
