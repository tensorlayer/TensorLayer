<a href="https://tensorlayer.readthedocs.io/">
    <div align="center">
        <img src="img/tl_transparent_logo.png" width="50%" height="30%"/>
    </div>
</a>

[![Build Status](https://img.shields.io/travis/tensorlayer/tensorlayer.svg?label=Travis&branch=master)](https://travis-ci.org/tensorlayer/tensorlayer)
[![PyPI version](https://badge.fury.io/py/tensorlayer.svg)](https://pypi.org/project/tensorlayer/)
[![Github commits (since latest release)](https://img.shields.io/github/commits-since/tensorlayer/tensorlayer/latest.svg)](https://github.com/tensorlayer/tensorlayer/compare/1.8.6rc0...master)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorlayer.svg)](https://pypi.org/project/tensorlayer/)
[![Supported TF Version](https://img.shields.io/badge/tensorflow-1.6.0+-blue.svg)](https://github.com/tensorflow/tensorflow/releases)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/ca2a29ddcf7445588beff50bee5406d9)](https://app.codacy.com/app/tensorlayer/tensorlayer)

[![CircleCI](https://img.shields.io/circleci/project/github/tensorlayer/tensorlayer.svg?label=Docker%20Build&branch=master)](https://circleci.com/gh/tensorlayer/tensorlayer/tree/master)
[![Docker Pulls](https://img.shields.io/docker/pulls/tensorlayer/tensorlayer.svg?maxAge=604800)](https://hub.docker.com/r/tensorlayer/tensorlayer/)
[![Documentation Status](https://img.shields.io/readthedocs/tensorlayer/stable.svg?label=ReadTheDocs)](https://tensorlayer.readthedocs.io/)
[![PyUP Updates](https://pyup.io/repos/github/tensorlayer/tensorlayer/shield.svg)](https://pyup.io/repos/github/tensorlayer/tensorlayer/)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/tensorlayer/Lobby)

<br/>

<a href="https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc" target="\_blank">
	<div align="center">
		<img src="img/join_slack.png" width="40%"/>
	</div>
</a>

<br/>

[![Mentioned in Awesome TensorLayer](https://awesome.re/mentioned-badge.svg)](https://github.com/tensorlayer/awesome-tensorlayer)
[![English Documentation](http://img.shields.io/badge/documentation-english-blue.svg)](https://tensorlayer.readthedocs.io/)
[![Chinese Documentation](http://img.shields.io/badge/documentation-中文-blue.svg)](https://tensorlayercn.readthedocs.io/)
[![Chinese Book](http://img.shields.io/badge/book-中文-blue.svg)](http://www.broadview.com.cn/book/5059/)

TensorLayer is a deep learning and reinforcement learning library on top of [TensorFlow](https://www.tensorflow.org). It provides rich neural layers and utility functions to help researchers and engineers build real-world AI applications. TensorLayer is awarded the 2017 Best Open Source Software by the prestigious [ACM Multimedia Society](http://www.acmmm.org/2017/mm-2017-awardees/).

# Why another deep learning library: TensorLayer

## Features

As TensorFlow users, we have been looking for a library that can serve for various development
 phases. This library is easy for beginners by providing rich neural network implementations,
examples and tutorials. Later, its APIs shall naturally allow users to leverage the powerful
features of TensorFlow, exhibiting best performance in addressing real-world problems. In the
end, the extra abstraction shall not compromise TensorFlow performance, and thus suit for
production deployment. TensorLayer is a novel library that aims to satisfy these requirements.

It has three key features:

- ***Simplicity*** : TensorLayer lifts the low-level dataflow abstraction of TensorFlow to *high-level* layers. It also provides users with [rich examples](https://github.com/tensorlayer/awesome-tensorlayer) to minimize learning barrier.
- ***Flexibility*** : TensorLayer APIs are transparent: it does not mask TensorFlow from users; but leaving massive hooks that support diverse *low-level tuning*.
- ***Zero-cost Abstraction*** : TensorLayer has negligible overheads and can thus achieve the *full performance* of TensorFlow.

## Negligible overhead

To show the overhead, we train classic deep learning models using TensorLayer and native TensorFlow
on a Titan X Pascal GPU.

|             	| CIFAR-10      	| PTB LSTM      	| Word2Vec      	|
|-------------	|---------------	|---------------	|---------------	|
| TensorLayer 	| 2528 images/s 	| 18063 words/s 	| 58167 words/s 	|
| TensorFlow  	| 2530 images/s 	| 18075 words/s 	| 58181 words/s 	|


## Why using TensorLayer instead of Keras or TFLearn

Similar to TensorLayer, Keras and TFLearn are also popular TensorFlow wrapper libraries.
These libraries are comfortable to start with. They provide high-level abstractions;
but mask the underlying engine from users. It is thus hard to customize model behaviors
and touch the essential features of TensorFlow.

Without compromise in simplicity, TensorLayer APIs are generally more flexible and transparent.
Users often find it easy to start with the examples and tutorials of TensorLayer, and then dive
into the TensorFlow low-level APIs only if need. TensorLayer does not create library lock-in. Users can easily import models from Keras, TFSlim and TFLearn into a TensorLayer environment.

TensorLayer has a fast growing usage in academic and industry organizations. It is used by researchers from
Imperial College London, Carnegie Mellon University, Stanford University,
University of Technology of Compiegne (UTC), Tsinghua University, UCLA,
and etc., as well as engineers from Google, Microsoft, Alibaba, Tencent, Xiaomi, Penguins Innovate, Bloomberg and many others.

# Installation

TensorLayer has pre-requisites including TensorFlow, numpy, matplotlib and nltk (optional). For GPU support, CUDA and cuDNN are required.

The simplest way to install TensorLayer is to use the Python Package Index (PyPI):

```bash
# for last stable version
pip install tensorlayer

# for latest release candidate
pip install --pre tensorlayer
```

Alternatively, you can install the development version by directly pulling from github:


```bash
pip install git+https://github.com/tensorlayer/tensorlayer.git
```

## Using Docker - a ready-to-use environment

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
nvidia-docker run -it --rm -p 8888:88888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu

# for GPU version and Python 3
docker pull tensorlayer/tensorlayer:latest-gpu-py3
nvidia-docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu-py3
```

# Contribute to TensorLayer

Please read the [Contributor Guideline](https://github.com/tensorlayer/tensorlayer/blob/rearrange-readme/CONTRIBUTING.md) before submitting your PRs.

# Citation
If you find this project useful, we would be grateful if you cite the TensorLayer paper：

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

TensorLayer is released under the Apache 2.0 license.
