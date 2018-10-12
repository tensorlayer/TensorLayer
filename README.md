<a href="https://tensorlayer.readthedocs.io/">
    <div align="center">
        <img src="img/tl_transparent_logo.png" width="50%" height="30%"/>
    </div>
</a>




![PyPI Stable Version](http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/github/release/tensorlayer/tensorlayer.svg?label=PyPI%20-%20Release)
![PyPI RC Version](http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/github/release/tensorlayer/tensorlayer/all.svg?label=PyPI%20-%20Pre-Release)
[![Github commits (since latest release)](http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/github/commits-since/tensorlayer/tensorlayer/latest.svg)](https://github.com/tensorlayer/tensorlayer/compare/1.10.1...master)
[![PyPI - Python Version](http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/pypi/pyversions/tensorlayer.svg)](https://pypi.org/project/tensorlayer/)
[![Supported TF Version](http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/badge/tensorflow-1.6.0+-blue.svg)](https://github.com/tensorflow/tensorflow/releases)
[![Downloads](http://pepy.tech/badge/tensorlayer)](http://pepy.tech/project/tensorlayer)

[![Build Status](http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/travis/tensorlayer/tensorlayer/master.svg?label=Travis)](https://travis-ci.org/tensorlayer/tensorlayer)
[![CircleCI](http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/circleci/project/github/tensorlayer/tensorlayer/master.svg?label=Docker%20Build)](https://circleci.com/gh/tensorlayer/tensorlayer/tree/master)
[![Documentation Status](http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/readthedocs/tensorlayer/latest.svg?label=ReadTheDocs-EN)](https://tensorlayer.readthedocs.io/)
[![Documentation Status](http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/readthedocs/tensorlayercn/latest.svg?label=ReadTheDocs-CN)](https://tensorlayercn.readthedocs.io/)
[![PyUP Updates](https://pyup.io/repos/github/tensorlayer/tensorlayer/shield.svg)](https://pyup.io/repos/github/tensorlayer/tensorlayer/)
[![Docker Pulls](http://ec2-35-178-47-120.eu-west-2.compute.amazonaws.com/docker/pulls/tensorlayer/tensorlayer.svg?maxAge=604800)](https://hub.docker.com/r/tensorlayer/tensorlayer/)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d6b118784e25435498e7310745adb848)](https://www.codacy.com/app/tensorlayer/tensorlayer)

<br/>

<a href="https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc" target="\_blank">
	<div align="center">
		<img src="img/join_slack.png" width="40%"/>
	</div>
</a>

<br/>

TensorLayer is a novel TensorFlow-based deep learning and reinforcement learning library designed for researchers and engineers. It provides a large collection of customizable neural layers / functions that are key to build real-world AI applications. TensorLayer is awarded the 2017 Best Open Source Software by the [ACM Multimedia Society](http://www.acmmm.org/2017/mm-2017-awardees/).

# Features

As deep learning practitioners, we have been looking for a library that can address various development
 purposes. This library is easy to adopt by providing diverse examples, tutorials and pre-trained models.
Also, it allow users to easily fine-tune TensorFlow; while being suitable for production deployment. TensorLayer aims to satisfy all these purposes. It has three key features:

- ***Simplicity*** : TensorLayer lifts the low-level dataflow interface of TensorFlow to *high-level* layers / models. It is very easy to learn through the rich [example codes](https://github.com/tensorlayer/awesome-tensorlayer) contributed by a wide community.
- ***Flexibility*** : TensorLayer APIs are transparent: it does not mask TensorFlow from users; but leaving massive hooks that help *low-level tuning* and *deep customization*.
- ***Zero-cost Abstraction*** : TensorLayer can achieve the *full power* of TensorFlow. The following table shows the training speeds of classic models using TensorLayer and native TensorFlow on a Titan X Pascal GPU.

	|             	| CIFAR-10      | PTB LSTM      | Word2Vec      |
	|-------------	|---------------|---------------|---------------|
	| TensorLayer 	| 2528 images/s | 18063 words/s | 58167 words/s |
	| TensorFlow  	| 2530 images/s | 18075 words/s | 58181 words/s |


TensorLayer stands at a unique spot in the library landscape. Other wrapper libraries like Keras and TFLearn also provide high-level abstractions. They, however, often
hide the underlying engine from users, which make them hard to customize
and fine-tune. On the contrary, TensorLayer APIs are generally flexible and transparent.
Users often find it easy to start with the examples and tutorials, and then dive
into TensorFlow seamlessly. In addition, TensorLayer does not create library lock-in through native supports for importing components from Keras, TFSlim and TFLearn.

TensorLayer has a fast growing usage among top researchers and engineers, from universities like
Imperial College London, UC Berkeley, Carnegie Mellon University, Stanford University, and
University of Technology of Compiegne (UTC), and companies like Google, Microsoft, Alibaba, Tencent, Xiaomi, and Bloomberg.

# Tutorials and Real-World Applications

You can find a large collection of tutorials, examples and real-world applications using TensorLayer through the following space:

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

# Install

TensorLayer has pre-requisites including TensorFlow, numpy, and others. For GPU support, CUDA and cuDNN are required.
The simplest way to install TensorLayer is to use the **Py**thon **P**ackage **I**ndex (PyPI):

```bash
# for last stable version
pip install --upgrade tensorlayer

# for latest release candidate
pip install --upgrade --pre tensorlayer

# if you want to install the additional dependencies, you can also run
pip install --upgrade tensorlayer[all]              # all additional dependencies
pip install --upgrade tensorlayer[extra]            # only the `extra` dependencies
pip install --upgrade tensorlayer[contrib_loggers]  # only the `contrib_loggers` dependencies
```

Alternatively, you can install the development version by directly pulling from github:


```bash
pip install git+https://github.com/tensorlayer/tensorlayer.git
```

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
nvidia-docker run -it --rm -p 8888:88888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu

# for GPU version and Python 3
docker pull tensorlayer/tensorlayer:latest-gpu-py3
nvidia-docker run -it --rm -p 8888:8888 -p 6006:6006 -e PASSWORD=JUPYTER_NB_PASSWORD tensorlayer/tensorlayer:latest-gpu-py3
```

# Contribute

Please read the [Contributor Guideline](https://github.com/tensorlayer/tensorlayer/blob/master/CONTRIBUTING.md) before submitting your PRs.

# Cite
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
