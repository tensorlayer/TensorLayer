<a href="https://tensorlayer.readthedocs.io/">
    <div align="center">
        <img src="img/tl_transparent_logo.png" width="50%" height="30%"/>
    </div>
</a>

|               | Information                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Documentation | [![Mentioned in Awesome TensorLayer](https://awesome.re/mentioned-badge.svg)](https://github.com/tensorlayer/awesome-tensorlayer) [![English Documentation](https://img.shields.io/badge/documentation-english-blue.svg)](https://tensorlayer.readthedocs.io/) [![Chinese Documentation](https://img.shields.io/badge/documentation-中文-blue.svg)](https://tensorlayercn.readthedocs.io/) [![Chinese Book](https://img.shields.io/badge/book-中文-blue.svg)](http://www.broadview.com.cn/book/5059/)                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Development   | [![Codacy Badge](https://api.codacy.com/project/badge/Grade/ca2a29ddcf7445588beff50bee5406d9)](https://app.codacy.com/app/tensorlayer/tensorlayer) [![PyPI version](https://badge.fury.io/py/tensorlayer.svg)](https://pypi.org/project/tensorlayer/) [![Github commits (since latest release)](https://img.shields.io/github/commits-since/tensorlayer/tensorlayer/latest.svg)](https://github.com/tensorlayer/tensorlayer/compare/1.8.6rc1...master) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorlayer.svg)](https://pypi.org/project/tensorlayer/) [![Supported TF Version](https://img.shields.io/badge/tensorflow-1.6.0+-blue.svg)](https://github.com/tensorflow/tensorflow/releases)                                                                                                                                                                                                                           |
| Build         | [![Build Status](https://img.shields.io/travis/tensorlayer/tensorlayer.svg?label=Travis&branch=master)](https://travis-ci.org/tensorlayer/tensorlayer) [![CircleCI](https://img.shields.io/circleci/project/github/tensorlayer/tensorlayer.svg?label=Docker%20Build&branch=master)](https://circleci.com/gh/tensorlayer/tensorlayer/tree/master) [![Documentation Status](https://img.shields.io/readthedocs/tensorlayer/latest.svg?label=ReadTheDocs-EN)](https://tensorlayer.readthedocs.io/) [![Documentation Status](https://img.shields.io/readthedocs/tensorlayercn/latest.svg?label=ReadTheDocs-CN)](https://tensorlayercn.readthedocs.io/) [![PyUP Updates](https://pyup.io/repos/github/tensorlayer/tensorlayer/shield.svg)](https://pyup.io/repos/github/tensorlayer/tensorlayer/) [![Docker Pulls](https://img.shields.io/docker/pulls/tensorlayer/tensorlayer.svg?maxAge=604800)](https://hub.docker.com/r/tensorlayer/tensorlayer/) |

<br/>

<a href="https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc" target="\_blank">
	<div align="center">
		<img src="img/join_slack.png" width="40%"/>
	</div>
</a>

<br/>

TensorLayer is a novel TensorFlow-based deep learning and reinforcement learning library designed for researchers and engineers. It provides a large collection of customizable neural layers / functions that are key to build real-world AI applications. TensorLayer is awarded the 2017 Best Open Source Software by the [ACM Multimedia Society](http://www.acmmm.org/2017/mm-2017-awardees/).

# Why another deep learning library: TensorLayer

As deep learning practitioners, we have been looking for a library that can address various development
 purposes. This library is easy to adopt by providing diverse examples, tutorials and pre-trained models. 
Also, it allow users to easily fine-tune TensorFlow; while being suitable for production deployment. TensorLayer aims to satisfy all these purposes. It has three key features:

- ***Simplicity*** : TensorLayer lifts the low-level dataflow interface of TensorFlow to *high-level* layers / models. It is very easy to learn through the rich [example codes](https://github.com/tensorlayer/awesome-tensorlayer) contributed by a wide community.
- ***Flexibility*** : TensorLayer APIs are transparent: it does not mask TensorFlow from users; but leaving massive hooks that help *low-level tuning* and *deep customization*.
- ***Zero-cost Abstraction*** : TensorLayer can achieve the *full power* of TensorFlow. The following table shows the training speeds of classic models using TensorLayer and native TensorFlow on a Titan X Pascal GPU.

		|             | CIFAR-10      | PTB LSTM      | Word2Vec      |
		|-------------|---------------|---------------|---------------|
		| TensorLayer | 2528 images/s | 18063 words/s | 58167 words/s |
		| TensorFlow  | 2530 images/s | 18075 words/s | 58181 words/s |


TensorLayer stands at a unique spot in the library landscape. Other wrapper libraries like Keras and TFLearn also provide high-level abstractions. They, however, often
hide the underlying engine from users, which make them hard to customize 
and fine-tune. On the contrary, TensorLayer APIs are generally flexible and transparent. 
Users often find it easy to start with the examples and tutorials, and then dive
into TensorFlow seamlessly. In addition, TensorLayer does not create library lock-in through native supports for importing components from Keras, TFSlim and TFLearn.

TensorLayer has a fast growing usage among top researchers and engineers, from universities like
Imperial College London, UC Berkeley, Carnegie Mellon University, Stanford University, and
University of Technology of Compiegne (UTC), and companies like Google, Microsoft, Alibaba, Tencent, Xiaomi, and Bloomberg.

# Install

TensorLayer has pre-requisites including TensorFlow, numpy, matplotlib and nltk (optional). For GPU support, CUDA and cuDNN are required. The simplest way to install TensorLayer is to use the Python Package Index (PyPI):

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

# Contribute

Please read the [Contributor Guideline](https://github.com/tensorlayer/tensorlayer/blob/rearrange-readme/CONTRIBUTING.md) before submitting your PRs.

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
