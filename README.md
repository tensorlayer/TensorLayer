<!--<div align="center">
	<div class="TensorFlow">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png" style=": left; margin-left: 5px; margin-bottom: 5px;"><br><br>
   </div>
   <div class="TensorLayer">
    <img src="https://www.tensorflow.org/images/tf_logo_transp.png" style=": right; margin-left: 5px; margin-bottom: 5px;">
    </div>
</div>
-->
<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/tl_transparent_logo.png" width="50%" height="30%"/>
</div>
</a>

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/ca2a29ddcf7445588beff50bee5406d9)](https://app.codacy.com/app/tensorlayer/tensorlayer?utm_source=github.com&utm_medium=referral&utm_content=tensorlayer/tensorlayer&utm_campaign=badger)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/tensorlayer/Lobby#?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Build Status](https://travis-ci.org/tensorlayer/tensorlayer.svg?branch=master)](https://travis-ci.org/tensorlayer/tensorlayer)
[![Documentation Status](https://readthedocs.org/projects/tensorlayer/badge/?version=latest)](http://tensorlayer.readthedocs.io/en/latest/?badge=latest)
[![Docker Pulls](https://img.shields.io/docker/pulls/tensorlayer/tensorlayer.svg?maxAge=604800)](https://hub.docker.com/r/tensorlayer/tensorlayer/)

TensorLayer is a deep learning and reinforcement learning library on top of [TensorFlow](https://www.tensorflow.org). It provides rich neural layers and utility functions to help researchers and engineers build real-world AI applications. TensorLayer is awarded the 2017 Best Open Source Software by the prestigious [ACM Multimedia Society](http://www.acmmm.org/2017/mm-2017-awardees/).

- Useful links: [Documentation](http://tensorlayer.readthedocs.io), [Examples](http://tensorlayer.readthedocs.io/en/latest/user/example.html), [中文文档](https://tensorlayercn.readthedocs.io), [中文书](http://www.broadview.com.cn/book/5059)

# News
* [18 Mar] Release experimental APIs for binary networks.
* [18 Jan] [《深度学习：一起玩转TensorLayer》](http://www.broadview.com.cn/book/5059) (Deep Learning using TensorLayer)
* [17 Dec] Release experimental APIs for distributed training (by [TensorPort](https://tensorport.com)). See [tiny example](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist_distributed.py).
* [17 Nov] Release data augmentation APIs for object detection, see [tl.prepro](http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html#object-detection).
* [17 Nov] Support [Convolutional LSTM](https://arxiv.org/abs/1506.04214), see [ConvLSTMLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#conv-lstm-layer).
* [17 Nov] Support [Deformable Convolution](https://arxiv.org/abs/1703.06211), see [DeformableConv2dLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#d-deformable-conv).
* [17 Sep] New example [Chatbot in 200 lines of code](https://github.com/zsdonghao/seq2seq-chatbot) for [Seq2Seq](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#simple-seq2seq).
<!--
* [17 Nov] Download [VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) in one line, see [load\_voc_dataset](http://tensorlayer.readthedocs.io/en/latest/modules/files.html#voc-2007-2012).
* [17 Sep] Release [ROI layer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#roi-layer) for Object Detection.
* [17 Jun] Release [SpatialTransformer2dAffineLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#spatial-transformer) for [Spatial Transformer Networks](https://github.com/zsdonghao/Spatial-Transformer-Nets) see [example code](https://github.com/zsdonghao/Spatial-Transformer-Nets).
* [17 Jun] Release [Sub-pixel Convolution 2D](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#super-resolution-layer) for Super-resolution see [SRGAN code](https://github.com/zsdonghao/SRGAN).
* [17 May] You can now use TensorLayer with [TFSlim](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py) and [Keras](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_keras.py) together!
-->



# Installation

TensorLayer has pre-requisites including TensorFlow, numpy, matplotlib and nltk (optional). For GPU support, CUDA and cuDNN are required. 
The simplest way to install TensorLayer is: 

```bash
# for master version (Recommended)
$ pip install git+https://github.com/tensorlayer/tensorlayer.git 

# for stable version 
$ pip install tensorlayer
```

Dockerfile is supplied to build images, build as usual

```bash
# for CPU version
$ docker build -t tensorlayer:latest .

# for GPU version
$ docker build -t tensorlayer:latest-gpu -f Dockerfile.gpu . 
```

Please check [documentation](http://tensorlayer.readthedocs.io/en/latest/user/installation.html) for detailed instructions. 


# Examples and Tutorials

Examples can be found [in this folder](https://github.com/zsdonghao/tensorlayer/tree/master/example) and [Github topic](https://github.com/search?q=topic%3Atensorlayer&type=Repositories).

## Basics
 - Multi-layer perceptron (MNIST) - Classification task, see [tutorial\_mnist\_simple.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist_simple.py).
 - Multi-layer perceptron (MNIST) - Classification using Iterator, see [method1](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mlp_dropout1.py) and [method2](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mlp_dropout2.py).


## Computer Vision
 - Denoising Autoencoder (MNIST). Classification task, see [tutorial_mnist.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py).
 - Stacked Denoising Autoencoder and Fine-Tuning (MNIST). Classification task, see [tutorial_mnist.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py).
 - Convolutional Network (MNIST). Classification task, see [tutorial_mnist.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py).
 - Convolutional Network (CIFAR-10). Classification task, see [tutorial\_cifar10.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cifar10.py) and [tutorial\_cifar10_tfrecord.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cifar10_tfrecord.py).
 - VGG 16 (ImageNet). Classification task, see [tl.models.VGG16](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_models_vgg16.py) or [tutorial_vgg16.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg16.py).
 - VGG 19 (ImageNet). Classification task, see [tutorial_vgg19.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg19.py).
 - InceptionV3 (ImageNet). Classification task, see [tutorial\_inceptionV3_tfslim.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py).
 - SqueezeNet (ImageNet). Model compression, see [tl.models.SqueezeNetV1](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_models_squeezenetv1.py) or [tutorial_squeezenet.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_squeezenet.py)
 - MobileNet (ImageNet). Model compression, see [tl.models.MobileNetV1](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_mobilenetv1.py) or [tutorial_mobilenet.py](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mobilenet.py).
 - BinaryNet. Model compression, see [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_binarynet_mnist_cnn.py) [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_binarynet_cifar10_tfrecord.py).
 - Ternary Weight Network. Model compression, see [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ternaryweight_mnist_cnn.py) [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ternaryweight_cifar10_tfrecord.py).
 - DoReFa-Net. Model compression, see [mnist](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_dorefanet_mnist_cnn.py) [cifar10](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_dorefanet_cifar10_tfrecord.py).
 - Wide ResNet (CIFAR) by [ritchieng](https://github.com/ritchieng/wideresnet-tensorlayer).
 - More CNN implementations of [TF-Slim](https://github.com/tensorflow/models/tree/master/research/slim) can be connected to TensorLayer via SlimNetsLayer.
 - [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) by [zsdonghao](https://github.com/zsdonghao/Spatial-Transformer-Nets).
 - [U-Net for brain tumor segmentation](https://github.com/zsdonghao/u-net-brain-tumor) by [zsdonghao](https://github.com/zsdonghao/u-net-brain-tumor).
 - Variational Autoencoder (VAE) for (CelebA) by [yzwxx](https://github.com/yzwxx/vae-celebA).
 - Variational Autoencoder (VAE) for (MNIST) by [BUPTLdy](https://github.com/BUPTLdy/tl-vae).
 - Image Captioning - Reimplementation of Google's [im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt) by [zsdonghao](https://github.com/zsdonghao/Image-Captioning).

## Natural Language Processing
 - Recurrent Neural Network (LSTM). Apply multiple LSTM to PTB dataset for language modeling, see [tutorial_ptb_lstm.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm.py) and [tutorial\_ptb\_lstm\_state\_is_tuple.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py).
 - Word Embedding (Word2vec). Train a word embedding matrix, see [tutorial\_word2vec_basic.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial\_word2vec_basic.py).
 - Restore Embedding matrix. Restore a pre-train embedding matrix, see [tutorial\_generate_text.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_generate_text.py).
 - Text Generation. Generates new text scripts, using LSTM network, see [tutorial\_generate_text.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_generate_text.py).
 - Chinese Text Anti-Spam by [pakrchen](https://github.com/pakrchen/text-antispam).
 - [Chatbot in 200 lines of code](https://github.com/zsdonghao/seq2seq-chatbot) for [Seq2Seq](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#simple-seq2seq).
 - FastText Sentence Classification (IMDB), see [tutorial\_imdb\_fasttext.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_imdb_fasttext.py) by [tomtung](https://github.com/tomtung).

## Adversarial Learning
- DCGAN (CelebA). Generating images by [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) by [zsdonghao](https://github.com/zsdonghao/dcgan).
- [Generative Adversarial Text to Image Synthesis](https://github.com/zsdonghao/text-to-image) by [zsdonghao](https://github.com/zsdonghao/text-to-image).
- [Unsupervised Image to Image Translation with Generative Adversarial Networks](https://github.com/zsdonghao/Unsup-Im2Im) by [zsdonghao](https://github.com/zsdonghao/Unsup-Im2Im).
- [Improved CycleGAN](https://github.com/luoxier/CycleGAN_Tensorlayer) with resize-convolution by [luoxier](https://github.com/luoxier/CycleGAN_Tensorlayer)
- [Super Resolution GAN](https://arxiv.org/abs/1609.04802) by [zsdonghao](https://github.com/zsdonghao/SRGAN).
- [DAGAN: Fast Compressed Sensing MRI Reconstruction](https://github.com/nebulaV/DAGAN) by [nebulaV](https://github.com/nebulaV/DAGAN).

## Reinforcement Learning
 - Policy Gradient / Network (Atari Ping Pong), see [tutorial\_atari_pong.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_atari_pong.py).
 - Deep Q-Network (Frozen lake), see [tutorial\_frozenlake_dqn.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_frozenlake_dqn.py).
 - Q-Table learning algorithm (Frozen lake), see [tutorial\_frozenlake\_q_table.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_frozenlake_q_table.py).
 - Asynchronous Policy Gradient using TensorDB (Atari Ping Pong) by [nebulaV](https://github.com/akaraspt/tl_paper).
 - AC for discrete action space (Cartpole), see [tutorial\_cartpole_ac.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cartpole_ac.py).
 - A3C for continuous action space (Bipedal Walker), see [tutorial\_bipedalwalker_a3c*.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_bipedalwalker_a3c_continuous_action.py).
 - [DAGGER](https://www.cs.cmu.edu/%7Esross1/publications/Ross-AIStats11-NoRegret.pdf) for ([Gym Torcs](https://github.com/ugo-nama-kun/gym_torcs)) by [zsdonghao](https://github.com/zsdonghao/Imitation-Learning-Dagger-Torcs).
 - [TRPO](https://arxiv.org/abs/1502.05477) for continuous and discrete action space by [jjkke88](https://github.com/jjkke88/RL_toolbox).

## Miscellaneous
 - Distributed Training. [mnist](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist_distributed.py) and [imagenet](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_imagenet_inceptionV3_distributed.py) by [jorgemf](https://github.com/jorgemf).
 - Merge TF-Slim into TensorLayer. [tutorial\_inceptionV3_tfslim.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py).
 - Merge Keras into TensorLayer. [tutorial_keras.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py).
 - Data augmentation with TFRecord. Effective way to load and pre-process data, see [tutorial_tfrecord*.py](https://github.com/zsdonghao/tensorlayer/tree/master/example) and [tutorial\_cifar10_tfrecord.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cifar10_tfrecord.py).
 - Data augmentation with TensorLayer, see [tutorial\_image_preprocess.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_image_preprocess.py).
 - TensorDB by [fangde](https://github.com/fangde) see [here](https://github.com/akaraspt/tl_paper).
- A simple web service - [TensorFlask](https://github.com/JoelKronander/TensorFlask) by [JoelKronander](https://github.com/JoelKronander).
- Float 16 half-precision model, see [tutorial\_mnist_float16.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist_float16.py)

## Notes
TensorLayer provides two set of Convolutional layer APIs, see [(Advanced)](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#convolutional-layer-pro) and [(Basic)](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#convolutional-layer-simplified) on readthedocs website.
<!-- 
* If you get into trouble, you can start a discussion on [Slack](https://join.slack.com/t/tensorlayer/shared_invite/MjI1NjQ5NTUxOTY5LTE1MDI3MDYwNTItYzYwNmFiZmZkOA), [Gitter](https://gitter.im/tensorlayer/Lobby#?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge>),
[Help Wanted Issues](https://waffle.io/zsdonghao/tensorlayer),
[QQ group](https://github.com/zsdonghao/tensorlayer/blob/master/img/img_qq.png) and [Wechat group](https://github.com/shorxp/tensorlayer-chinese/blob/master/docs/wechat_group.md). 
-->


# Features

## Design Philosophy

As TensorFlow users, we have been looking for a library that can serve for various development phases. This library is easy for beginners by providing rich neural network implementations,
examples and tutorials. Later, its APIs shall naturally allow users to leverage the powerful features of TensorFlow, exhibiting best performance in addressing real-world problems. In the end, the extra abstraction shall not compromise TensorFlow performance, and thus suit for production deployment. TensorLayer is a novel library that aims to satisfy these requirements. It has three key features:

- *Simplicity* : TensorLayer lifts the low-level dataflow abstraction of TensorFlow to **high-level** layers. It also provides users with massive examples and tutorials to minimize learning barrier.
- *Flexibility* : TensorLayer APIs are transparent: it does not mask TensorFlow from users; but leaving massive hooks that support diverse **low-level tuning**.
- *Zero-cost Abstraction* : TensorLayer is able to achieve the **full performance** of TensorFlow.

## Negligible Overhead

TensorLayer has negligible performance overhead. We benchmark classic deep learning
models using TensorLayer and native TensorFlow
on a Titan X Pascal GPU. Here are the training speeds of respective tasks:

|             	| CIFAR-10      	| PTB LSTM      	| Word2Vec      	|
|-------------	|---------------	|---------------	|---------------	|
| TensorLayer 	| 2528 images/s 	| 18063 words/s 	| 58167 words/s 	|
| TensorFlow  	| 2530 images/s 	| 18075 words/s 	| 58181 words/s 	|

## Compared with Keras and TFLearn

Similar to TensorLayer, Keras and TFLearn are also popular TensorFlow wrapper libraries.
These libraries are comfortable to start with. They provide high-level abstractions; 
but mask the underlying engine from users. It is thus hard to 
customize model behaviors and touch the essential features of TensorFlow. 
Without compromise in simplicity, TensorLayer APIs are generally more flexible and transparent.
Users often find it easy to start with the examples and tutorials of TensorLayer, and 
then dive into the TensorFlow low-level APIs only if need.
TensorLayer does not create library lock-in. Users can easily import models from Keras, TFSlim and TFLearn into
a TensorLayer environment.


# Documentation

The documentation [[Online]](http://tensorlayer.readthedocs.io/en/latest/) [[PDF]](https://media.readthedocs.org/pdf/tensorlayer/latest/tensorlayer.pdf) [[Epub]](http://readthedocs.org/projects/tensorlayer/downloads/epub/latest/) [[HTML]](http://readthedocs.org/projects/tensorlayer/downloads/htmlzip/latest/) describes the usages of TensorLayer APIs. It is also a self-contained document that walks through different types of deep neural networks, reinforcement learning and their applications in Natural Language Processing (NLP) problems. 

We have included the corresponding modularized implementations of Google TensorFlow Deep Learning tutorial, so you can read the TensorFlow tutorial [[en]](https://www.tensorflow.org/versions/master/tutorials/index.html) [[cn]](http://wiki.jikexueyuan.com/project/tensorflow-zh/) along with our document.
[Chinese documentation](http://tensorlayercn.readthedocs.io/zh/latest/) is also available.

<!---
# Your First Program

The first program trains a multi-layer perception network to solve the MNIST problem. We use the well-known  [scikit](http://scikit-learn.org/stable/)-style functions such as ``fit()`` and ``test()``. The program is self-explained.

```python
import tensorflow as tf
import tensorlayer as tl

sess = tf.InteractiveSession()

# Prepare data
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,784))

# Define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

# Define the neural network structure
network = tl.layers.InputLayer(x, name='input')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')

# The softmax is implemented internally in tl.cost.cross_entropy(y, y_) to
# speed up computation, so we use identity here.
# see tf.nn.sparse_softmax_cross_entropy_with_logits()
network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output')
                                
# Define cost function and metric.
y = network.outputs
cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# Define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

# Initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# Print network information
network.print_params()
network.print_layers()

# Train the network, we recommend to use tl.iterate.minibatches()
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=500, n_epoch=500, print_freq=5,
            X_val=X_val, y_val=y_val, eval_train=False)

# Evaluation
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

# Save the network to .npz file
tl.files.save_npz(network.all_params , name='model.npz')

sess.close()
```

We provide many helper functions (like `fit()` , `test()`) that is similar to Keras to facilitate your development; however, if you want to obtain a fine-grain control over the model or its training process, you can use TensorFlow’s methods like `sess.run()` in your program directly (`tutorial_mnist.py` provides more details about this). Many more DL and RL examples can be found [here](http://tensorlayer.readthedocs.io/en/latest/user/example.html).

[Tricks to use TL](https://github.com/wagamamaz/tensorlayer-tricks) is also a good introduction to use TensorLayer.
-->


# Academic and Industry Users

TensorLayer has an open and fast growing community. 
It has been widely used by researchers from Imperial College London, Carnegie Mellon University, Stanford University, 
Tsinghua University, UCLA, Linköping University and etc., 
as well as engineers from Google, Microsoft, Alibaba, Tencent, Penguins Innovate, ReFULE4, Bloomberg, GoodAILab and many others.

- 🇬🇧 If you have any question, we suggest to create an issue to discuss with us.
- 🇨🇳 我们有中文讨论社区: 如[QQ群](img/img_qq.png)和[微信群](https://github.com/shorxp/tensorlayer-chinese/blob/master/docs/wechat_group.md).


# Contribution Guideline

[Guideline in 5 lines](./CONTRIBUTING.md)


# Citation
If you find this project useful, we would be grateful if you cite the TensorLayer paper：

```
@article{tensorlayer2017,
author = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
journal = {ACM Multimedia},
title = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
url = {http://tensorlayer.org},
year = {2017}
}
```

# License

TensorLayer is released under the Apache 2.0 license.
