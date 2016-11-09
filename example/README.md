[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/tensorlayer/Lobby#?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Help Wanted Issues](https://badge.waffle.io/zsdonghao/tensorlayer.svg?label=up-for-grabs&title=Help Wanted Issues)](https://waffle.io/zsdonghao/tensorlayer)



Example
==========

Note
====
* For Simplied Convolutional layer APIs see [Convolutional layer (Simplified)](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#convolutional-layer-simplified) on readthedocs website.
* If you get into trouble, you can start a discussion on [Gitter](https://gitter.im/tensorlayer/Lobby#?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge>),
[Help Wanted Issues](https://waffle.io/zsdonghao/tensorlayer),
[QQ group](https://github.com/zsdonghao/tensorlayer/blob/master/img/img_qq.png) and [Wechat group](tensorlayer@gmail.com).


Basics
============

 - Multi-layer perceptron (MNIST). A multi-layer perceptron implementation for MNIST classification task, see ``tutorial_mnist_simple.py``.

Computer Vision
==================

 - Denoising Autoencoder (MNIST). A multi-layer perceptron implementation for MNIST classification task, see ``tutorial_mnist.py``.
 - Stacked Denoising Autoencoder and Fine-Tuning (MNIST). A multi-layer perceptron implementation for MNIST classification task, see ``tutorial_mnist.py``.
 - Convolutional Network (MNIST). A Convolutional neural network implementation for classifying MNIST dataset, see ``tutorial_mnist.py``.
 - Convolutional Network (CIFAR-10). A Convolutional neural network implementation for classifying CIFAR-10 dataset, see ``tutorial_cifar10.py`` and ``tutorial_cifar10_tfrecord.py``.
 - VGG 16 (ImageNet). A Convolutional neural network implementation for classifying ImageNet dataset, see ``tutorial_vgg16.py``.
 - VGG 19 (ImageNet). A Convolutional neural network implementation for classifying ImageNet dataset, see ``tutorial_vgg19.py``.
 - InceptionV3 (ImageNet). A Convolutional neural network implementation for classifying ImageNet dataset, see ``tutorial_inceptionV3_tfslim.py``.
 - Wild ResNet (CIFAR) by [ritchieng](https://github.com/ritchieng/wideresnet-tensorlayer).
 - More CNN implementations of `TF-Slim <https://github.com/tensorflow/models/tree/master/slim#pre-trained-models>`_ can be connected to TensorLayer via SlimNetsLayer.

Natural Language Processing
==============================

 - Recurrent Neural Network (LSTM). Apply multiple LSTM to PTB dataset for language modeling, see ``tutorial_ptb_lstm_state_is_tuple.py``.
 - Word Embedding - Word2vec. Train a word embedding matrix, see ``tutorial_word2vec_basic.py``.
 - Restore Embedding matrix. Restore a pre-train embedding matrix, see ``tutorial_generate_text.py``.
 - Text Generation. Generates new text scripts, using LSTM network, see ``tutorial_generate_text.py``.
 - Machine Translation (WMT). Translate English to French. Apply Attention mechanism and Seq2seq to WMT English-to-French translation data, see ``tutorial_translate.py``.

Reinforcement Learning
==============================

 - Deep Reinforcement Learning - Pong Game. Teach a machine to play Pong games, see ``tutorial_atari_pong.py``.


Special Examples
=================

 - Merge TF-Slim into TensorLayer. ``tutorial_inceptionV3_tfslim.py``.
 - MultiplexerLayer. ``tutorial_mnist_multiplexer.py``.
 - Data augmentation with TFRecord. Effective way to load and pre-process data, see ``tutorial_tfrecord*.py`` and ``tutorial_cifar10_tfrecord.py``.
 - Data augmentation with TensorLayer, see ``tutorial_image_preprocess.py``.

 
