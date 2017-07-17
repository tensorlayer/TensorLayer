.. _more:

============
Example
============


Basics
============

 - Multi-layer perceptron (MNIST). A multi-layer perceptron implementation for MNIST classification task, see ``tutorial_mnist_simple.py`` on `GitHub`_.

Computer Vision
==================

 - Denoising Autoencoder (MNIST). A multi-layer perceptron implementation for MNIST classification task, see ``tutorial_mnist.py`` on `GitHub`_.
 - Stacked Denoising Autoencoder and Fine-Tuning (MNIST). A multi-layer perceptron implementation for MNIST classification task, see ``tutorial_mnist.py`` on `GitHub`_.
 - Convolutional Network (MNIST). A Convolutional neural network implementation for classifying MNIST dataset, see ``tutorial_mnist.py`` on `GitHub`_.
 - Convolutional Network (CIFAR-10). A Convolutional neural network implementation for classifying CIFAR-10 dataset, see ``tutorial_cifar10.py`` and ``tutorial_cifar10_tfrecord.py``on `GitHub`_.
 - VGG 16 (ImageNet). A Convolutional neural network implementation for classifying ImageNet dataset, see ``tutorial_vgg16.py`` on `GitHub`_.
 - VGG 19 (ImageNet). A Convolutional neural network implementation for classifying ImageNet dataset, see ``tutorial_vgg19.py`` on `GitHub`_.
 - InceptionV3 (ImageNet). A Convolutional neural network implementation for classifying ImageNet dataset, see ``tutorial_inceptionV3_tfslim.py`` on `GitHub`_.
 - Wide ResNet (CIFAR) by `ritchieng <https://github.com/ritchieng/wideresnet-tensorlayer>`_.
 - More CNN implementations of `TF-Slim <https://github.com/tensorflow/models/tree/master/slim#pre-trained-models>`_ can be connected to TensorLayer via SlimNetsLayer.
 - `Spatial Transformer Networks <https://arxiv.org/abs/1506.02025>`_ by `zsdonghao <https://github.com/zsdonghao/Spatial-Transformer-Nets>`_.
 - `U-Net for brain tumor segmentation <https://github.com/zsdonghao/u-net-brain-tumor>`_ by `zsdonghao <https://github.com/zsdonghao/u-net-brain-tumor>`_.


Natural Language Processing
==============================

 - Recurrent Neural Network (LSTM). Apply multiple LSTM to PTB dataset for language modeling, see ``tutorial_ptb_lstm_state_is_tuple.py`` on `GitHub`_.
 - Word Embedding - Word2vec. Train a word embedding matrix, see ``tutorial_word2vec_basic.py`` on `GitHub`_.
 - Restore Embedding matrix. Restore a pre-train embedding matrix, see ``tutorial_generate_text.py`` on `GitHub`_.
 - Text Generation. Generates new text scripts, using LSTM network, see ``tutorial_generate_text.py`` on `GitHub`_.
 - Machine Translation (WMT). Translate English to French. Apply Attention mechanism and Seq2seq to WMT English-to-French translation data, see ``tutorial_translate.py`` on `GitHub`_.

Adversarial Learning
========================
 - DCGAN - Generating images by `Deep Convolutional Generative Adversarial Networks <http://arxiv.org/abs/1511.06434>`_ by `zsdonghao <https://github.com/zsdonghao/dcgan>`_.
 - `Generative Adversarial Text to Image Synthesis <https://github.com/zsdonghao/text-to-image>`_ by `zsdonghao <https://github.com/zsdonghao/text-to-image>`_.
 - `Unsupervised Image to Image Translation with Generative Adversarial Networks <https://github.com/zsdonghao/Unsup-Im2Im>`_ by `zsdonghao <https://github.com/zsdonghao/Unsup-Im2Im>`_.
 - `Super Resolution GAN <https://arxiv.org/abs/1609.04802>`_ by `zsdonghao <https://github.com/zsdonghao/SRGAN>`_.

Reinforcement Learning
==============================

 - Policy Gradient / Network - Pong Game. Teach a machine to play Pong games, see ``tutorial_atari_pong.py`` on `GitHub`_.
 - Q-Network - Frozen lake, see ``tutorial_frozenlake_q_network.py`` on `GitHub`_.
 - Q-Table learning algorithm - Frozen lake, see ``tutorial_frozenlake_q_table.py`` on `GitHub`_.
 - Asynchronous Deep Reinforcement Learning - Pong Game by `nebulaV <https://github.com/akaraspt/tl_paper>`_.

Applications
==============

- Image Captioning - Reimplementation of Google's `im2txt <https://github.com/tensorflow/models/tree/master/im2txt>`_ by `zsdonghao <https://github.com/zsdonghao/Image-Captioning>`_.
- A simple web service - `TensorFlask <https://github.com/JoelKronander/TensorFlask>`_ by `JoelKronander <https://github.com/JoelKronander>`_.

Special Examples
=================

 - Merge TF-Slim into TensorLayer. ``tutorial_inceptionV3_tfslim.py`` on `GitHub`_.
 - Merge Keras into TensorLayer. ``tutorial_keras.py`` on `GitHub`_.
 - MultiplexerLayer. ``tutorial_mnist_multiplexer.py`` on `GitHub`_.
 - Data augmentation with TFRecord. Effective way to load and pre-process data, see ``tutorial_tfrecord*.py`` and ``tutorial_cifar10_tfrecord.py`` on `GitHub`_.
 - Data augmentation with TensorLayer, see ``tutorial_image_preprocess.py`` on `GitHub`_.
 - TensorDB by `fangde <https://github.com/fangde>`_ see `here <https://github.com/akaraspt/tl_paper>`_.

..
  Applications
  =============

  There are some good applications implemented by TensorLayer.
  You may able to find some useful examples for your project.
  If you want to share your application, please contact tensorlayer@gmail.com.

  1D CNN + LSTM for Biosignal
  ---------------------------------

  Author : `Akara Supratak <https://akaraspt.github.io>`_

  Introduction
  ^^^^^^^^^^^^

  Implementation
  ^^^^^^^^^^^^^^

  Citation
  ^^^^^^^^





.. _GitHub: https://github.com/zsdonghao/tensorlayer
.. _Deeplearning Tutorial: http://deeplearning.stanford.edu/tutorial/
.. _Convolutional Neural Networks for Visual Recognition: http://cs231n.github.io/
.. _Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/
.. _TensorFlow tutorial: https://www.tensorflow.org/versions/r0.9/tutorials/index.html
.. _Understand Deep Reinforcement Learning: http://karpathy.github.io/2016/05/31/rl/
.. _Understand Recurrent Neural Network: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
.. _Understand LSTM Network: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
.. _Word Representations: http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/
