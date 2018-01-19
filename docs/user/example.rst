.. _example:

============
Example
============


Basics
============

 - Multi-layer perceptron (MNIST). Classification task, see `tutorial_mnist_simple.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist_simple.py>`_.
 - Multi-layer perceptron (MNIST). Classification using Iterator, see `method1 <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mlp_dropout1.py>`_ and `method2 <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mlp_dropout2.py>`_.

Computer Vision
==================

 - Denoising Autoencoder (MNIST). Classification task, see `tutorial_mnist.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py>`_.
 - Stacked Denoising Autoencoder and Fine-Tuning (MNIST). A MLP classification task, see `tutorial_mnist.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py>`_.
 - Convolutional Network (MNIST). Classification task, see `tutorial_mnist.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py>`_.
 - Convolutional Network (CIFAR-10). Classification task, see `tutorial_cifar10.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cifar10.py>`_ and `tutorial_cifar10_tfrecord.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cifar10_tfrecord.py>`_.
 - VGG 16 (ImageNet). Classification task, see `tutorial_vgg16.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg16.py>`_.
 - VGG 19 (ImageNet). Classification task, see `tutorial_vgg19.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_vgg19.py>`_.
 - InceptionV3 (ImageNet). Classification task, see `tutorial_inceptionV3_tfslim.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py>`_.
 - Wide ResNet (CIFAR) by `ritchieng <https://github.com/ritchieng/wideresnet-tensorlayer>`_.
 - More CNN implementations of `TF-Slim <https://github.com/tensorflow/models/tree/master/research/slim>`_ can be connected to TensorLayer via SlimNetsLayer.
 - `Spatial Transformer Networks <https://arxiv.org/abs/1506.02025>`_ by `zsdonghao <https://github.com/zsdonghao/Spatial-Transformer-Nets>`_.
 - `U-Net for brain tumor segmentation <https://github.com/zsdonghao/u-net-brain-tumor>`_ by `zsdonghao <https://github.com/zsdonghao/u-net-brain-tumor>`_.
 - Variational Autoencoder (VAE) for (CelebA) by `yzwxx <https://github.com/yzwxx/vae-celebA>`_.
 - Variational Autoencoder (VAE) for (MNIST) by `BUPTLdy <https://github.com/BUPTLdy/tl-vae>`_.
 - Image Captioning - Reimplementation of Google's `im2txt <https://github.com/tensorflow/models/tree/master/research/im2txt>`_ by `zsdonghao <https://github.com/zsdonghao/Image-Captioning>`_.

Natural Language Processing
==============================

 - Recurrent Neural Network (LSTM). Apply multiple LSTM to PTB dataset for language modeling, see `tutorial_ptb_lstm_state_is_tuple.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py>`_.
 - Word Embedding (Word2vec). Train a word embedding matrix, see `tutorial_word2vec_basic.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial\_word2vec_basic.py>`_.
 - Restore Embedding matrix. Restore a pre-train embedding matrix, see `tutorial_generate_text.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_generate_text.py>`_.
 - Text Generation. Generates new text scripts, using LSTM network, see `tutorial_generate_text.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_generate_text.py>`_.
 - Chinese Text Anti-Spam by `pakrchen <https://github.com/pakrchen/text-antispam>`_.
 - `Chatbot in 200 lines of code <https://github.com/zsdonghao/seq2seq-chatbot>`_ for `Seq2Seq <http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#simple-seq2seq>`_.
 - FastText Sentence Classification (IMDB), see `tutorial_imdb_fasttext.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_imdb_fasttext.py>`_ by `tomtung <https://github.com/tomtung>`_.

Adversarial Learning
========================
 - DCGAN (CelebA). Generating images by `Deep Convolutional Generative Adversarial Networks <http://arxiv.org/abs/1511.06434>`_ by `zsdonghao <https://github.com/zsdonghao/dcgan>`_.
 - `Generative Adversarial Text to Image Synthesis <https://github.com/zsdonghao/text-to-image>`_ by `zsdonghao <https://github.com/zsdonghao/text-to-image>`_.
 - `Unsupervised Image to Image Translation with Generative Adversarial Networks <https://github.com/zsdonghao/Unsup-Im2Im>`_ by `zsdonghao <https://github.com/zsdonghao/Unsup-Im2Im>`_.
 - `Improved CycleGAN <https://github.com/luoxier/CycleGAN_Tensorlayer>`_ with resize-convolution by `luoxier <https://github.com/luoxier/CycleGAN_Tensorlayer>`_.
 - `Super Resolution GAN <https://arxiv.org/abs/1609.04802>`_ by `zsdonghao <https://github.com/zsdonghao/SRGAN>`_.
 - `DAGAN: Fast Compressed Sensing MRI Reconstruction <https://github.com/nebulaV/DAGAN>`_ by `nebulaV <https://github.com/nebulaV/DAGAN>`_.

Reinforcement Learning
==============================

 - Policy Gradient / Network (Atari Ping Pong), see `tutorial_atari_pong.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_atari_pong.py>`_.
 - Deep Q-Network (Frozen lake), see `tutorial_frozenlake_dqn.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_frozenlake_dqn.py>`_.
 - Q-Table learning algorithm (Frozen lake), see `tutorial_frozenlake_q_table.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_frozenlake_q_table.py>`_.
 - Asynchronous Policy Gradient using TensorDB (Atari Ping Pong) by `nebulaV <https://github.com/akaraspt/tl_paper>`_.
 - AC for discrete action space (Cartpole), see `tutorial_cartpole_ac.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cartpole_ac.py>`_.
 - A3C for continuous action space (Bipedal Walker), see `tutorial_bipedalwalker_a3c*.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_bipedalwalker_a3c_continuous_action.py>`_.
 - `DAGGER <https://www.cs.cmu.edu/%7Esross1/publications/Ross-AIStats11-NoRegret.pdf>`_ for (`Gym Torcs <https://github.com/ugo-nama-kun/gym_torcs>`_) by `zsdonghao <https://github.com/zsdonghao/Imitation-Learning-Dagger-Torcs>`_.
 - `TRPO <https://arxiv.org/abs/1502.05477>`_ for continuous and discrete action space by `jjkke88 <https://github.com/jjkke88/RL_toolbox>`_.

Special Examples
=================

 - Distributed Training. `mnist <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist_distributed.py>`_ and `imagenet <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py>`_ by `jorgemf <https://github.com/jorgemf>`_.
 - Merge TF-Slim into TensorLayer. `tutorial_inceptionV3_tfslim.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py>`_.
 - Merge Keras into TensorLayer. `tutorial_keras.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py>`_.
 - Data augmentation with TFRecord. Effective way to load and pre-process data, see `tutorial_tfrecord*.py <https://github.com/zsdonghao/tensorlayer/tree/master/example>`_ and `tutorial_cifar10_tfrecord.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_cifar10_tfrecord.py>`_.
 - Data augmentation with TensorLayer, see `tutorial_image_preprocess.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_image_preprocess.py>`_.
 - TensorDB by `fangde <https://github.com/fangde>`_ see `here <https://github.com/akaraspt/tl_paper>`_.
 - A simple web service - `TensorFlask <https://github.com/JoelKronander/TensorFlask>`_ by `JoelKronander <https://github.com/JoelKronander>`_.
 - Float 16 half-precision model, see `tutorial_mnist_float16.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist_float16.py>`_.

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
