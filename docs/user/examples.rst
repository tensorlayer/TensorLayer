.. _example:

============
Examples
============

We list some examples here, but more tutorials and applications can be found in `Github examples <https://github.com/tensorlayer/tensorlayer/tree/master/examples>`__ and `Awesome-TensorLayer <https://github.com/tensorlayer/awesome-tensorlayer>`_.

Basics
============

 - Multi-layer perceptron (MNIST), simple usage. Classification task, see `tutorial_mnist_simple.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mnist_simple.py>`__.
 - Multi-layer perceptron (MNIST), dynamic model. Classification with dropout using iterator, see `tutorial_mnist_mlp_dynamic.py method2 <https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mnist_mlp_dynamic.py>`__.
 - Multi-layer perceptron (MNIST), static model. Classification with dropout using iterator, see `tutorial_mnist_mlp_static.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_mnist_mlp_static.py>`__.
 - Convolutional Network (CIFAR-10). Classification task, see `tutorial_cifar10_cnn_static.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/tutorial_cifar10_cnn_static.py>`_.
 - TensorFlow dataset API for object detection see `here <https://github.com/tensorlayer/tensorlayer/blob/master/examples/data_process/tutorial_tf_dataset_voc.py>`__.
 - Data augmentation with TFRecord. Effective way to load and pre-process data, see `tutorial_tfrecord*.py <https://github.com/tensorlayer/tensorlayer/tree/master/examples/data_process>`__ and `tutorial_cifar10_tfrecord.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/basic_tutorials/data_process/tutorial_tfrecord.py>`__.
 - Data augmentation with TensorLayer. See `tutorial_fast_affine_transform.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/data_process/tutorial_fast_affine_transform.py>`__ (for quick test only).

Pretrained Models
==================

 - VGG 16 (ImageNet). Classification task, see `tutorial_models_vgg16 <https://github.com/tensorlayer/tensorlayer/blob/master/examples/pretrained_cnn/tutorial_models_vgg16.py>`__.
 - VGG 19 (ImageNet). Classification task, see `tutorial_models_vgg19.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/pretrained_cnn/tutorial_vgg19.py>`__.
 - SqueezeNet (ImageNet). Model compression, see `tutorial_models_squeezenetv1.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/pretrained_cnn/tutorial_models_squeezenetv1.py>`__.
 - MobileNet (ImageNet). Model compression, see `tutorial_models_mobilenetv1.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/pretrained_cnn/tutorial_models_mobilenetv1.py>`__.
 - All pretrained models in `pretrained-models <https://github.com/tensorlayer/pretrained-models>`__.

Vision
==================

 - Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization, see `examples <https://github.com/tensorlayer/adaptive-style-transfer>`__.
 - ArcFace: Additive Angular Margin Loss for Deep Face Recognition, see `InsignFace <https://github.com/auroua/InsightFace_TF>`__.
 - BinaryNet. Model compression, see `mnist <https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_binarynet_mnist_cnn.py>`__ `cifar10 <https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_binarynet_cifar10_tfrecord.py>`__.
 - Ternary Weight Network. Model compression, see `mnist <https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_ternaryweight_mnist_cnn.py>`__ `cifar10 <https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_ternaryweight_cifar10_tfrecord.py>`__.
 - DoReFa-Net. Model compression, see `mnist <https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_dorefanet_mnist_cnn.py>`__ `cifar10 <https://github.com/tensorlayer/tensorlayer/blob/master/examples/quantized_net/tutorial_dorefanet_cifar10_tfrecord.py>`__.
 - QuanCNN. Model compression, sees `mnist <https://github.com/XJTUI-AIR-FALCON/tensorlayer/blob/master/examples/quantized_net/tutorial_quanconv_mnist.py>`__ `cifar10 <https://github.com/XJTUI-AIR-FALCON/tensorlayer/blob/master/examples/quantized_net/tutorial_quanconv_cifar10.py>`__.
 - Wide ResNet (CIFAR) by `ritchieng <https://github.com/ritchieng/wideresnet-tensorlayer>`__.
 - `Spatial Transformer Networks <https://arxiv.org/abs/1506.02025>`__ by `zsdonghao <https://github.com/zsdonghao/Spatial-Transformer-Nets>`__.
 - `U-Net for brain tumor segmentation <https://github.com/zsdonghao/u-net-brain-tumor>`__ by `zsdonghao <https://github.com/zsdonghao/u-net-brain-tumor>`__.
 - Variational Autoencoder (VAE) for (CelebA) by `yzwxx <https://github.com/yzwxx/vae-celebA>`__.
 - Variational Autoencoder (VAE) for (MNIST) by `BUPTLdy <https://github.com/BUPTLdy/tl-vae>`__.
 - Image Captioning - Reimplementation of Google's `im2txt <https://github.com/tensorflow/models/tree/master/research/im2txt>`__ by `zsdonghao <https://github.com/zsdonghao/Image-Captioning>`__.

Adversarial Learning
========================
 - DCGAN (CelebA). Generating images by `Deep Convolutional Generative Adversarial Networks <http://arxiv.org/abs/1511.06434>`__ by `zsdonghao <https://github.com/tensorlayer/dcgan>`__.
 - `Generative Adversarial Text to Image Synthesis <https://github.com/zsdonghao/text-to-image>`__ by `zsdonghao <https://github.com/zsdonghao/text-to-image>`__.
 - `Unsupervised Image to Image Translation with Generative Adversarial Networks <https://github.com/zsdonghao/Unsup-Im2Im>`__ by `zsdonghao <https://github.com/zsdonghao/Unsup-Im2Im>`__.
 - `Improved CycleGAN <https://github.com/luoxier/CycleGAN_Tensorlayer>`__ with resize-convolution by `luoxier <https://github.com/luoxier/CycleGAN_Tensorlayer>`__.
 - `Super Resolution GAN <https://arxiv.org/abs/1609.04802>`__ by `zsdonghao <https://github.com/tensorlayer/SRGAN>`__.
 - `BEGAN: Boundary Equilibrium Generative Adversarial Networks <http://arxiv.org/abs/1703.10717>`__ by `2wins <https://github.com/2wins/BEGAN-tensorlayer>`__.
 - `DAGAN: Fast Compressed Sensing MRI Reconstruction <https://github.com/nebulaV/DAGAN>`__ by `nebulaV <https://github.com/nebulaV/DAGAN>`__.

Natural Language Processing
==============================

 - Recurrent Neural Network (LSTM). Apply multiple LSTM to PTB dataset for language modeling, see `tutorial_ptb_lstm_state_is_tuple.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_ptb/tutorial_ptb_lstm_state_is_tuple.py>`__.
 - Word Embedding (Word2vec). Train a word embedding matrix, see `tutorial_word2vec_basic.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_word_embedding/tutorial\_word2vec_basic.py>`__.
 - Restore Embedding matrix. Restore a pre-train embedding matrix, see `tutorial_generate_text.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_generation/tutorial_generate_text.py>`__.
 - Text Generation. Generates new text scripts, using LSTM network, see `tutorial_generate_text.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_generation/tutorial_generate_text.py>`__.
 - Chinese Text Anti-Spam by `pakrchen <https://github.com/pakrchen/text-antispam>`__.
 - `Chatbot in 200 lines of code <https://github.com/tensorlayer/seq2seq-chatbot>`__ for `Seq2Seq <http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#simple-seq2seq>`__.
 - FastText Sentence Classification (IMDB), see `tutorial_imdb_fasttext.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/text_classification/tutorial_imdb_fasttext.py>`__ by `tomtung <https://github.com/tomtung>`__.

Reinforcement Learning
==============================

 - Policy Gradient / Network (Atari Ping Pong), see `tutorial_atari_pong.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_atari_pong.py>`__.
 - Deep Q-Network (Frozen lake), see `tutorial_frozenlake_dqn.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_frozenlake_dqn.py>`__.
 - Q-Table learning algorithm (Frozen lake), see `tutorial_frozenlake_q_table.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_frozenlake_q_table.py>`__.
 - Asynchronous Policy Gradient using TensorDB (Atari Ping Pong) by `nebulaV <https://github.com/akaraspt/tl_paper>`__.
 - AC for discrete action space (Cartpole), see `tutorial_cartpole_ac.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_cartpole_ac.py>`__.
 - A3C for continuous action space (Bipedal Walker), see `tutorial_bipedalwalker_a3c*.py <https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_bipedalwalker_a3c_continuous_action.py>`__.
 - `DAGGER <https://www.cs.cmu.edu/%7Esross1/publications/Ross-AIStats11-NoRegret.pdf>`__ for (`Gym Torcs <https://github.com/ugo-nama-kun/gym_torcs>`__) by `zsdonghao <https://github.com/zsdonghao/Imitation-Learning-Dagger-Torcs>`__.
 - `TRPO <https://arxiv.org/abs/1502.05477>`__ for continuous and discrete action space by `jjkke88 <https://github.com/jjkke88/RL_toolbox>`__.

Miscellaneous
=================

- `Sipeed <https://github.com/sipeed/Maix-EMC>`__ : Run TensorLayer on AI Chips

..
   - TensorDB by `fangde <https://github.com/fangde>`__ see `tl_paper <https://github.com/akaraspt/tl_paper>`__.
   - A simple web service - `TensorFlask <https://github.com/JoelKronander/TensorFlask>`__ by `JoelKronander <https://github.com/JoelKronander>`__.

..
  Applications
  =============

  There are some good applications implemented by TensorLayer.
  You may able to find some useful examples for your project.
  If you want to share your application, please contact tensorlayer@gmail.com.

  1D CNN + LSTM for Biosignal
  ---------------------------------

  Author : `Akara Supratak <https://akaraspt.github.io>`__

  Introduction
  ^^^^^^^^^^^^

  Implementation
  ^^^^^^^^^^^^^^

  Citation
  ^^^^^^^^





.. _GitHub: https://github.com/tensorlayer/tensorlayer
.. _Deeplearning Tutorial: http://deeplearning.stanford.edu/tutorial/
.. _Convolutional Neural Networks for Visual Recognition: http://cs231n.github.io/
.. _Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/
.. _TensorFlow tutorial: https://www.tensorflow.org/versions/r0.9/tutorials/index.html
.. _Understand Deep Reinforcement Learning: http://karpathy.github.io/2016/05/31/rl/
.. _Understand Recurrent Neural Network: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
.. _Understand LSTM Network: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
.. _Word Representations: http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/
