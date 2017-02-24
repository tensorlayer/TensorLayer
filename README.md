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
	<img src="img/img_tensorlayer.png" width="30%" height="30%"/>
</div>
</a>

[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/tensorlayer/Lobby#?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Help Wanted Issues](https://badge.waffle.io/zsdonghao/tensorlayer.svg?label=up-for-grabs&title=Help Wanted Issues)](https://waffle.io/zsdonghao/tensorlayer)

TensorLayer is a Deep Learning (DL) and Reinforcement Learning (RL) library extended from [Google TensorFlow](https://www.tensorflow.org). It provides popular DL and RL modules that can be easily customized and assembled for tackling real-world machine learning problems. 

* NEWS: Compatible with [TF-Slim](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#connect-tf-slim) and [Keras](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#connect-keras) 🆕
* NEWS: Compatible with TF 1.0 🆕
* NEWS: [Tricks to use TL](https://github.com/wagamamaz/tensorlayer-tricks) 🆕

# Why TensorLayer

TensorLayer grow out from a need to combine the power of TensorFlow with the right building modules for deep neural networks. According to our years of research and practical experiences of tackling real-world machine learning problems, we come up with three design goals for TensorLayer:

- **Simplicity**: We make TensorLayer easy to work with by providing mass tutorials that can be deployed and run through in minutes. A TensorFlow user may find it easier to bootstrap with the simple, high-level APIs provided by TensorLayer, and then deep dive into their implementation details if need. 
- **Flexibility**: Developing an effective DL algorithm for a specific domain typically requires careful tunings from many aspects. Without the loss of simplicity, TensorLayer allows users to customize their modules by manipulating the native APIs of TensorFlow (e.g., training parameters, iteration control and tensor components).
- **Performance**: TensorLayer aims to provide zero-cost abstraction for TensorFlow. With its first-class support for TensorFlow, it can easily run on either heterogeneous platforms or multiple computation nodes without compromise in performance.

A frequent question regarding TensorLayer is that why do we develop this library instead of leveraging existing ones like [Keras](https://github.com/fchollet/keras) and [Tflearn](https://github.com/tflearn/tflearn). TensorLayer differentiates with those with its pursuits for flexibility and performance. A DL user may find it comfortable to bootstrap with Keras and Tflearn. These libraries provide high-level abstractions that completely mask the underlying engine from users. Though good for using, it becomes hard to tune and modify from the bottom, which is necessary when addressing domain-specific problems (i.e., one model does not fit all). 

In contrast, TensorLayer advocates a flexible programming paradigm where third-party neural network libraries can be used interchangeably with the native TensorFlow functions. This allows users to enjoy the ease of using powerful pre-built modules without sacrificing the control of the underlying training process. TensorLayer's noninvasive nature also makes it easy to consolidate with other wrapper libraries such as [TF-Slim](https://github.com/tensorflow/models/tree/master/slim). However, flexibility does not come with the loss of performance and heterogeneity support. TensorLayer allows seamless distributed and heterogeneous deployment with its first-class support for the TensorFlow roadmap. 

# Installation

TensorLayer has install prerequisites including TensorFlow, numpy and matplotlib. For GPU support, CUDA and cuDNN are required. Please check [here](http://tensorlayer.readthedocs.io/en/latest/user/installation.html) for detailed instructions.

If you already had the pre-requisites ready (numpy, scipy, scikit-image, matplotlib and nltk(optional)), the simplest way to install TensorLayer in your python program is: 

```bash
[for master version] pip install git+https://github.com/zsdonghao/tensorlayer.git (Highly Recommended)
[for stable version] pip install tensorlayer
```

# Documentation

The documentation [[Online]](http://tensorlayer.readthedocs.io/en/latest/) [[PDF]](https://media.readthedocs.org/pdf/tensorlayer/latest/tensorlayer.pdf) [[Epub]](http://readthedocs.org/projects/tensorlayer/downloads/epub/latest/) [[HTML]](http://readthedocs.org/projects/tensorlayer/downloads/htmlzip/latest/) describes the usages of TensorLayer APIs. It is also a self-contained document that walks through different types of deep neural networks, reinforcement learning and their applications in Natural Language Processing (NLP) problems. 

We have included the corresponding modularized implementations of Google TensorFlow Deep Learning tutorial, so you can read the TensorFlow tutorial [[en]](https://www.tensorflow.org/versions/master/tutorials/index.html) [[cn]](http://wiki.jikexueyuan.com/project/tensorflow-zh/) along with our document.

[Chinese documentation](http://tensorlayercn.readthedocs.io/zh/latest/) is also available.

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
network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')

# The softmax is implemented internally in tl.cost.cross_entropy(y, y_) to
# speed up computation, so we use identity here.
# see tf.nn.sparse_softmax_cross_entropy_with_logits()
network = tl.layers.DenseLayer(network, n_units=10, act = tf.identity, name='output_layer')
                                
# Define cost function and metric.
y = network.outputs
cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# Define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

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

# Contribution Guideline

TensorLayer is a major ongoing research project in Data Science Institute, Imperial College London. TensorLayer contributors are from Imperial College, Tsinghua University, Carnegie Mellon University, Google, Microsoft, Bloomberg and etc.
The goal of the project is to develop a compositional language while complex learning systems
can be build through composition of neural network modules.
The whole development is now participated by numerous contributors [here](https://github.com/zsdonghao/tensorlayer/releases).

<!--
TensorLayer started as an internal repository at Imperial College London, helping researchers to test their new methods. It now encourages researchers from all over the world to controbute their methods so as to promote the development of machine learning. You can either contact us to discuss your ideas, or fork our repository and make a pull request.
 -->

- 🇬🇧 If you are in London, we can discuss in person. Drop us an email to organize a meetup: tensorlayer@gmail.com.
- 🇨🇳 我们有官方的 [中文文档](http://tensorlayercn.readthedocs.io/zh/latest)。另外, 我们建立了多种交流渠道，如[QQ 群](img/img_qq.png)和微信群*（申请入群时请star该项目，并告知github用户名）*. 需加入微信群，请将个人介绍和微信号发送到 tensorlayer@gmail.com.
- 🇹🇭 เราขอเรียนเชิญนักพัฒนาคนไทยทุกคนที่สนใจจะเข้าร่วมทีมพัฒนา TensorLayer ติดต่อสอบถามรายละเอียดเพิ่มเติมได้ที่ tensorlayer@gmail.com.


# Examples


## Note
* TensorLayer provides two set of Convolutional layer APIs, see [(Professional)](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#convolutional-layer-pro)and [(Simplified)](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#convolutional-layer-simplified) on readthedocs website.
* If you get into trouble, you can start a discussion on [Gitter](https://gitter.im/tensorlayer/Lobby#?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge>),
[Help Wanted Issues](https://waffle.io/zsdonghao/tensorlayer),
[QQ group](https://github.com/zsdonghao/tensorlayer/blob/master/img/img_qq.png) and [Wechat group](tensorlayer@gmail.com).


## Basics
 - Multi-layer perceptron (MNIST). A multi-layer perceptron implementation for MNIST classification task, see ``tutorial_mnist_simple.py``.

## Computer Vision
 - Denoising Autoencoder (MNIST). A multi-layer perceptron implementation for MNIST classification task, see ``tutorial_mnist.py``.
 - Stacked Denoising Autoencoder and Fine-Tuning (MNIST). A multi-layer perceptron implementation for MNIST classification task, see ``tutorial_mnist.py``.
 - Convolutional Network (MNIST). A Convolutional neural network implementation for classifying MNIST dataset, see ``tutorial_mnist.py``.
 - Convolutional Network (CIFAR-10). A Convolutional neural network implementation for classifying CIFAR-10 dataset, see ``tutorial_cifar10.py`` and ``tutorial_cifar10_tfrecord.py``.
 - VGG 16 (ImageNet). A Convolutional neural network implementation for classifying ImageNet dataset, see ``tutorial_vgg16.py``.
 - VGG 19 (ImageNet). A Convolutional neural network implementation for classifying ImageNet dataset, see ``tutorial_vgg19.py``.
 - InceptionV3 (ImageNet). A Convolutional neural network implementation for classifying ImageNet dataset, see ``tutorial_inceptionV3_tfslim.py``.
 - Wide ResNet (CIFAR) by [ritchieng](https://github.com/ritchieng/wideresnet-tensorlayer).
 - More CNN implementations of [TF-Slim](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models) can be connected to TensorLayer via SlimNetsLayer.

## Natural Language Processing
 - Recurrent Neural Network (LSTM). Apply multiple LSTM to PTB dataset for language modeling, see ``tutorial_ptb_lstm_state_is_tuple.py``.
 - Word Embedding - Word2vec. Train a word embedding matrix, see ``tutorial_word2vec_basic.py``.
 - Restore Embedding matrix. Restore a pre-train embedding matrix, see ``tutorial_generate_text.py``.
 - Text Generation. Generates new text scripts, using LSTM network, see ``tutorial_generate_text.py``.
 - Machine Translation (WMT). Translate English to French. Apply Attention mechanism and Seq2seq to WMT English-to-French translation data, see ``tutorial_translate.py``.

## Reinforcement Learning
 - Deep Reinforcement Learning - Pong Game. Teach a machine to play Pong games, see ``tutorial_atari_pong.py``.


## Applications
- Image Captioning - Reimplementation of Google's [im2txt](https://github.com/tensorflow/models/tree/master/im2txt) by [zsdonghao](https://github.com/zsdonghao/Image-Captioning).
- DCGAN - Generating images by [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) by [zsdonghao](https://github.com/zsdonghao/dcgan).
- A simple web service - [TensorFlask](https://github.com/JoelKronander/TensorFlask) by [JoelKronander](https://github.com/JoelKronander)

## Special Examples
 - Merge TF-Slim into TensorLayer. ``tutorial_inceptionV3_tfslim.py``.
 - Merge Keras into TensorLayer. ``tutorial_keras.py``.
 - MultiplexerLayer. ``tutorial_mnist_multiplexer.py``.
 - Data augmentation with TFRecord. Effective way to load and pre-process data, see ``tutorial_tfrecord*.py`` and ``tutorial_cifar10_tfrecord.py``.
 - Data augmentation with TensorLayer, see ``tutorial_image_preprocess.py``.



# License

TensorLayer is releazed under the Apache 2.0 license.
