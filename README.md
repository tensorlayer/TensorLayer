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
	<img src="img/img_tensorlayer.png" width="40%" height="40%"/>
</div>
</a>

[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/tensorlayer/Lobby#?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Help Wanted Issues](https://badge.waffle.io/zsdonghao/tensorlayer.svg?label=up-for-grabs&title=Help Wanted Issues)](https://waffle.io/zsdonghao/tensorlayer)

TensorLayer is a Deep Learning (DL) and Reinforcement Learning (RL) library extended from [Google TensorFlow](https://www.tensorflow.org). It provides popular DL and RL modules that can be easily customized and assembled for tackling real-world machine learning problems. 

# Why TensorLayer

TensorLayer grow out from a need to combine the power of TensorFlow with the right building modules for deep neural networks. According to our years of research and practical experiences of tackling real-world machine learning problems, we come up with three design goals for TensorLayer:

- **Simplicity**: We make TensorLayer easy to work with by providing mass tutorials that can be deployed and run through in minutes. A TensorFlow user may find it easier to bootstrap with the simple, high-level APIs provided by TensorLayer, and then deep dive into their implementation details if need. 
- **Flexibility**: Developing an effective DL algorithm for a specific domain typically requires careful tunings from many aspects. Without the loss of simplicity, TensorLayer allows users to customize their modules by manipulating the native APIs of TensorFlow (e.g., training parameters, iteration control and tensor components).
- **Performance**: TensorLayer aims to provide zero-cost abstraction for TensorFlow. With its first-class support for TensorFlow, it can easily run on either heterogeneous platforms or multiple computation nodes without compromise in performance.

A frequent question regarding TensorLayer is that why do we develop a new library instead of leveraging existing ones like [Keras](https://github.com/fchollet/keras) and [Tflearn](https://github.com/tflearn/tflearn). TensorLayer differentiates with those with its pursuits for flexibility and performance. A DL user may find it comfortable to bootstrap with Keras and Tflearn. These libraries provide high-level abstractions and hide engine implementation details from users. Though good for using, it becomes hard to tune and modify from the bottom, which is necessary when addressing domain-specific problems (i.e., one model does not fit all). Nevertheless, flexibility does not always come with the loss of performance. TensorLayer allows seamless distributed and heterogeneous deployment with its first-class support for the TensorFlow runtime.

# Installation

TensorLayer has install prerequisites including TensorFlow, numpy and matplotlib. For GPU support, CUDA and cuDNN are required. Please check [here](http://tensorlayer.readthedocs.io/en/latest/user/installation.html) for detailed instructions.

If you already had the pre-requisites ready, the simplest way to install TensorLayer in your python program is: 

```python
pip install tensorlayer
or
pip install git+https://github.com/zsdonghao/tensorlayer.git
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
cost = tl.cost.cross_entropy(y, y_)
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# Define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# Initialize all variables
sess.run(tf.initialize_all_variables())

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

TensorLayer is a major ongoing research project in Data Science Institute, Imperial College London.
The goal of the project is to develop a compositional language while complex learning systems
can be build through composition of neural network modules.
The whole development is now participated by numerous contributors [here](https://github.com/zsdonghao/tensorlayer/releases).

<!--
TensorLayer started as an internal repository at Imperial College London, helping researchers to test their new methods. It now encourages researchers from all over the world to controbute their methods so as to promote the development of machine learning. You can either contact us to discuss your ideas, or fork our repository and make a pull request.
 -->

- 🇬🇧 If you are in London, we can discuss in person. Drop us an email to organize a meetup: tensorlayer@gmail.com.
- 🇨🇳 我们有官方的 [中文文档](http://tensorlayercn.readthedocs.io/zh/latest)。另外, 我们建立了多种交流渠道，如[QQ 群](img/img_qq.png) 和微信群. 需加入微信群，请将个人介绍和微信号发送到 tensorlayer@gmail.com.
- 🇹🇭 เราขอเรียนเชิญนักพัฒนาคนไทยทุกคนที่สนใจจะเข้าร่วมทีมพัฒนา TensorLayer ติดต่อสอบถามรายละเอียดเพิ่มเติมได้ที่ tensorlayer@gmail.com.

# License

TensorLayer is releazed under the Apache 2.0 license.
