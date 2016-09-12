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

TensorLayer is a Deep Learning (DL) and Reinforcement Learning (RL) library extended from [Google TensorFlow](https://www.tensorflow.org). It provides popular DL and RL modules that can be easily customized and assembled for tackling real-world machine learning problems. 

TensorLayer grow out from a need to combine the power of TensorFlow with the right modules for building neural networks. According to our years of experiences of working on real-world machine learning problems, we identify three features that are critical for a library that can be easily leveraged by researchers and practitioners:

- **Easy-to-use**: We make TensorLayer easy to work with by providing mass tutorials that can be deployed and run through in minutes. A TensorFlow user may find it easier to bootstrap with the simple APIs provided by TensorLayer, and then dive into their implementation details only if necessary. 
- **Flexibility**: Developing machine learning pipelines for your specific requirements typically requires careful algorithm tunings from time to time. Without the loss of simplicity, TensorLayer allows users to customize their modules by exposing the low-level APIs of TensorFlow (e.g., training parameters, iteration control and tensor components). A customized algorithm can be therefore quickly extended from the rich sample codes that have been available in tutorials.
- **Performance**: TensorLayer provides zero-cost abstraction of TensorFlow. It does not enforce any extra overhead. It can efficiently run on either heterogenous platforms or multiple computation nodes.

A frequent question regarding TensorLayer is that why do we develop a new library instead of leveraging existing ones like [Keras](https://github.com/fchollet/keras) and [Tflearn](https://github.com/tflearn/tflearn). TensorLayer differentiates from those with its pursuits for flexibility and performance. 

A machine learning user may find it comfortable to bootstrap with Keras and Tflearn. However, she can quickly realize that it becomes necessary to carefully customize her modules as machine learning problems can largely vary from others. These libraries provide a high-level abstraction to hide as many as details of low-level engine from users. Though good for mastering, it becomes hard for them to be tuned from the bottom, which is an essential requirement for many researchers and practitioners. In the end, an experimental algorithm may need to be deployed and tested in a real-world setting. TensorLayer allows seamless deployment into distributed and heterogeneous environments with its first-class support for TensorFlow. 

# Installation

The simplest way to install TensorLayer is as follow. 

```python
pip install git+https://github.com/zsdonghao/tensorlayer.git
```

However, TensorLayer has some prerequisites that need to be installed first, including TensorFlow, numpy and matplotlib. For GPU support CUDA and cuDNN are required. Besides, TensorLayer can be installed as editable mode. Please check [detailed installation instructions](http://tensorlayer.readthedocs.io/en/latest/user/installation.html).

# Simple Tutorial

We provide a lot of simple functions (like `fit()` , `test()`), however, if you want to understand the details and be a machine learning expert, we suggest you to train the network by using TensorFlow’s methods like `sess.run()`, see `tutorial_mnist.py` for more details. More examples can be found  [here](http://tensorlayer.readthedocs.io/en/latest/user/example.html).

```python
import tensorflow as tf
import tensorlayer as tl
import time

sess = tf.InteractiveSession()

# prepare data
X_train, y_train, X_val, y_val, X_test, y_test = \
                                tl.files.load_mnist_dataset(shape=(-1,784))
sess = tf.InteractiveSession()
# define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

# define the network
network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')

# the softmax is implemented internally in tl.cost.cross_entropy(y, y_) to
# speed up computation, so we use identity here.
# see tf.nn.sparse_softmax_cross_entropy_with_logits()
network = tl.layers.DenseLayer(network, n_units=10,
                                act = tf.identity,
                                name='output_layer')
# define cost function and metric.
y = network.outputs
cost = tl.cost.cross_entropy(y, y_)
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# initialize all variables
sess.run(tf.initialize_all_variables())

# print network information
network.print_params()
network.print_layers()

# train the network, we recommend to use tl.iterate.minibatches()  检查这里
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=500, n_epoch=500, print_freq=5,
            X_val=X_val, y_val=y_val, eval_train=False)

# evaluation
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

# save the network to .npz file
tl.files.save_npz(network.all_params , name='model.npz')
sess.close()
```

# Documentation

The documentation [[Online]](http://tensorlayer.readthedocs.io/en/latest/) [[PDF]](https://media.readthedocs.org/pdf/tensorlayer/latest/tensorlayer.pdf) [[Epub]](http://readthedocs.org/projects/tensorlayer/downloads/epub/latest/) [[HTML]](http://readthedocs.org/projects/tensorlayer/downloads/htmlzip/latest/) describes the usages of TensorLayer APIs. It is also a self-contained document that walks through different types of deep neural networks, reinforcement learning and their applications in Natural Language Processing (NLP) problems. We have included the corresponding modularized implementations of Google TensorFlow Deep Learning tutorial, so you can read the TensorFlow tutorial [[en]](https://www.tensorflow.org/versions/master/tutorials/index.html) [[cn]](http://wiki.jikexueyuan.com/project/tensorflow-zh/) along with our documents.

# Contribution Guide

TensorLayer started as an internal repository at Imperial College London, helping researchers to test their new methods. It now encourages researchers from all over the world to controbute their methods so as to promote the development of machine learning. You can either contact us directly to discuss your ideas, or fork our repository and make a pull request.

- 🇬🇧If you are in London, we can discuss in person
- 🇨🇳为了方便华人开发者，我们正在建立 [中文文档](http://tensorlayercn.readthedocs.io/zh/latest/)，与此同时我们建立了多种交流渠道，您可把个人介绍和微信号发送到 haodong_cs@163.com 申请加入
- 🇹🇭เราขอเรียนเชิญนักพัฒนาคนไทยทุกคนที่สนใจจะเข้าร่วมทีมพัฒนา TensorLayer ติดต่อสอบถามรายละเอียดเพิ่มเติมได้ที่ haodong_cs@163.com

TensorLayer is released under the Apache 2.0 license.
