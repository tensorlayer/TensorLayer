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


# TensorLayer: Deep learning and Reinforcement learning library for Academic and Industry.

TensorLayer is a deep learning and reinforcement learning library for researchers and practitioners. It is an extension library for [Google TensorFlow](https://www.tensorflow.org). It providers high-level APIs and pre-built training blocks that can largely simplify the development of complex learning models. TensorLayer is easy to be extended and customized for your needs. In addition, we provide a rich set of examples and tutorials to help you to build up your own deep learning and reinforcement learning algorithms.

The documentation describes the usages of TensorLayer APIs. It is also a self-contained document that includes a tutorial to walk through different types of neural networks, deep reinforcement learning and Natural Language Processing (NLP) etc. We have included the corresponding modularized implementations of Google TensorFlow Deep Learning tutorial, so you could read TensorFlow tutorial as the same time [[en]](https://www.tensorflow.org/versions/master/tutorials/index.html) [[cn]](http://wiki.jikexueyuan.com/project/tensorflow-zh/).

TensorLayer grow out from a need to combine the power of TensorFlow with the availability of the right building blocks for training neural networks. Its development is guided by a number of design goals:


<!--(改这里)-->

<!--Beginner: almost zero pre-knowledge, want to quickly bootstrap and have quick results-->
<!--intermediate user: need to customize their models to solve their domain-specific problems.-->
<!--advanced: needs to easily scale out the machine learning algorithm to multiple nodes, and deploy algorithms to different platforms.-->

### Beginner ---> Advanced user 是一个由上到下的图，先确定内容，再画图
Beginner
- Mass tutorials : Proviced modularized implementations of Google TensorFlow's Tutorials
- Easy to use	 : Proviced both scikit-type API and professional API
- Modularization : Proviced Layer and Wrapper xxx

Intermediate user
- Flexibility	 : Easy to reuse layer and 高级的模块化？（请看下面留言～～不知道如何表达好） 
- Transparency   : XX Easy to implement Dynamic Structure Network

Advanced user
- Extensibility	 : Easy to extend ???
- Performance    : Same speed with pure TensorFlow xx, simplicity don't sacrifice performance
- Cross-platform : Train on GPU, run on anywhere, from Distributed GPU Clusters to Embedded Systems


#### milo，这是Reuse 的一个例子，其他库很难做到，但TL很容易。你帮忙看看怎么表达好？？ 这篇Paper也是模块化的一个例子 https://arxiv.org/pdf/1511.02799v3.pdf 请看看～～ 我已经用TL实现了这篇paper，非常简单。感觉它和 https://en.wikipedia.org/wiki/Modular_neural_network 的意思是类似的，但如何才能让普罗大众听得明白呢？

```
Image0 -- CNN --
                 Concate --MLP -- Softmax
Image1 -- CNN --
	
	这两个CNN是同一个CNN
```
--


<!--科研难点：RL，DL哪些是必要的模块？抽象太高灵活性不足、太低可用性不足。-->

<!--Modularization-->
<!--- Level 1: -->
<!--	-- The input of a module is the output of previous module(s)-->
<!--	-- Modularization makes the network easy to be modified and optimized-->
<!--- Level 2: -->
<!--	-- A module can have special behavior such as cost function and pre-train method-->
<!--	-- Pre-train a module, then reuse the module to other network is common, so as to improve the accuracy or implement more complex applications-->

<!--- Level 3: -->
<!--	-- Module can be more complex, it could even be a dynamic module-->
<!--	-- Dynamic module can be seem as many networks which sharing some modules-->



 - **Transparency**: Developing advanced learning algorithms requires low-level tunning of the underlying training engine. TensorLayer exposes the implementation details of the TensorFlow in a structured way, and allows users to do low-level engine manupulations, such as the configurations of training process, iteration, initialization as well as the access to Tensor components and TPUs.
 - **Extensibility**: Be easy to use, extend and modify, to facilitate use in research and practition activities. A network is abstracted to regularization, cost and outputs of each layer. Other wraping libraries for TensorFlow are easy to be merged into TensorLayer, suitable for researchers.
 - **Performance**: The running speed under GPU support is the same with TensorFlow. TensorLayer can also run in a distributed  mode.
 - **Low learning curve**: To facilitate bootstrapping, we provide mass format-consistent examples covering Dropout, DropConnect, Denoising Autoencoder, LSTM, CNN etc, speed up your development.

🆕🆕🆕 [Chinese documentation](http://tensorlayercn.readthedocs.io/) is released.

Now, go through the [Overview](#Overview) to see how powerful it is !!!

-

####🇨🇳为了方便华人开发者，我们正在建立 [中文文档](http://tensorlayercn.readthedocs.io/zh/latest/)，与此同时我们建立了多种交流渠道，您可把个人介绍和微信号发送到 haodong_cs@163.com 申请加入

####🇹🇭เราขอเรียนเชิญนักพัฒนาคนไทยทุกคนที่สนใจจะเข้าร่วมทีมพัฒนา TensorLayer ติดต่อสอบถามรายละเอียดเพิ่มเติมได้ที่ haodong_cs@163.com

####🇬🇧If you are in London, we can discuss face to face.

Documentation [[Online]](http://tensorlayer.readthedocs.io/en/latest/) [[PDF]](https://media.readthedocs.org/pdf/tensorlayer/latest/tensorlayer.pdf) [[Epub]](http://readthedocs.org/projects/tensorlayer/downloads/epub/latest/) [[HTML]](http://readthedocs.org/projects/tensorlayer/downloads/htmlzip/latest/)



--
# Installation

The simplest way to install TensorLayer is as follow. 

```python
pip install git+https://github.com/zsdonghao/tensorlayer.git
```

However, TensorLayer has some prerequisites that need to be installed first, including TensorFlow, numpy and matplotlib. For GPU support CUDA and cuDNN are required. Besides, TensorLayer can be installed as editable mode. Please check [detailed installation instructions](http://tensorlayer.readthedocs.io/en/latest/user/installation.html).

--


# Hello World


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


--

<!--# Examples-->
<!--More examples can be found [here](http://tensorlayer.readthedocs.io/en/latest/user/example.html).-->

<!--### *Fully Connected Network*-->
<!--TensorLayer provides large amount of state-of-the-art Layers including Dropout, DropConnect, ReconLayer and so on.-->

<!--**<font color="grey"> Placeholder: </font>**-->

<!--All placeholder and variables can be initialized by the same way with Tensorflow's tutorial. For details please read *[tensorflow-placeholder](https://www.tensorflow.org/versions/master/api_docs/python/io_ops.html#placeholder)*, *[tensorflow-variables](https://www.tensorflow.org/versions/master/how_tos/variables/index.html)* and *[tensorflow-math](https://www.tensorflow.org/versions/r0.9/api_docs/python/math_ops.html)*.-->

<!--```python-->
<!--# For MNIST example, 28x28 images have 784 pixels, i.e, 784 inputs.-->
<!--import tensorflow as tf-->
<!--import tensorlayer as tl-->
<!--x = tf.placeholder(tf.float32, shape=[None, 784], name='x')-->
<!--y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')-->
<!--```-->

<!--**<font color="grey"> Rectifying Network with Dropout: </font>**-->

<!--```python-->
<!--# Define the network-->
<!--network = tl.layers.InputLayer(x, name='input_layer')-->
<!--network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')-->
<!--network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')-->
<!--network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')-->
<!--network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')-->
<!--network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')-->
<!--network = tl.layers.DenseLayer(network, n_units=10, act = tl.activation.identity, name='output_layer')-->
<!--# Start training-->
<!--...-->
<!--```-->
<!--**<font color="grey"> Vanilla Sparse Autoencoder: </font>**-->


<!--```python-->
<!--# Define the network-->
<!--network = tl.layers.InputLayer(x, name='input_layer')-->
<!--network = tl.layers.DenseLayer(network, n_units=196, act = tf.nn.sigmoid, name='sigmoid1')-->
<!--recon_layer1 = tl.layers.ReconLayer(network, x_recon=x, n_units=784, act = tf.nn.sigmoid, name='recon_layer1')-->
<!--# Start pre-train-->
<!--sess.run(tf.initialize_all_variables())-->
<!--recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name=None, n_epoch=200, batch_size=128, print_freq=10, save=True, save_name='w1pre_')-->
<!--...-->
<!--```-->
<!--**<font color="grey"> Denoising Autoencoder: </font>**-->


<!--```python-->
<!--# Define the network-->
<!--network = tl.layers.InputLayer(x, name='input_layer')-->
<!--network = tl.layers.DropoutLayer(network, keep=0.5, name='denoising1')   -->
<!--network = tl.layers.DenseLayer(network, n_units=196, act = tf.nn.relu, name='relu1')-->
<!--recon_layer1 = tl.layers.ReconLayer(network, x_recon=x, n_units=784, act = tf.nn.softplus, name='recon_layer1')-->
<!--# Start pre-train-->
<!--sess.run(tf.initialize_all_variables())-->
<!--recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=200, batch_size=128, print_freq=10, save=True, save_name='w1pre_')-->
<!--...-->
<!--```-->

<!--**<font color="grey"> Stacked Denoising Autoencoders: </font>**-->

<!--```python-->
<!--# Define the network-->
<!--network = tl.layers.InputLayer(x, name='input_layer')-->
<!--# denoise layer for Autoencoders-->
<!--network = tl.layers.DropoutLayer(network, keep=0.5, name='denoising1')-->
<!--# 1st layer-->
<!--network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')-->
<!--network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')-->
<!--x_recon1 = network.outputs-->
<!--recon_layer1 = tl.layers.ReconLayer(network, x_recon=x, n_units=784, act = tf.nn.softplus, name='recon_layer1')-->
<!--# 2nd layer-->
<!--network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')-->
<!--network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')-->
<!--recon_layer2 = tl.layers.ReconLayer(network, x_recon=x_recon1, n_units=800, act = tf.nn.softplus, name='recon_layer2')-->
<!--# 3rd layer-->
<!--network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')-->
<!--network = tl.layers.DenseLayer(network, n_units=10, act = tl.activation.identity, name='output_layer')-->

<!--sess.run(tf.initialize_all_variables())-->

<!--# Print all parameters before pre-train-->
<!--network.print_params()-->

<!--# Pre-train Layer 1-->
<!--recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10, save=True, save_name='w1pre_')-->
<!--# Pre-train Layer 2-->
<!--recon_layer2.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10, save=False)-->
<!--# Start training-->
<!--...-->
<!--```-->

<!--### *Convolutional Neural Network*-->

<!--Instead of feeding the images as 1D vectors, the images can be imported as 4D matrix, where [None, 28, 28, 1] represents [batchsize, height, width, channels]. Set 'batchsize' to 'None' means data with different batchsize can all filled into the placeholder.-->

<!--```python-->
<!--x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])-->
<!--y_ = tf.placeholder(tf.int64, shape=[None,])-->
<!--```-->

<!--**<font color="grey"> CNNs + MLP: </font>**-->

<!--A 2 layers CNN followed by 2 fully connected layers can be defined by the following codes:-->

<!--```python-->
<!--network = tl.layers.InputLayer(x, name='input_layer')-->
<!--network = tl.layers.Conv2dLayer(network,-->
<!--                        act = tf.nn.relu,-->
<!--                        shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch-->
<!--                        strides=[1, 1, 1, 1],-->
<!--                        padding='SAME',-->
<!--                        name ='cnn_layer1')     # output: (?, 28, 28, 32)-->
<!--network = tl.layers.Pool2dLayer(network,-->
<!--                        ksize=[1, 2, 2, 1],-->
<!--                        strides=[1, 2, 2, 1],-->
<!--                        padding='SAME',-->
<!--                        pool = tf.nn.max_pool,-->
<!--                        name ='pool_layer1',)   # output: (?, 14, 14, 32)-->
<!--network = tl.layers.Conv2dLayer(network,-->
<!--                        act = tf.nn.relu,-->
<!--                        shape = [5, 5, 32, 64], # 64 features for each 5x5 patch-->
<!--                        strides=[1, 1, 1, 1],-->
<!--                        padding='SAME',-->
<!--                        name ='cnn_layer2')     # output: (?, 14, 14, 64)-->
<!--network = tl.layers.Pool2dLayer(network,-->
<!--                        ksize=[1, 2, 2, 1],-->
<!--                        strides=[1, 2, 2, 1],-->
<!--                        padding='SAME',-->
<!--                        pool = tf.nn.max_pool,-->
<!--                        name ='pool_layer2',)   # output: (?, 7, 7, 64)-->
<!--network = tl.layers.FlattenLayer(network, name='flatten_layer')                                # output: (?, 3136)-->
<!--network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')                              # output: (?, 3136)-->
<!--network = tl.layers.DenseLayer(network, n_units=256, act = tf.nn.relu, name='relu1')           # output: (?, 256)-->
<!--network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')                              # output: (?, 256)-->
<!--network = tl.layers.DenseLayer(network, n_units=10, act = tl.activation.identity, name='output_layer')    # output: (?, 10)-->
<!--```-->
<!--For more powerful functions, please go to *[Read the Docs](http://tensorlayer.readthedocs.io/en/latest/)*.-->


<!--### *Recurrent Neural Network*-->

<!--**<font color="grey"> LSTM: </font>** -->

<!--Please go to *[Understand LSTM](http://tensorlayer.readthedocs.io/en/latest/user/tutorial.html#run-the-ptb-example)*.-->


<!--### *Reinforcement Learning*-->
<!--To understand Reinforcement Learning, a Blog (*[Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)*) and a Paper (*[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)*) are recommended. To play with RL, use *[OpenAI Gym](https://github.com/openai/gym)* as benchmark is recommended.-->

<!--**<font color="grey"> Pong Game: </font>**-->

<!--Atari Pong Game is a single agent example. *[Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)* using 130 lines of Python only *[(Code link)](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)* can be reimplemented as follow.-->

<!--```python-->
<!--# Policy network-->
<!--network = tl.layers.InputLayer(x, name='input_layer')-->
<!--network = tl.layers.DenseLayer(network, n_units= H , act = tf.nn.relu, name='relu_layer')-->
<!--network = tl.layers.DenseLayer(network, n_units= 1 , act = tf.nn.sigmoid, name='output_layer')-->
<!--```-->

<!--For RL part, please read *[Policy Gradient](http://tensorlayer.readthedocs.io/en/latest/user/tutorial.html#understand-reinforcement-learning)*.-->



<!--### *Cost Function*-->

<!--TensorLayer provides a simple way to creat you own cost function. Take a MLP below for example.-->

<!--```python-->
<!--network = tl.InputLayer(x, name='input_layer')-->
<!--network = tl.DropoutLayer(network, keep=0.8, name='drop1')-->
<!--network = tl.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')-->
<!--network = tl.DropoutLayer(network, keep=0.5, name='drop2')-->
<!--network = tl.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')-->
<!--network = tl.DropoutLayer(network, keep=0.5, name='drop3')-->
<!--network = tl.DenseLayer(network, n_units=10, act = tl.activation.identity, name='output_layer')-->
<!--```-->

<!--**<font color="grey"> Regularization of Weights: </font>**-->

<!--After initializing the variables, the informations of network parameters can be observed by using **<font color="grey">network.print_params()</font>**.-->

<!--```python-->
<!--sess.run(tf.initialize_all_variables())-->
<!--network.print_params()-->
<!-->> param 0: (784, 800) (mean: -0.000000, median: 0.000004 std: 0.035524)-->
<!-->> param 1: (800,) (mean: 0.000000, median: 0.000000 std: 0.000000)-->
<!-->> param 2: (800, 800) (mean: 0.000029, median: 0.000031 std: 0.035378)-->
<!-->> param 3: (800,) (mean: 0.000000, median: 0.000000 std: 0.000000)-->
<!-->> param 4: (800, 10) (mean: 0.000673, median: 0.000763 std: 0.049373)-->
<!-->> param 5: (10,) (mean: 0.000000, median: 0.000000 std: 0.000000)-->
<!-->> num of params: 1276810-->
<!--```-->

<!--The output of network is **<font color="grey">network.outputs</font>**, then the cross entropy can be defined as follow. Besides, to regularize the weights, the **<font color="grey">network.all_params</font>** contains all parameters of the network. In this case, **<font color="grey">network.all_params</font>** = [W1, b1, W2, b2, Wout, bout] according to param 0, 1 ... 5 shown by **<font color="grey">network.print_params()</font>**. Then max-norm regularization on W1 and W2 can be performed as follow.-->

<!--```python-->
<!--y = network.outputs-->
<!--cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))-->
<!--cost = cross_entropy-->
<!--cost = cost + tl.cost.maxnorm_regularizer(1.0)(network.all_params[0]) + tl.cost.maxnorm_regularizer(1.0)(network.all_params[2])-->
<!--```-->
<!--In addition, all TensorFlow's regularizers like **<font color="grey">tf.contrib.layers.l2_regularizer</font>** can be used with TensorLayer.-->

<!--**<font color="grey"> Regularization of Activation Outputs: </font>**-->

<!--Instance method **<font color="grey">network.print_layers()</font>** prints all outputs of different layers in order. To achieve regularization on activation output, you can use **<font color="grey">network.all_layers</font>** which contains all outputs of different layers. If you want to use L1 penalty on the activations of first hidden layer, just simply add **<font color="grey">tf.contrib.layers.l2_regularizer(lambda_l1)(network.all_layers[1])</font>** to the cost function.-->

<!--```python-->
<!--network.print_layers()-->
<!-->> layer 0: Tensor("dropout/mul_1:0", shape=(?, 784), dtype=float32)-->
<!-->> layer 1: Tensor("Relu:0", shape=(?, 800), dtype=float32)-->
<!-->> layer 2: Tensor("dropout_1/mul_1:0", shape=(?, 800), dtype=float32)-->
<!-->> layer 3: Tensor("Relu_1:0", shape=(?, 800), dtype=float32)-->
<!-->> layer 4: Tensor("dropout_2/mul_1:0", shape=(?, 800), dtype=float32)-->
<!-->> layer 5: Tensor("add_2:0", shape=(?, 10), dtype=float32)-->
<!--```-->
<!--For more powerful functions, please go to *[Read the Docs](http://tensorlayer.readthedocs.io/en/latest/)*.-->

<!--# Easy to Modify (delete ?)-->
<!--**<font color="grey"> Modifying Pre-train Behaviour: </font>**-->


<!--Greedy layer-wise pretrain is an important task for deep neural network initialization, while there are many kinds of pre-train metrics according to different architectures and applications.-->

<!--For example, the pre-train process of *[Vanilla Sparse Autoencoder](http://deeplearning.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity)* can be implemented by using KL divergence as the following code, but for *[Deep Rectifier Network](http://www.jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf)*, the sparsity can be implemented by using the L1 regularization of activation output.-->

<!--```python-->
<!--# Vanilla Sparse Autoencoder-->
<!--beta = 4-->
<!--rho = 0.15-->
<!--p_hat = tf.reduce_mean(activation_out, reduction_indices = 0)-->
<!--KLD = beta * tf.reduce_sum( rho * tf.log(tf.div(rho, p_hat)) + (1- rho) * tf.log((1- rho)/ (tf.sub(float(1), p_hat))) )-->
<!--```-->

<!--For this reason, TensorLayer provides a simple way to modify or design your own pre-train metrice. For Autoencoder, TensorLayer uses **ReconLayer.*__*init__()** to define the reconstruction layer and cost function, to define your own cost function, just simply modify the **self.cost** in **ReconLayer.*__*init__()**. To creat your own cost expression please read *[Tensorflow Math](https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html)*. By default, **ReconLayer** only updates the weights and biases of previous 1 layer by using **self.train_params = self.all _params[-4:]**, where the 4 parameters are [W_encoder, b_encoder, W_decoder, b_decoder]. If you want to update the parameters of previous 2 layers, simply modify **[-4:]** to **[-6:]**.-->


<!--```python    -->
<!--ReconLayer.__init__(...):-->
<!--    ...-->
<!--    self.train_params = self.all_params[-4:]-->
<!--    ...-->
<!--	self.cost = mse + L1_a + L2_w-->
<!--```-->

<!--**<font color="grey"> Adding Customized Regularizer: </font>**-->

<!--See tensorlayer/cost.py-->


# Ways to Contribute

TensorLayer was initially developed as a part of project in Imperial College London, helping researchers to test their new methods. It now encourage researches from all over the world to publish their new methods so as to promote the development of machine learning.

Your method can be merged into TensorLayer, if you can prove it is better than the existing methods. Test script with detailed descriptions is required.

Example 也可以作为 contribution


# Repository Structure

```
<folder>
├── tensorlayer  		<--- library source code
│
├── setup.py			<--- use ‘python setup.py install’ or ‘pip install . -e‘, to install
├── docs 				<--- readthedocs folder
│   └── _build          <--- not included in the remote repo but can be generated in `docs` using `make html`
│   	 └──html
│			 └──index.html <--- homepage of the documentation
├── tutorials_*.py	 	<--- tutorials include NLP, DL, RL etc.
├── .. 
```

