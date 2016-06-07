# TensorLayer: Deep learning library for Tensorflow.

TensorLayer is a transparent deep learning library built on the top of *[Google Tensorflow](https://www.tensorflow.org)*. It was designed to provide a higher-level API to TensorFlow in order to speed-up experimentations. TensorLayer is easy to be extended and modified, suitable for both machine learning researches and applications.


TensorLayer features include:

- Fast prototyping through highly modular built-in neural network layers, pre-train metrices, regularizers, optimizers, cost functions...
- Implemented by straightforward code, easy to modify and extend by yourself...
- Other libraries for Tensorflow are easy to merged into TensorLayer, suitable for machine learning researches...
- Many official examples covering Dropout, DropNeuron, Autoencoder, LSTM, ResNet... are given, suitable for machine learning applications...

### Table of Contents
0. [Overview](#Overview)
0. [Easy-to-modify](#Easy-to-modify)
0. [Installation](#Installation)


## Overview
**Placeholder**

All placeholder and variables can be initialized by the same way from Tensorflow's tutorial. For details please read *[tensorflow-placeholder](https://www.tensorflow.org/versions/master/api_docs/python/io_ops.html#placeholder)* and *[tensorflow-variables](https://www.tensorflow.org/versions/master/how_tos/variables/index.html)*.

```python
# For MNIST example, 28*28 images have 784 pixels, i.e, 784 inputs.
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
```

--
**Rectifying Network with Dropout**

```python
# Define the network
network = InputLayer(x, name='input_layer')
network = DropoutLayer(network, keep=0.8, name='drop1')
network = DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
network = DropoutLayer(network, keep=0.5, name='drop2')
network = DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')
network = DropoutLayer(network, keep=0.5, name='drop3')
network = DenseLayer(network, n_units=10, act = identity, name='output_layer')
# Start training
...
```
--
**Vanilla Sparse Autoencoder**

```python
# Define the network
network = InputLayer(x, name='input_layer')
network = DenseLayer(network, n_units=196, act = tf.nn.sigmoid, name='sigmoid1')
recon_layer1 = ReconLayer(network, x_recon=x, n_units=784, act = tf.nn.sigmoid, name='recon_layer1')
# Start pre-train
recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name=None, n_epoch=200, batch_size=128, print_freq=10, save=True, save_name='w1pre_')
# Start fine-tune
...
```
--
**Denoising Autoencoder**

```python
# Define the network
network = InputLayer(x, name='input_layer')
network = DropoutLayer(network, keep=0.5, name='denoising1')   
network = DenseLayer(network, n_units=196, act = tf.nn.relu, name='relu1')
recon_layer1 = ReconLayer(network, x_recon=x, n_units=784, act = tf.nn.softplus, name='recon_layer1')
# Start pre-train
recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=200, batch_size=128, print_freq=10, save=True, save_name='w1pre_')
# Start fine-tune
...
```
--
**Stacked Denoising Autoencoders**

```python
# Define the network
network = InputLayer(x, name='input_layer')
# denoise layer for Autoencoders
network = DropoutLayer(network, keep=0.5, name='denoising1')
# 1st layer
network = DropoutLayer(network, keep=0.8, name='drop1')
network = DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
x_recon1 = network.outputs
recon_layer1 = ReconLayer(network, x_recon=x, n_units=784, act = tf.nn.softplus, name='recon_layer1')
# 2nd layer
network = DropoutLayer(network, keep=0.5, name='drop2')
network = DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')
recon_layer2 = ReconLayer(network, x_recon=x_recon1, n_units=800, act = tf.nn.softplus, name='recon_layer2')
# 3rd layer
network = DropoutLayer(network, keep=0.5, name='drop3')
network = DenseLayer(network, n_units=10, act = identity, name='output_layer')

sess.run(tf.initialize_all_variables())

# Print all parameters before pre-train
network.print_params()

# Pre-train Layer 1
recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10, save=True, save_name='w1pre_')
# Pre-train Layer 2
recon_layer2.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10, save=False)
# Start training
...
```

## Easy-to-modify
**Modifying Pre-train Behaviour**

Greedy layer-wise pretrain is an important task for deep neural network initialization, while there are many kinds of pre-train metrices according to different architectures and applications.

For example, the pre-train of *[Vanilla Sparse Autoencoder](http://deeplearning.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity)* can be implemented by using KL divergence as follow, but for *[Deep Rectifier Network](http://www.jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf)*, the sparsity can be implemented by using the L1 regularization of activation output.

```python
# Vanilla Sparse Autoencoder
beta = 4
rho = 0.15
p_hat = tf.reduce_mean(activation_out, reduction_indices = 0)   # theano: p_hat = T.mean( self.a[i], axis=0 )
KLD = beta * tf.reduce_sum( rho * tf.log(tf.div(rho, p_hat)) + (1- rho) * tf.log((1- rho)/ (tf.sub(float(1), p_hat))) )
```

For this reason, TensorLayer provides a simple way to modify or design your own pre-train metrice. For Autoencoder, TensorLayer uses **ReconLayer.*__*init__()** to define the reconstruction layer and cost function, to define your own cost function, just simply modify the **self.cost** in **ReconLayer.*__*init__()**.
        
	ReconLayer.__init__(...):
	    ...
		self.cost = mse + L1_a + L2_w
--
**Adding Customized Regularizer**

--

## Installation

**TensorFlow Installation**

This library requires Tensorflow (version >= 0.8) to be installed: *[Tensorflow installation instructions](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html)*.

--
**GPU Setup**

GPU-version of Tensorflow requires CUDA and CuDNN to be installed.

*[CUDA, CuDNN installation instructions](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#optional-install-cuda-gpus-on-linux)*.
	
*[CUDA download](https://developer.nvidia.com/cuda-downloads)*.

*[CuDNN download](https://developer.nvidia.com/cudnn)*.

--
**TensorLayer Installation**
```python
pip install git+https://github.com/xxx/xxx.git
```

Otherwise, you can also install from source by running (from source folder):

```python
python setup.py install
```
--