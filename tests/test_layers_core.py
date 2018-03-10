import tensorflow as tf
import tensorlayer as tl

## DenseLayer
x = tf.placeholder(tf.float32, shape=[None, 30])
net = tl.layers.InputLayer(x, name='input')
net = tl.layers.DenseLayer(net, 10, name='dense')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape[-1] != 10:
    raise Exception("shape dont match")

if len(net.all_layers) != 1:
    raise Exception("layers dont match")

if len(net.all_params) != 2:
    raise Exception("params dont match")

if net.count_params() != 310:
    raise Exception("params dont match")

## OneHotInputLayer
x = tf.placeholder(tf.int32, shape=[None])
net = tl.layers.OneHotInputLayer(x, depth=8, name='onehot')
print(net)

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape[-1] != 8:
    raise Exception("shape dont match")

if len(net.all_layers) != 0:
    raise Exception("layers dont match")

if len(net.all_params) != 0:
    raise Exception("params dont match")

if net.count_params() != 0:
    raise Exception("params dont match")

## Word2vecEmbeddingInputlayer
batch_size = 8
train_inputs = tf.placeholder(tf.int32, shape=(batch_size))
train_labels = tf.placeholder(tf.int32, shape=(batch_size, 1))
net = tl.layers.Word2vecEmbeddingInputlayer(
    inputs=train_inputs, train_labels=train_labels, vocabulary_size=1000, embedding_size=200, num_sampled=64, name='word2vec')
cost = net.nce_cost
train_params = net.all_params

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape != [8, 200]:
    raise Exception("shape dont match")

if len(net.all_layers) != 1:
    raise Exception("layers dont match")

if len(net.all_params) != 3:
    raise Exception("params dont match")

if net.count_params() != 401000:
    raise Exception("params dont match")

## EmbeddingInputlayer
batch_size = 8
x = tf.placeholder(tf.int32, shape=(batch_size, ))
net = tl.layers.EmbeddingInputlayer(inputs=x, vocabulary_size=1000, embedding_size=50, name='embed')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape != [batch_size, 50]:  # (8, 50)
    raise Exception("shape dont match")

if len(net.all_layers) != 1:
    raise Exception("layers dont match")

if len(net.all_params) != 1:
    raise Exception("params dont match")

if net.count_params() != 50000:
    raise Exception("params dont match")

## AverageEmbeddingInputlayer
batch_size = 8
length = 5
x = tf.placeholder(tf.int32, shape=(batch_size, length))
net = tl.layers.AverageEmbeddingInputlayer(inputs=x, vocabulary_size=1000, embedding_size=50, name='avg')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape != [batch_size, 50]:  # (8, 50)
    raise Exception("shape dont match")

if len(net.all_layers) != 1:
    raise Exception("layers dont match")

if len(net.all_params) != 1:
    raise Exception("params dont match")

if net.count_params() != 50000:
    raise Exception("params dont match")

## ReconLayer
x = tf.placeholder(tf.float32, shape=(None, 784))
net = tl.layers.InputLayer(x, name='input')
net = tl.layers.DenseLayer(net, n_units=196, act=tf.nn.sigmoid, name='dense2')
net = tl.layers.ReconLayer(net, x_recon=x, n_units=784, act=tf.nn.sigmoid, name='recon')
# sess = tf.InteractiveSession()
# tl.layers.initialize_global_variables(sess)
# X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
# net.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name=None, n_epoch=1, batch_size=128, print_freq=1, save=True, save_name='w1pre_')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape[-1] != 784:
    raise Exception("shape dont match")

if len(net.all_layers) != 2:
    raise Exception("layers dont match")

if len(net.all_params) != 4:
    raise Exception("params dont match")

if net.count_params() != 308308:
    raise Exception("params dont match")

## GaussianNoiseLayer
x = tf.placeholder(tf.float32, shape=(64, 784))
net = tl.layers.InputLayer(x, name='input')
net = tl.layers.DenseLayer(net, n_units=100, act=tf.nn.relu, name='dense3')
net = tl.layers.GaussianNoiseLayer(net, name='gaussian')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape != [64, 100]:
    raise Exception("shape dont match")

if len(net.all_layers) != 2:
    raise Exception("layers dont match")

if len(net.all_params) != 2:
    raise Exception("params dont match")

if net.count_params() != 78500:
    raise Exception("params dont match")

## DropconnectDenseLayer
x = tf.placeholder(tf.float32, shape=(64, 784))
net = tl.layers.InputLayer(x, name='input')
net = tl.layers.DenseLayer(net, n_units=100, act=tf.nn.relu, name='dense4')
net = tl.layers.DropconnectDenseLayer(net, keep=0.8, name='dropconnect')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape != [64, 100]:
    raise Exception("shape dont match")

if len(net.all_layers) != 2:
    raise Exception("layers dont match")

if len(net.all_params) != 2:
    raise Exception("params dont match")

if net.count_params() != 78500:
    raise Exception("params dont match")
