import tensorflow as tf
import tensorlayer as tl

sess = tf.InteractiveSession()

## vector
x = tf.placeholder(tf.float32, shape=[None, 784])
inputs = tl.layers.InputLayer(x, name='input_layer')
net1 = tl.layers.DenseLayer(inputs, 100, act=tf.nn.relu, name='relu1_1')
net2 = tl.layers.DenseLayer(inputs, 100, act=tf.nn.relu, name='relu2_1')
net = tl.layers.ConcatLayer([net1, net2], 1, name='concat_layer')

net.print_params(False)
net.print_layers()

if len(net.all_layers) != 3:
    raise Exception("layers dont match")

if len(net.all_params) != 4:
    raise Exception("params dont match")

if net.count_params() != 157000:
    raise Exception("params dont match")

net_0 = tl.layers.DenseLayer(inputs, n_units=100, act=tf.nn.relu, name='net_0')
net_1 = tl.layers.DenseLayer(inputs, n_units=100, act=tf.nn.relu, name='net_1')
net = tl.layers.ElementwiseLayer([net_0, net_1], combine_fn=tf.minimum, name='minimum')

net.print_params(False)
net.print_layers()

if len(net.all_layers) != 3:
    raise Exception("layers dont match")

if len(net.all_params) != 4:
    raise Exception("params dont match")

if net.count_params() != 157000:
    raise Exception("params dont match")

## image
x = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
inputs = tl.layers.InputLayer(x, name='input')
net1 = tl.layers.Conv2d(inputs, 32, (3, 3), (2, 2), act=tf.nn.relu, name='c1')
net2 = tl.layers.Conv2d(inputs, 32, (3, 3), (2, 2), act=tf.nn.relu, name='c2')
net = tl.layers.ConcatLayer([net1, net2], -1, name='concat')

net.print_params(False)
net.print_layers()

shape = net.outputs.get_shape().as_list()
if shape[1:] != [50, 50, 64]:
    raise Exception("shape dont match")

if len(net.all_layers) != 3:
    raise Exception("layers dont match")

if len(net.all_params) != 4:
    raise Exception("params dont match")

if net.count_params() != 1792:
    raise Exception("params dont match")

net = tl.layers.ElementwiseLayer([net1, net2], combine_fn=tf.minimum, name='minimum2')
net.print_params(False)
net.print_layers()

shape = net.outputs.get_shape().as_list()
if shape[1:] != [50, 50, 32]:
    raise Exception("shape dont match")

if len(net.all_layers) != 3:
    raise Exception("layers dont match")

if len(net.all_params) != 4:
    raise Exception("params dont match")

if net.count_params() != 1792:
    raise Exception("params dont match")
