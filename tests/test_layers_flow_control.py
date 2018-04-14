import tensorflow as tf
import tensorlayer as tl

x = tf.placeholder(tf.float32, shape=(None, 784), name='x')

# define the network
net_in = tl.layers.InputLayer(x, name='in')
net_in = tl.layers.DropoutLayer(net_in, keep=0.8, name='in/drop')
# net 0
net_0 = tl.layers.DenseLayer(net_in, n_units=800, act=tf.nn.relu, name='net0/relu1')
net_0 = tl.layers.DropoutLayer(net_0, keep=0.5, name='net0/drop1')
net_0 = tl.layers.DenseLayer(net_0, n_units=800, act=tf.nn.relu, name='net0/relu2')
# net 1
net_1 = tl.layers.DenseLayer(net_in, n_units=800, act=tf.nn.relu, name='net1/relu1')
net_1 = tl.layers.DropoutLayer(net_1, keep=0.8, name='net1/drop1')
net_1 = tl.layers.DenseLayer(net_1, n_units=800, act=tf.nn.relu, name='net1/relu2')
net_1 = tl.layers.DropoutLayer(net_1, keep=0.8, name='net1/drop2')
net_1 = tl.layers.DenseLayer(net_1, n_units=800, act=tf.nn.relu, name='net1/relu3')
# multiplexer
net_mux = tl.layers.MultiplexerLayer(layers=[net_0, net_1], name='mux')
network = tl.layers.ReshapeLayer(net_mux, shape=(-1, 800), name='reshape')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
# output layer
network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output')

network.print_layers()
network.print_params(False)

if len(network.all_params) != 12:
    raise Exception("params do not match")

if len(network.all_layers) != 13:
    raise Exception("layers do not match")

if len(network.all_drop) != 5:
    raise Exception("drop do not match")
