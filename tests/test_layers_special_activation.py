import tensorflow as tf
import tensorlayer as tl

x = tf.placeholder(tf.float32, shape=[None, 30])
net = tl.layers.InputLayer(x, name='input')
net = tl.layers.DenseLayer(net, n_units=10, name='dense')
net = tl.layers.PReluLayer(net, name='prelu')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape[-1] != 10:
    raise Exception("shape do not match")

if len(net.all_layers) != 2:
    raise Exception("layers do not match")

if len(net.all_params) != 3:
    raise Exception("params do not match")

if net.count_params() != 320:
    raise Exception("params do not match")

net = tl.layers.PReluLayer(net, channel_shared=True, name='prelu2')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape[-1] != 10:
    raise Exception("shape do not match")

if len(net.all_layers) != 3:
    raise Exception("layers do not match")

if len(net.all_params) != 4:
    raise Exception("params do not match")

if net.count_params() != 321:
    raise Exception("params do not match")
