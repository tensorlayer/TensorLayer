import tensorflow as tf
import tensorlayer as tl

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
net = tl.layers.InputLayer(x, name='input')

## Flatten
net = tl.layers.FlattenLayer(net, name='flatten')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape[-1] != 784:
    raise Exception("shape do not match")

if len(net.all_layers) != 1:
    raise Exception("layers do not match")

if len(net.all_params) != 0:
    raise Exception("params do not match")

if net.count_params() != 0:
    raise Exception("params do not match")

## Reshape
net = tl.layers.ReshapeLayer(net, shape=(-1, 28, 28, 1), name='reshape')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape[1:] != [28, 28, 1]:
    raise Exception("shape do not match")

if len(net.all_layers) != 2:
    raise Exception("layers do not match")

if len(net.all_params) != 0:
    raise Exception("params do not match")

if net.count_params() != 0:
    raise Exception("params do not match")

## TransposeLayer
net = tl.layers.TransposeLayer(net, perm=[0, 1, 3, 2], name='trans')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape[1:] != [28, 1, 28]:
    raise Exception("shape do not match")

if len(net.all_layers) != 3:
    raise Exception("layers do not match")

if len(net.all_params) != 0:
    raise Exception("params do not match")

if net.count_params() != 0:
    raise Exception("params do not match")
