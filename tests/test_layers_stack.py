import tensorflow as tf
import tensorlayer as tl

x = tf.placeholder(tf.float32, shape=[None, 30])
net = tl.layers.InputLayer(x, name='input')
net1 = tl.layers.DenseLayer(net, 10, name='dense1')
net2 = tl.layers.DenseLayer(net, 10, name='dense2')
net3 = tl.layers.DenseLayer(net, 10, name='dense3')
net = tl.layers.StackLayer([net1, net2, net3], axis=1, name='stack')

net.print_layers()
net.print_params(False)

shape = net.outputs.get_shape().as_list()
if shape[-1] != 10:
    raise Exception("shape dont match")

if len(net.all_layers) != 4:
    raise Exception("layers dont match")

if len(net.all_params) != 6:
    raise Exception("params dont match")

if net.count_params() != 930:
    raise Exception("params dont match")

net = tl.layers.UnStackLayer(net, axis=1, name='unstack')
for n in net:
    print(n, n.outputs)
    shape = n.outputs.get_shape().as_list()
    if shape[-1] != 10:
        raise Exception("shape dont match")

    # n.print_layers()
    # n.print_params(False)

    if len(n.all_layers) != 4:
        raise Exception("layers dont match")

    if len(n.all_params) != 6:
        raise Exception("params dont match")

    if n.count_params() != 930:
        raise Exception("params dont match")
