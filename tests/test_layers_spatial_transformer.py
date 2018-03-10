import tensorflow as tf
from tensorlayer.layers import InputLayer, FlattenLayer, DenseLayer, DropoutLayer, SpatialTransformer2dAffineLayer, Conv2d

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])


def model(x, is_train, reuse):
    with tf.variable_scope("STN", reuse=reuse):
        nin = InputLayer(x, name='in')
        ## 1. Localisation network
        # use MLP as the localisation net
        nt = FlattenLayer(nin, name='flatten')
        nt = DenseLayer(nt, n_units=20, act=tf.nn.tanh, name='dense1')
        nt = DropoutLayer(nt, 0.8, True, is_train, name='drop1')
        # you can also use CNN instead for MLP as the localisation net
        # nt = Conv2d(nin, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc1')
        # nt = Conv2d(nt, 8, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc2')
        ## 2. Spatial transformer module (sampler)
        n = SpatialTransformer2dAffineLayer(nin, nt, out_size=[40, 40], name='spatial')
        s = n
        ## 3. Classifier
        n = Conv2d(n, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='conv1')
        n = Conv2d(n, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='conv2')
        n = FlattenLayer(n, name='flatten2')
        n = DenseLayer(n, n_units=1024, act=tf.nn.relu, name='out1')
        n = DenseLayer(n, n_units=10, act=tf.identity, name='out2')
    return n, s


net, s = model(x, is_train=True, reuse=False)
_, _ = model(x, is_train=False, reuse=True)

net.print_layers()
net.print_params(False)

shape = s.outputs.get_shape().as_list()
if shape[1:] != [40, 40, 1]:
    raise Exception("shape dont match")

if len(net.all_layers) != 9:
    raise Exception("layers dont match")

if len(net.all_params) != 12:
    raise Exception("params dont match")

if net.count_params() != 1667980:
    raise Exception("params dont match")
