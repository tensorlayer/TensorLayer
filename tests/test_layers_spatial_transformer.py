import tensorflow as tf
import tensorlayer as tl

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])


def model(x, is_train, reuse):
    with tf.variable_scope("STN", reuse=reuse):
        nin = tl.layers.InputLayer(x, name='in')
        ## 1. Localisation network
        # use MLP as the localisation net
        nt = tl.layers.FlattenLayer(nin, name='flatten')
        nt = tl.layers.DenseLayer(nt, n_units=20, act=tf.nn.tanh, name='dense1')
        nt = tl.layers.DropoutLayer(nt, keep=0.8, is_fix=True, is_train=is_train, name='drop1')
        # you can also use CNN instead for MLP as the localisation net
        # nt = Conv2d(nin, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc1')
        # nt = Conv2d(nt, 8, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc2')
        ## 2. Spatial transformer module (sampler)
        n = tl.layers.SpatialTransformer2dAffineLayer(nin, theta_layer=nt, out_size=[40, 40], name='spatial')
        s = n
        ## 3. Classifier
        n = tl.layers.Conv2d(n, n_filter=16, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, padding='SAME', name='conv1')
        n = tl.layers.Conv2d(n, n_filter=16, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, padding='SAME', name='conv2')
        n = tl.layers.FlattenLayer(n, name='flatten2')
        n = tl.layers.DenseLayer(n, n_units=1024, act=tf.nn.relu, name='out1')
        n = tl.layers.DenseLayer(n, n_units=10, act=tf.identity, name='out2')
    return n, s


net, s = model(x, is_train=True, reuse=False)
_, _ = model(x, is_train=False, reuse=True)

net.print_layers()
net.print_params(False)

shape = s.outputs.get_shape().as_list()
if shape[1:] != [40, 40, 1]:
    raise Exception("shape do not match")

if len(net.all_layers) != 9:
    raise Exception("layers do not match")

if len(net.all_params) != 12:
    raise Exception("params do not match")

if net.count_params() != 1667980:
    raise Exception("params do not match")
