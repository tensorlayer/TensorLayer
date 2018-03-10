import tensorflow as tf
import tensorlayer as tl


def model(x, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        n = tl.layers.InputLayer(x, name='in')
        n = tl.layers.Conv2d(n, 80, name='conv2d_1')
        n = tl.layers.BatchNormLayer(n, name='norm_batch')
        n = tl.layers.Conv2d(n, 80, name='conv2d_2')
        n = tl.layers.LocalResponseNormLayer(n, name='norm_local')
        n = tl.layers.LayerNormLayer(n, name='norm_layer')
        n = tl.layers.InstanceNormLayer(n, name='norm_instance')
    return n


x = tf.placeholder(tf.float32, [None, 100, 100, 3])
net = model(x, True, False)
_ = model(x, False, True)

net.print_layers()
net.print_params(False)

if len(net.all_layers) != 6:
    raise Exception("layers dont match")

if len(net.all_params) != 12:
    raise Exception("params dont match")

if net.count_params() != 60560:
    raise Exception("params dont match")
