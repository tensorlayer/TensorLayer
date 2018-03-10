import tensorflow as tf
from tensorlayer.layers import InputLayer, TimeDistributedLayer, DenseLayer

sess = tf.InteractiveSession()
batch_size = 32
timestep = 20
input_dim = 100

## no reuse
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, timestep, input_dim], name="encode_seqs")
net = InputLayer(x, name='input')
net = TimeDistributedLayer(net, layer_class=DenseLayer, args={'n_units': 50, 'name': 'dense'}, name='time_dense')

if net.outputs.get_shape().as_list() != [32, 20, 50]:
    raise Exception("shape dont match")
# ... (32, 20, 50)
net.print_params(False)
if net.count_params() != 5050:
    raise Exception("params dont match")


## reuse
def model(x, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        net = InputLayer(x, name='input')
        net = TimeDistributedLayer(net, layer_class=DenseLayer, args={'n_units': 50, 'name': 'dense'}, name='time_dense')
    return net


net_train = model(x, is_train=True, reuse=False)
net_test = model(x, is_train=False, reuse=True)
