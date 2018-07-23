"""
This tutorial assumes you run a MongoDB in localhost.

Install and run MongoDB on Mac : https://gist.github.com/subfuzion/9630872
"""

import time
import tensorflow as tf
import tensorlayer as tl
import pickle

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

sess = tf.InteractiveSession()

# prepare data
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
# define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

# define the network
def mlp(x, is_train=True, reuse=False):
    with tf.variable_scope("MLP", reuse=reuse):
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.DropoutLayer(net, keep=0.8, is_fix=True, is_train=is_train, name='drop1')
        net = tl.layers.DenseLayer(net, n_units=800, act=tf.nn.relu, name='relu1')
        net = tl.layers.DropoutLayer(net, keep=0.5, is_fix=True, is_train=is_train, name='drop2')
        net = tl.layers.DenseLayer(net, n_units=800, act=tf.nn.relu, name='relu2')
        net = tl.layers.DropoutLayer(net, keep=0.5, is_fix=True, is_train=is_train, name='drop3')
        net = tl.layers.DenseLayer(net, n_units=10, act=None, name='output')
    return net

# define inferences
net_train = mlp(x, is_train=True, reuse=False)
net_test = mlp(x, is_train=False, reuse=True)

# cost for training
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')

# cost and accuracy for evalution
y2 = net_test.outputs
cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# define the optimizer
train_params = tl.layers.get_variables_with_name('MLP', True, False)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

def save_graph(network=None, name='graph.pkl'):
    """Save the graph of TL model into pickle file.

    Examples
    --------
    >>>
    """
    graphs = network.all_graphs
    with open(name, 'wb') as file:
        return pickle.dump(graphs, file, protocol=pickle.HIGHEST_PROTOCOL)

from tensorlayer.layers.utils import list_remove_repeat
from tensorlayer import tl_logging as logging
def load_graph(name='model.pkl'):
    """Restore TL model from a graph file, return TL model.

    Returns
    --------
    network : TensorLayer layer
        The input placeholder are attributes of the returned TL layer object.

    Examples
    --------
    - see ``save_graph``
    """
    logging.info("Loading TL graph from {}".format(name))
    with open(name, 'rb') as file:
        graphs = pickle.load(file)

    input_list = list()
    input_dict = dict()
    layer_dict = dict()
    ## loop every layers
    for graph in graphs:
        ## get current layer class
        name, layer_kwargs = graph
        layer_class = layer_kwargs.pop('class')     # class of current layer
        prev_layer = layer_kwargs.pop('prev_layer') # name of previous layer
        ## convert activation from string to function
        try:
            act = layer_kwargs.pop('act')
            # print(dir(tl.act))
            if act in dir(tl.act):
                layer_kwargs.update({'act': eval('tl.act.'+act)})
            else:
                layer_kwargs.update({'act': eval('tf.nn.'+act)})

        except Exception as e: # no act
            pass
            # exit(e)

        print(name, prev_layer, layer_class, layer_kwargs)

        if layer_class == 'placeholder': ## create placeholder
            dtype = layer_kwargs.pop('dtype')
            shape = layer_kwargs.pop('shape')
            _placeholder = tf.placeholder(tf.float32, shape, name=name.split(':')[0]) #globals()['tf.'+dtype]
            # input_dict.update({name: _placeholder})
            input_list.append((name, _placeholder))
        else:   ## create network
            try:    # if previous layer is layer
                net = layer_dict[prev_layer]
                layer_kwargs.update({'prev_layer': net})
            except: # if previous layer is input placeholder
                for n, t in input_list:
                    if n == prev_layer:
                        _placeholder = t
                layer_kwargs.update({'inputs': _placeholder})
                # net = globals()['tl.layers.'+layer_class](layer_kwargs)
            layer_kwargs.update({'name': name})
            print(layer_kwargs)
            net = eval('tl.layers.'+layer_class)(**layer_kwargs)
            layer_dict.update({name: net})

    # for key in input_dict: # set input placeholder into the lastest layer
    #     layer_dict[name].globals()[key] = input_dict[key]
    #     logging.info("  attributes: {:3} {:15} {:15}".format(n, input_dict[key].get_shape().as_list(), input_dict[key].dtype.name))
    return layer_dict[name]

save_graph(net_test, 'graph.pkl')
with tf.Graph().as_default() as graph:
    net = load_graph('graph.pkl')

import os
def save_graph_params(network=None, name='model', sess=None):
    os.mkdir(name)
    save_graph(network, os.join.path(name, 'graph.pkl'))

def load_graph_params():
    pass
