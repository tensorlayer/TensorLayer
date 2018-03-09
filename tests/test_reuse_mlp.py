import time
import tensorflow as tf
import tensorlayer as tl

sess = tf.InteractiveSession()

# prepare data
X_train, y_train, X_val, y_val, X_test, y_test = \
                                tl.files.load_mnist_dataset(shape=(-1,784))
# define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None], name='y_')


# define the network
def mlp(x, is_train=True, reuse=False):
    with tf.variable_scope("MLP", reuse=reuse):
        tl.layers.set_name_reuse(reuse)  # print warning
        network = tl.layers.InputLayer(x, name='input')
        network = tl.layers.DropoutLayer(network, keep=0.8, is_fix=True, is_train=is_train, name='drop1')
        network = tl.layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu1')
        network = tl.layers.DropoutLayer(network, keep=0.5, is_fix=True, is_train=is_train, name='drop2')
        network = tl.layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu2')
        network = tl.layers.DropoutLayer(network, keep=0.5, is_fix=True, is_train=is_train, name='drop3')
        network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output')
    return network


# define inferences
net_train = mlp(x, is_train=True, reuse=False)
net_test = mlp(x, is_train=False, reuse=True)

try:
    is_except = False
    net_test = mlp(x, is_train=False, reuse=False)
except Exception as e:
    is_except = True
    print(e)

if is_except == False:
    raise Exception("it should not be success")
