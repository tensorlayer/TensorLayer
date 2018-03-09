import tensorflow as tf
from tensorlayer.layers import *

x = tf.placeholder("float32", [None, 100, 30])
n = InputLayer(x, name='in1')
n = GlobalMaxPool1d(n)
print(n)

x = tf.placeholder("float32", [None, 100, 100, 30])
n = InputLayer(x, name='in2')
n = GlobalMaxPool2d(n)
print(n)

x = tf.placeholder("float32", [None, 100, 100, 100, 30])
n = InputLayer(x, name='in3')
n = MaxPool3d(n)
n.print_layers()
print(n)

x = tf.placeholder("float32", [None, 100, 100, 100, 30])
n = InputLayer(x, name='in4')
n = MeanPool3d(n)
n.print_layers()
print(n)
