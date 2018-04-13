import tensorflow as tf
import tensorlayer as tl

## 1D
x = tf.placeholder(tf.float32, (None, 100))
n = tl.layers.InputLayer(x, name='in')
n = tl.layers.DenseLayer(n, n_units=100, name='d1')
n = tl.layers.DenseLayer(n, n_units=100, name='d2')

n = tl.layers.ExpandDimsLayer(n, axis=2)
print(n)
shape = n.outputs.get_shape().as_list()
if shape[-1] != 1:
    raise Exception("shape do not match")

n = tl.layers.TileLayer(n, multiples=[-1, 1, 3])
print(n)
shape = n.outputs.get_shape().as_list()
if shape[-1] != 3:
    raise Exception("shape do not match")

n.print_layers()
n.print_params(False)
# print(n.all_layers, n.all_params)
if len(n.all_layers) != 4:
    raise Exception("layers do not match")
if len(n.all_params) != 4:
    raise Exception("params do not match")
