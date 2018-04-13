import tensorflow as tf
import tensorlayer as tl

x = tf.placeholder(tf.float32, [None, 100])
n = tl.layers.InputLayer(x, name='in')
n = tl.layers.DenseLayer(n, n_units=80, name='d1')
n = tl.layers.DenseLayer(n, n_units=80, name='d2')
print(n)
n.print_layers()
n.print_params(False)
print(n.count_params())

if n.count_params() != 14560:
    raise Exception("params do not match")

shape = n.outputs.get_shape().as_list()
if shape[-1] != 80:
    raise Exception("shape do not match")

if len(n.all_layers) != 2:
    raise Exception("layers do not match")

if len(n.all_params) != 4:
    raise Exception("params do not match")

for l in n:
    print(l)

n2 = n[:, :30]
print(n2)
n2.print_layers()

shape = n2.outputs.get_shape().as_list()
if shape[-1] != 30:
    raise Exception("shape do not match")

for l in n2:
    print(l)
