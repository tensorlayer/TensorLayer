import tensorflow as tf
import tensorlayer as tl

## 1D
t_signal = tf.placeholder('float32', [10, 100, 4], name='x')
n = tl.layers.InputLayer(t_signal, name='in')
n = tl.layers.Conv1d(n, n_filter=32, filter_size=3, stride=1, padding='SAME', name='conv1d')
n = tl.layers.SubpixelConv1d(n, scale=2, name='subpixel')
print(n.outputs.shape)
# ... (10, 200, 2)
n.print_layers()
n.print_params(False)

shape = n.outputs.get_shape().as_list()
if shape != [10, 200, 16]:
    raise Exception("shape do not match")

if len(n.all_layers) != 2:
    raise Exception("layers do not match")

if len(n.all_params) != 2:
    raise Exception("params do not match")

if n.count_params() != 416:
    raise Exception("params do not match")

## 2D
x = tf.placeholder('float32', [10, 100, 100, 3], name='x')
n = tl.layers.InputLayer(x, name='in')
n = tl.layers.Conv2d(n, n_filter=32, filter_size=(3, 2), strides=(1, 1), padding='SAME', name='conv2d')
n = tl.layers.SubpixelConv2d(n, scale=2, name='subpixel2d')
print(n.outputs.shape)

n.print_layers()
n.print_params(False)

shape = n.outputs.get_shape().as_list()
if shape != [10, 200, 200, 8]:
    raise Exception("shape do not match")

if len(n.all_layers) != 2:
    raise Exception("layers do not match")

if len(n.all_params) != 2:
    raise Exception("params do not match")

if n.count_params() != 608:
    raise Exception("params do not match")
