import tensorflow as tf
from tensorlayer.layers import SubpixelConv1d, SubpixelConv2d, InputLayer, Conv1d, Conv2d

## 1D
t_signal = tf.placeholder('float32', [10, 100, 4], name='x')
n = InputLayer(t_signal, name='in')
n = Conv1d(n, 32, 3, 1, padding='SAME', name='conv1d')
n = SubpixelConv1d(n, scale=2, name='subpixel')
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
n = InputLayer(x, name='in')
n = Conv2d(n, 32, (3, 2), (1, 1), padding='SAME', name='conv2d')
n = SubpixelConv2d(n, scale=2, name='subpixel2d')
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
