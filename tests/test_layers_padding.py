import tensorflow as tf
import tensorlayer as tl

## 1D
x = tf.placeholder(tf.float32, (None, 100, 1))
n = tl.layers.InputLayer(x)
n1 = tl.layers.ZeroPad1d(n, padding=1)
n1.print_layers()
shape = n1.outputs.get_shape().as_list()
if shape[1:] != [102, 1]:
    raise Exception("shape do not match")

n2 = tl.layers.ZeroPad1d(n, padding=(2, 3))
n2.print_layers()
shape = n2.outputs.get_shape().as_list()
if shape[1:] != [105, 1]:
    raise Exception("shape do not match")

## 2D
x = tf.placeholder(tf.float32, (None, 100, 100, 3))
n = tl.layers.InputLayer(x)
n1 = tl.layers.ZeroPad2d(n, padding=2)
n1.print_layers()
shape = n1.outputs.get_shape().as_list()
if shape[1:] != [104, 104, 3]:
    raise Exception("shape do not match")

n2 = tl.layers.ZeroPad2d(n, padding=(2, 3))
n2.print_layers()
shape = n2.outputs.get_shape().as_list()
if shape[1:] != [104, 106, 3]:
    raise Exception("shape do not match")

n3 = tl.layers.ZeroPad2d(n, padding=((3, 3), (4, 4)))
n3.print_layers()
shape = n3.outputs.get_shape().as_list()
if shape[1:] != [106, 108, 3]:
    raise Exception("shape do not match")

## 3D
x = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))
n = tl.layers.InputLayer(x)
n1 = tl.layers.ZeroPad3d(n, padding=2)
n1.print_layers()
shape = n1.outputs.get_shape().as_list()
if shape[1:] != [104, 104, 104, 3]:
    raise Exception("shape do not match")

n2 = tl.layers.ZeroPad3d(n, padding=(2, 3, 4))
n2.print_layers()
shape = n2.outputs.get_shape().as_list()
if shape[1:] != [104, 106, 108, 3]:
    raise Exception("shape do not match")

n3 = tl.layers.ZeroPad3d(n, padding=((3, 3), (4, 4), (5, 5)))
n3.print_layers()
shape = n3.outputs.get_shape().as_list()
if shape[1:] != [106, 108, 110, 3]:
    raise Exception("shape do not match")
