import tensorflow as tf
import tensorlayer as tl

## 1D
x = tf.placeholder(tf.float32, (None, 100, 1))
nin = tl.layers.InputLayer(x, name='in1')

n = tl.layers.Conv1dLayer(nin, shape=(5, 1, 32), stride=2)
shape = n.outputs.get_shape().as_list()
if (shape[1] != 50) or (shape[2] != 32):
    raise Exception("shape do not match")

n = tl.layers.Conv1d(nin, n_filter=32, filter_size=5, stride=2)
print(n)
shape = n.outputs.get_shape().as_list()
if (shape[1] != 50) or (shape[2] != 32):
    raise Exception("shape do not match")

# AtrousConv1dLayer

## 2D
x = tf.placeholder(tf.float32, (None, 100, 100, 3))
nin = tl.layers.InputLayer(x, name='in2')
n = tl.layers.Conv2dLayer(
    nin,
    act=tf.nn.relu,
    shape=(5, 5, 3, 32),
    strides=(1, 2, 2, 1),
    padding='SAME',
    W_init=tf.truncated_normal_initializer(stddev=5e-2),
    b_init=tf.constant_initializer(value=0.0),
    name='conv2dlayer')
print(n)
shape = n.outputs.get_shape().as_list()
if (shape[1] != 50) or (shape[2] != 50) or (shape[3] != 32):
    raise Exception("shape do not match")

n = tl.layers.Conv2d(nin, n_filter=32, filter_size=(3, 3), strides=(2, 2), act=None, name='conv2d')
shape = n.outputs.get_shape().as_list()
if (shape[1] != 50) or (shape[2] != 50) or (shape[3] != 32):
    raise Exception("shape do not match")
n.print_params(False)
if len(n.all_params) != 2:
    raise Exception("params do not match")

n = tl.layers.Conv2d(nin, n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, b_init=None, name='conv2d_no_bias')
print(n)
shape = n.outputs.get_shape().as_list()
if (shape[1] != 50) or (shape[2] != 50) or (shape[3] != 32):
    raise Exception("shape do not match")
if len(n.all_params) != 1:
    raise Exception("params do not match")

n = tl.layers.DeConv2dLayer(nin, shape=(5, 5, 32, 3), output_shape=(100, 200, 200, 32), strides=(1, 2, 2, 1), name='deconv2dlayer')
print(n)
shape = n.outputs.get_shape().as_list()
if (shape[1] != 200) or (shape[2] != 200) or (shape[3] != 32):
    raise Exception("shape do not match")

print(nin.outputs)
n = tl.layers.DeConv2d(nin, n_filter=32, filter_size=(3, 3), strides=(2, 2), name='DeConv2d')
print(n)
shape = n.outputs.get_shape().as_list()
print(shape[1:])
# if (shape[1] != 200) or (shape[2] != 200) or (shape[3] != 32): # TODO: why [None None None 32] ?
if (shape[1:] != [200, 200, 32]):
    raise Exception("shape do not match")
n = tl.layers.DeConv2d(n, n_filter=16, filter_size=(3, 3), strides=(2, 2), name='DeConv2d_2')
print(n)
shape = n.outputs.get_shape().as_list()
if (shape[1:] != [400, 400, 16]):
    raise Exception("shape do not match")
if len(n.all_params) != 4:
    raise Exception("params do not match")

n = tl.layers.DepthwiseConv2d(nin, shape=(3, 3), strides=(2, 2), act=tf.nn.relu, depth_multiplier=2, name='depthwise')
print(n)
shape = n.outputs.get_shape().as_list()
if (shape[1] != 50) or (shape[2] != 50) or (shape[3] != 6):
    raise Exception("shape do not match")

n = tl.layers.Conv2d(nin, n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, name='conv2d2')
n = tl.layers.GroupConv2d(n, n_filter=32, filter_size=(3, 3), strides=(2, 2), name='group')
print(n)
shape = n.outputs.get_shape().as_list()
if (shape[1] != 25) or (shape[2] != 25) or (shape[3] != 32):
    raise Exception("shape do not match")

# n = UpSampling2dLayer
# n = DownSampling2dLayer

# offset1 = tl.layers.Conv2d(nin, 18, (3, 3), (1, 1), padding='SAME', name='offset1')
# net = tl.layers.DeformableConv2d(nin, offset1, 32, (3, 3), name='deformable1')
# offset2 = tl.layers.Conv2d(net, 18, (3, 3), (1, 1), padding='SAME', name='offset2')
# net = tl.layers.DeformableConv2d(net, offset2, 64, (3, 3), name='deformable2')
# net.print_layers()
# net.print_params(False)

# AtrousConv2dLayer

n = tl.layers.SeparableConv2d(nin, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, name='seperable1')
n.print_layers()
n.print_params(False)

shape = n.outputs.get_shape().as_list()
if shape[1:] != [98, 98, 32]:
    raise Exception("shape do not match")

if len(n.all_layers) != 1:
    raise Exception("layers do not match")

if len(n.all_params) != 3:
    raise Exception("params do not match")

if n.count_params() != 155:
    raise Exception("params do not match")
# exit()

## 3D
x = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))
nin = tl.layers.InputLayer(x, name='in3')

n = tl.layers.Conv3dLayer(nin, shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1))
print(n)
shape = n.outputs.get_shape().as_list()
if (shape[1] != 50) or (shape[2] != 50) or (shape[3] != 50) or (shape[4] != 32):
    raise Exception("shape do not match")

# n = tl.layers.DeConv3dLayer(nin, shape=(2, 2, 2, 128, 3), output_shape=(100, 12, 32, 32, 128), strides=(1, 2, 2, 2, 1))
# print(n)
# shape = n.outputs.get_shape().as_list()

n = tl.layers.DeConv3d(nin, n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2))
shape = n.outputs.get_shape().as_list()
print(shape)
if (shape[1] != 200) or (shape[2] != 200) or (shape[3] != 200) or (shape[4] != 32):
    raise Exception("shape do not match")
