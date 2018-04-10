import tensorflow as tf
import tensorlayer as tl

## 1D ========================================================================
x = tf.placeholder(tf.float32, (None, 100, 1))
nin = tl.layers.InputLayer(x, name='in1')
nin = tl.layers.Conv1d(nin, n_filter=32, filter_size=5, stride=2, name='conv1d')
print(nin)
shape = nin.outputs.get_shape().as_list()
if (shape[1] != 50) or (shape[2] != 32):
    raise Exception("shape do not match")

n = tl.layers.MaxPool1d(nin, filter_size=3, strides=2, padding='same', name='maxpool1d')
print(n)
shape = n.outputs.get_shape().as_list()
# print(shape[1:3])
if shape[1:3] != [25, 32]:
    raise Exception("shape do not match")

n = tl.layers.MeanPool1d(nin, filter_size=3, strides=2, padding='same', name='meanpool1d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape[1:3] != [25, 32]:
    raise Exception("shape do not match")

n = tl.layers.GlobalMaxPool1d(nin, name='maxpool1d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape[-1] != 32:
    raise Exception("shape do not match")

n = tl.layers.GlobalMeanPool1d(nin, name='meanpool1d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape[-1] != 32:
    raise Exception("shape do not match")

## 2D ========================================================================
x = tf.placeholder(tf.float32, (None, 100, 100, 3))
nin = tl.layers.InputLayer(x, name='in2')
nin = tl.layers.Conv2d(nin, n_filter=32, filter_size=(3, 3), strides=(2, 2), name='conv2d')
print(nin)
shape = nin.outputs.get_shape().as_list()
if (shape[1] != 50) or (shape[2] != 50) or (shape[3] != 32):
    raise Exception("shape do not match")

n = tl.layers.MaxPool2d(nin, filter_size=(3, 3), strides=(2, 2), padding='SAME', name='maxpool2d')
print(n)
shape = n.outputs.get_shape().as_list()
# print(shape[1:3])
if shape[1:4] != [25, 25, 32]:
    raise Exception("shape do not match")

n = tl.layers.MeanPool2d(nin, filter_size=(3, 3), strides=(2, 2), padding='SAME', name='meanpool2d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape[1:4] != [25, 25, 32]:
    raise Exception("shape do not match")

n = tl.layers.GlobalMaxPool2d(nin, name='maxpool2d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape[-1] != 32:
    raise Exception("shape do not match")

n = tl.layers.GlobalMeanPool2d(nin, name='meanpool2d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape[-1] != 32:
    raise Exception("shape do not match")

## 3D ========================================================================
x = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))
nin = tl.layers.InputLayer(x, name='in')

n = tl.layers.MeanPool3d(nin, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME', name='meanpool3d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape != [None, 50, 50, 50, 3]:
    raise Exception("shape do not match")

n = tl.layers.GlobalMaxPool3d(nin)
print(n)
shape = n.outputs.get_shape().as_list()
if shape != [None, 3]:
    raise Exception("shape do not match")

n = tl.layers.GlobalMeanPool3d(nin)
print(n)
shape = n.outputs.get_shape().as_list()
if shape != [None, 3]:
    raise Exception("shape do not match")
