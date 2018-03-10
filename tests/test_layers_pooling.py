import tensorflow as tf
import tensorlayer as tl

## 1D ========================================================================
x = tf.placeholder(tf.float32, (None, 100, 1))
nin = tl.layers.InputLayer(x, name='in1')
nin = tl.layers.Conv1d(nin, 32, 5, 2, name='conv1d')
print(nin)
shape = nin.outputs.get_shape().as_list()
if (shape[1] != 50) or (shape[2] != 32):
    raise Exception("shape dont match")

n = tl.layers.MaxPool1d(nin, 3, 2, 'same', name='maxpool1d')
print(n)
shape = n.outputs.get_shape().as_list()
# print(shape[1:3])
if shape[1:3] != [25, 32]:
    raise Exception("shape dont match")

n = tl.layers.MeanPool1d(nin, 3, 2, 'same', name='meanpool1d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape[1:3] != [25, 32]:
    raise Exception("shape dont match")

n = tl.layers.GlobalMaxPool1d(nin, name='maxpool1d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape[-1] != 32:
    raise Exception("shape dont match")

n = tl.layers.GlobalMeanPool1d(nin, name='meanpool1d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape[-1] != 32:
    raise Exception("shape dont match")

## 2D ========================================================================
x = tf.placeholder(tf.float32, (None, 100, 100, 3))
nin = tl.layers.InputLayer(x, name='in2')
nin = tl.layers.Conv2d(nin, 32, (3, 3), (2, 2), name='conv2d')
print(nin)
shape = nin.outputs.get_shape().as_list()
if (shape[1] != 50) or (shape[2] != 50) or (shape[3] != 32):
    raise Exception("shape dont match")

n = tl.layers.MaxPool2d(nin, (3, 3), (2, 2), 'SAME', name='maxpool2d')
print(n)
shape = n.outputs.get_shape().as_list()
# print(shape[1:3])
if shape[1:4] != [25, 25, 32]:
    raise Exception("shape dont match")

n = tl.layers.MeanPool2d(nin, (3, 3), (2, 2), 'SAME', name='meanpool2d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape[1:4] != [25, 25, 32]:
    raise Exception("shape dont match")

n = tl.layers.GlobalMaxPool2d(nin, name='maxpool2d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape[-1] != 32:
    raise Exception("shape dont match")

n = tl.layers.GlobalMeanPool2d(nin, name='meanpool2d')
print(n)
shape = n.outputs.get_shape().as_list()
if shape[-1] != 32:
    raise Exception("shape dont match")
