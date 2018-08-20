# Add REFLECT padding to input tensor
from tensorlayer.layers import *
import tensorflow as tf

class ReflectPaddingLayer(Layer):
    def __init__(
        self,
        prev_layer = None,
        name ='ReflectPaddingLayer',
    ):
        # check layer name (fixed)
        Layer.__init__(self, prev_layer=prev_layer, name=name)

        # the input of this layer is the output of previous layer (fixed)
        self.inputs = prev_layer.outputs

        # operation (customized)
        self.outputs = tf.pad(self.inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

class UpsampleLayer(Layer):
    def __init__(
        self,
        prev_layer = None,
        scale=2,
        name ='UpsampleLayer',
    ):
        # check layer name (fixed)
        Layer.__init__(self, prev_layer=prev_layer, name=name)

        # the input of this layer is the output of previous layer (fixed)
        self.inputs = prev_layer.outputs

        height = tf.shape(self.inputs)[1] * scale
        width = tf.shape(self.inputs)[2] * scale

        # operation (customized)
        self.outputs = tf.image.resize_images(self.inputs, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

