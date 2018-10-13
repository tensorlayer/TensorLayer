#! /usr/bin/python
# -*- coding: utf-8 -*-

## activation.py
__all__ = [
    'PReluLayer',
    'PRelu6Layer',
    'PTRelu6Layer',
]

def PReluLayer(*args, **kwargs):
    raise Exception("PReluLayer(net, name='a') --> PRelu(name='a')(net))")

def PRelu6Layer(*args, **kwargs):
    raise Exception("PRelu6Layer(net, name='a') --> PRelu6(name='a')(net))")

def PTRelu6Layer(*args, **kwargs):
    raise Exception("PTRelu6Layer(net, name='a') --> PTRelu(name='a')(net))")

## convolution/atrous_conv.py
__all__ += [
    'AtrousConv1dLayer',
    'AtrousConv2dLayer',
    'AtrousDeConv2dLayer',
]
def AtrousConv1dLayer(*args, **kwargs):
    raise Exception("use Conv1d with dilation instead")
def AtrousConv2dLayer(*args, **kwargs):
    raise Exception("use Conv2d with dilation instead")
def AtrousDeConv2dLayer(*args, **kwargs):
    raise Exception("AtrousDeConv2dLayer(net, name='a') --> AtrousDeConv2d(name='a')(net)")

## dense/base_dense.py
__all__ += [
    'DenseLayer',
]
def DenseLayer(*args, **kwargs):
    raise Exception("DenseLayer(net, name='a') --> Dense(name='a')(net)")

## dense/binary_dense.py
__all__ += [
    'BinaryDenseLayer',
]
def BinaryDenseLayer(*args, **kwargs):
    raise Exception("BinaryDenseLayer(net, name='a') --> BinaryDense(name='a')(net)")

## dense/dorefa_dense.py
__all__ += [
    'DorefaDenseLayer',
]
def DorefaDenseLayer(*args, **kwargs):
    raise Exception("DorefaDenseLayer(net, name='a') --> DorefaDense(name='a')(net)")

## dense/dropconnect.py
__all__ += [
    'DropconnectDenseLayer',
]
def DropconnectDenseLayer(*args, **kwargs):
    raise Exception("DropconnectDenseLayer(net, name='a') --> DropconnectDense(name='a')(net)")

## dense/ternary_dense.py
__all__ += [
    'TernaryDenseLayer',
]
def TernaryDenseLayer(*args, **kwargs):
    raise Exception("TernaryDenseLayer(net, name='a') --> TernaryDense(name='a')(net)")

## dropout.py
__all__ += [
    'DropoutLayer',
]
def DropoutLayer(*args, **kwargs):
    raise Exception("DropoutLayer(net, is_train=True, name='a') --> Dropout(name='a')(net, is_train=True)")

## extend.py
__all__ += [
    'ExpandDimsLayer',
    'TileLayer',
]
def ExpandDimsLayer(*args, **kwargs):
    raise Exception("ExpandDimsLayer(net, name='a') --> ExpandDims(name='a')(net)")
def TileLayer(*args, **kwargs):
    raise Exception("TileLayer(net, name='a') --> Tile(name='a')(net)")

## image_resampling.py
__all__ += [
    'UpSampling2dLayer',
    'DownSampling2dLayer',
]
def UpSampling2dLayer(*args, **kwargs):
    raise Exception("UpSampling2dLayer(net, name='a') --> UpSampling2d(name='a')(net)")
def DownSampling2dLayer(*args, **kwargs):
    raise Exception("DownSampling2dLayer(net, name='a') --> DownSampling2d(name='a')(net)")

## importer.py
__all__ += [
    'SlimNetsLayer',
    'KerasLayer',
]
def SlimNetsLayer(*args, **kwargs):
    raise Exception("SlimNetsLayer(net, name='a') --> SlimNets(name='a')(net)")
def KerasLayer(*args, **kwargs):
    raise Exception("KerasLayer(net, name='a') --> Keras(name='a')(net)")

## inputs.py
__all__ += [
    'InputLayer',
    'OneHotInputLayer',
    'Word2vecEmbeddingInputlayer',
    'EmbeddingInputlayer',
    'AverageEmbeddingInputlayer',
]
def InputLayer(*args, **kwargs):
    raise Exception("InputLayer(x, name='a') --> Input(name='a')(x)")
def OneHotInputLayer(*args, **kwargs):
    raise Exception("OneHotInputLayer(x, name='a') --> OneHotInput(name='a')(x)")
def Word2vecEmbeddingInputlayer(*args, **kwargs):
    raise Exception("Word2vecEmbeddingInputlayer(x, name='a') --> Word2vecEmbeddingInput(name='a')(x)")
def AverageEmbeddingInputlayer(*args, **kwargs):
    raise Exception("AverageEmbeddingInputlayer(x, name='a') --> AverageEmbeddingInput(name='a')(x)")

## lambda.py
__all__+= [
    'LambdaLayer',
    'ElementwiseLambdaLayer',
]
def LambdaLayer(*args, **kwargs):
    raise Exception("LambdaLayer(x, lambda x: 2*x, name='a') --> Lambda(lambda x: 2*x, name='a')(x)")
def ElementwiseLambdaLayer(*args, **kwargs):
    raise Exception("ElementwiseLambdaLayer(x, ..., name='a') --> ElementwiseLambda(..., name='a')(x)")

## merge.py
__all__ += [
    'ConcatLayer',
    'ElementwiseLayer',
]
def ConcatLayer(*args, **kwargs):
    raise Exception("ConcatLayer(x, ..., name='a') --> Concat(..., name='a')(x)")
def ElementwiseLayer(*args, **kwargs):
    raise Exception("ElementwiseLayer(x, ..., name='a') --> Elementwise(..., name='a')(x)")

## noise.py
__all__ += [
    'GaussianNoiseLayer',
]
def GaussianNoiseLayer(*args, **kwargs):
    raise Exception("GaussianNoiseLayer(x, ..., name='a') --> GaussianNoise(..., name='a')(x)")

## normalization.py
__all__ += [
    'BatchNormLayer',
    'InstanceNormLayer',
    'LayerNormLayer',
    'LocalResponseNormLayer',
    'GroupNormLayer',
    'SwitchNormLayer',
]
def BatchNormLayer(*args, **kwargs):
    raise Exception("BatchNormLayer(x, is_train=True, name='a') --> BatchNorm(name='a')(x, is_train=True)")
def InstanceNormLayer(*args, **kwargs):
    raise Exception("InstanceNormLayer(x, name='a') --> InstanceNorm(name='a')(x)")
def LayerNormLayer(*args, **kwargs):
    raise Exception("LayerNormLayer(x, name='a') --> LayerNorm(name='a')(x)")
def LocalResponseNormLayer(*args, **kwargs):
    raise Exception("LocalResponseNormLayer(x, name='a') --> LocalResponseNorm(name='a')(x)")
def GroupNormLayer(*args, **kwargs):
    raise Exception("GroupNormLayer(x, name='a') --> GroupNorm(name='a')(x)")
def SwitchNormLayer(*args, **kwargs):
    raise Exception("SwitchNormLayer(x, name='a') --> SwitchNorm(name='a')(x)")

## quantize_layer.py
__all__+= [
    'SignLayer',
]
def SignLayer(*args, **kwargs):
    raise Exception("SignLayer(x, name='a') --> Sign(name='a')(x)")

## recurrent/lstm_layers.py
__all__ += [
    'ConvLSTMLayer',
]
def ConvLSTMLayer(*args, **kwargs):
    raise Exception("ConvLSTMLayer(x, name='a') --> ConvLSTM(name='a')(x)")

## recurrent/rnn_dynamic_layers.py
__all__ += [
    'DynamicRNNLayer',
    'BiDynamicRNNLayer',
]
def DynamicRNNLayer(*args, **kwargs):
    raise Exception("DynamicRNNLayer(x, is_train=True, name='a') --> DynamicRNN(name='a')(x, is_train=True)")
def BiDynamicRNNLayer(*args, **kwargs):
    raise Exception("BiDynamicRNNLayer(x, is_train=True, name='a') --> BiDynamicRNN(name='a')(x, is_train=True)")

## recurrent/rnn_layers.py
__all__ += [
    'RNNLayer',
    'BiRNNLayer',
]
def RNNLayer(*args, **kwargs):
    raise Exception("RNNLayer(x, name='a') --> RNN(name='a')(x)")
def BiRNNLayer(*args, **kwargs):
    raise Exception("BiRNNLayer(x, is_train=True, name='a') --> BiRNNLayer(name='a')(x, is_train=True)")

## reshape.py
__all__ += [
    'FlattenLayer',
    'ReshapeLayer',
    'TransposeLayer',
]
def FlattenLayer(*args, **kwargs):
    raise Exception("FlattenLayer(x, name='a') --> Flatten(name='a')(x)")
def ReshapeLayer(*args, **kwargs):
    raise Exception("ReshapeLayer(x, name='a') --> Reshape(name='a')(x)")
def TransposeLayer(*args, **kwargs):
    raise Exception("TransposeLayer(x, name='a') --> Transpose(name='a')(x)")

## scale.py
__all__ += [
    'ScaleLayer',
]
def ScaleLayer(*args, **kwargs):
    raise Exception("ScaleLayer(x, name='a') --> Scale(name='a')(x)")

## spatial_transformer.py
__all__ += ['SpatialTransformer2dAffineLayer']
def SpatialTransformer2dAffineLayer(*args, **kwargs):
    raise Exception("SpatialTransformer2dAffineLayer(x1, x2, name='a') --> SpatialTransformer2dAffine(name='a')(x1, x2)")

## stack.py
__all__ += [
    'StackLayer',
    'UnStackLayer',
]
def StackLayer(*args, **kwargs):
    raise Exception("StackLayer(x1, x2, name='a') --> Stack(name='a')(x1, x2)")
def UnStackLayer(*args, **kwargs):
    raise Exception("UnStackLayer(x1, x2, name='a') --> UnStack(name='a')(x1, x2)")

## time_distributed.py
__all__ += [
    'TimeDistributedLayer',
]
def TimeDistributedLayer(*args, **kwargs):
    raise Exception("TimeDistributedLayer(x1, x2, name='a') --> TimeDistributed(name='a')(x1, x2)")
