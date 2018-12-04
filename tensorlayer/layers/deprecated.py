#! /usr/bin/python
# -*- coding: utf-8 -*-

__all__ = []


class NonExistingLayerError(Exception):
    pass


# activation.py
__all__ += [
    'PReluLayer',
    'PRelu6Layer',
    'PTRelu6Layer',
]


def PReluLayer(*args, **kwargs):
    raise NonExistingLayerError("PReluLayer(net, name='a') --> PRelu(name='a')(net))")


def PRelu6Layer(*args, **kwargs):
    raise NonExistingLayerError("PRelu6Layer(net, name='a') --> PRelu6(name='a')(net))")


def PTRelu6Layer(*args, **kwargs):
    raise NonExistingLayerError("PTRelu6Layer(net, name='a') --> PTRelu(name='a')(net))")


# convolution/atrous_conv.py
__all__ += [
    'AtrousConv1dLayer',
    'AtrousConv2dLayer',
    'AtrousDeConv2dLayer',
]


def AtrousConv1dLayer(*args, **kwargs):
    raise NonExistingLayerError("use `tl.layers.Conv1d` with dilation instead")


def AtrousConv2dLayer(*args, **kwargs):
    raise NonExistingLayerError("use `tl.layers.Conv2d` with dilation instead")


def AtrousDeConv2dLayer(*args, **kwargs):
    raise NonExistingLayerError("AtrousDeConv2dLayer(net, name='a') --> AtrousDeConv2d(name='a')(net)")


# dense/base_dense.py
__all__ += [
    'DenseLayer',
]


def DenseLayer(*args, **kwargs):
    raise NonExistingLayerError("DenseLayer(net, name='a') --> Dense(name='a')(net)")


# dense/binary_dense.py
__all__ += [
    'BinaryDenseLayer',
]


def BinaryDenseLayer(*args, **kwargs):
    raise NonExistingLayerError("BinaryDenseLayer(net, name='a') --> BinaryDense(name='a')(net)")


# dense/dorefa_dense.py
__all__ += [
    'DorefaDenseLayer',
]


def DorefaDenseLayer(*args, **kwargs):
    raise NonExistingLayerError("DorefaDenseLayer(net, name='a') --> DorefaDense(name='a')(net)")


# dense/dropconnect.py
__all__ += [
    'DropconnectDenseLayer',
]


def DropconnectDenseLayer(*args, **kwargs):
    raise NonExistingLayerError("DropconnectDenseLayer(net, name='a') --> DropconnectDense(name='a')(net)")


# dense/ternary_dense.py
__all__ += [
    'TernaryDenseLayer',
]


def TernaryDenseLayer(*args, **kwargs):
    raise NonExistingLayerError("TernaryDenseLayer(net, name='a') --> TernaryDense(name='a')(net)")


# dropout.py
__all__ += [
    'DropoutLayer',
]


def DropoutLayer(*args, **kwargs):
    raise NonExistingLayerError("DropoutLayer(net, is_train=True, name='a') --> Dropout(name='a')(net, is_train=True)")


# extend.py
__all__ += [
    'ExpandDimsLayer',
    'TileLayer',
]


def ExpandDimsLayer(*args, **kwargs):
    raise NonExistingLayerError("ExpandDimsLayer(net, name='a') --> ExpandDims(name='a')(net)")


def TileLayer(*args, **kwargs):
    raise NonExistingLayerError("TileLayer(net, name='a') --> Tile(name='a')(net)")


# image_resampling.py
__all__ += [
    'UpSampling2dLayer',
    'DownSampling2dLayer',
]


def UpSampling2dLayer(*args, **kwargs):
    raise NonExistingLayerError("UpSampling2dLayer(net, name='a') --> UpSampling2d(name='a')(net)")


def DownSampling2dLayer(*args, **kwargs):
    raise NonExistingLayerError("DownSampling2dLayer(net, name='a') --> DownSampling2d(name='a')(net)")


# importer.py
__all__ += [
    'SlimNetsLayer',
    'KerasLayer',
]


def SlimNetsLayer(*args, **kwargs):
    raise NonExistingLayerError("SlimNetsLayer(net, name='a') --> SlimNets(name='a')(net)")


def KerasLayer(*args, **kwargs):
    raise NonExistingLayerError("KerasLayer(net, name='a') --> Keras(name='a')(net)")


# inputs.py
__all__ += [
    'InputLayer',
    'OneHotInputLayer',
    'Word2vecEmbeddingInputlayer',
    'EmbeddingInputlayer',
    'AverageEmbeddingInputlayer',
]


def InputLayer(*args, **kwargs):
    raise NonExistingLayerError("InputLayer(x, name='a') --> Input(name='a')(x)")


def OneHotInputLayer(*args, **kwargs):
    raise NonExistingLayerError("OneHotInputLayer(x, name='a') --> OneHotInput(name='a')(x)")


def Word2vecEmbeddingInputlayer(*args, **kwargs):
    raise NonExistingLayerError("Word2vecEmbeddingInputlayer(x, name='a') --> Word2vecEmbeddingInput(name='a')(x)")


def EmbeddingInputlayer(*args, **kwargs):
    raise NonExistingLayerError("EmbeddingInputlayer(x, name='a') --> EmbeddingInput(name='a')(x)")


def AverageEmbeddingInputlayer(*args, **kwargs):
    raise NonExistingLayerError("AverageEmbeddingInputlayer(x, name='a') --> AverageEmbeddingInput(name='a')(x)")


# lambda.py
__all__ += [
    'LambdaLayer',
    'ElementwiseLambdaLayer',
]


def LambdaLayer(*args, **kwargs):
    raise NonExistingLayerError("LambdaLayer(x, lambda x: 2*x, name='a') --> Lambda(lambda x: 2*x, name='a')(x)")


def ElementwiseLambdaLayer(*args, **kwargs):
    raise NonExistingLayerError("ElementwiseLambdaLayer(x, ..., name='a') --> ElementwiseLambda(..., name='a')(x)")


# merge.py
__all__ += [
    'ConcatLayer',
    'ElementwiseLayer',
]


def ConcatLayer(*args, **kwargs):
    raise NonExistingLayerError("ConcatLayer(x, ..., name='a') --> Concat(..., name='a')(x)")


def ElementwiseLayer(*args, **kwargs):
    raise NonExistingLayerError("ElementwiseLayer(x, ..., name='a') --> Elementwise(..., name='a')(x)")


# noise.py
__all__ += [
    'GaussianNoiseLayer',
]


def GaussianNoiseLayer(*args, **kwargs):
    raise NonExistingLayerError("GaussianNoiseLayer(x, ..., name='a') --> GaussianNoise(..., name='a')(x)")


# normalization.py
__all__ += [
    'BatchNormLayer',
    'InstanceNormLayer',
    'LayerNormLayer',
    'LocalResponseNormLayer',
    'GroupNormLayer',
    'SwitchNormLayer',
]


def BatchNormLayer(*args, **kwargs):
    raise NonExistingLayerError("BatchNormLayer(x, is_train=True, name='a') --> BatchNorm(name='a')(x, is_train=True)")


def InstanceNormLayer(*args, **kwargs):
    raise NonExistingLayerError("InstanceNormLayer(x, name='a') --> InstanceNorm(name='a')(x)")


def LayerNormLayer(*args, **kwargs):
    raise NonExistingLayerError("LayerNormLayer(x, name='a') --> LayerNorm(name='a')(x)")


def LocalResponseNormLayer(*args, **kwargs):
    raise NonExistingLayerError("LocalResponseNormLayer(x, name='a') --> LocalResponseNorm(name='a')(x)")


def GroupNormLayer(*args, **kwargs):
    raise NonExistingLayerError("GroupNormLayer(x, name='a') --> GroupNorm(name='a')(x)")


def SwitchNormLayer(*args, **kwargs):
    raise NonExistingLayerError("SwitchNormLayer(x, name='a') --> SwitchNorm(name='a')(x)")


# quantize_layer.py
__all__ += [
    'SignLayer',
]


def SignLayer(*args, **kwargs):
    raise NonExistingLayerError("SignLayer(x, name='a') --> Sign(name='a')(x)")


# recurrent/lstm_layers.py
__all__ += [
    'ConvLSTMLayer',
]


def ConvLSTMLayer(*args, **kwargs):
    raise NonExistingLayerError("ConvLSTMLayer(x, name='a') --> ConvLSTM(name='a')(x)")


# recurrent/rnn_dynamic_layers.py
__all__ += [
    'DynamicRNNLayer',
    'BiDynamicRNNLayer',
]


def DynamicRNNLayer(*args, **kwargs):
    raise NonExistingLayerError(
        "DynamicRNNLayer(x, is_train=True, name='a') --> DynamicRNN(name='a')(x, is_train=True)"
    )


def BiDynamicRNNLayer(*args, **kwargs):
    raise NonExistingLayerError(
        "BiDynamicRNNLayer(x, is_train=True, name='a') --> BiDynamicRNN(name='a')(x, is_train=True)"
    )


# recurrent/rnn_layers.py
__all__ += [
    'RNNLayer',
    'BiRNNLayer',
]


def RNNLayer(*args, **kwargs):
    raise NonExistingLayerError("RNNLayer(x, name='a') --> RNN(name='a')(x)")


def BiRNNLayer(*args, **kwargs):
    raise NonExistingLayerError("BiRNNLayer(x, is_train=True, name='a') --> BiRNN(name='a')(x, is_train=True)")


# reshape.py
__all__ += [
    'FlattenLayer',
    'ReshapeLayer',
    'TransposeLayer',
]


def FlattenLayer(*args, **kwargs):
    raise NonExistingLayerError("FlattenLayer(x, name='a') --> Flatten(name='a')(x)")


def ReshapeLayer(*args, **kwargs):
    raise NonExistingLayerError("ReshapeLayer(x, name='a') --> Reshape(name='a')(x)")


def TransposeLayer(*args, **kwargs):
    raise NonExistingLayerError("TransposeLayer(x, name='a') --> Transpose(name='a')(x)")


# scale.py
__all__ += [
    'ScaleLayer',
]


def ScaleLayer(*args, **kwargs):
    raise NonExistingLayerError("ScaleLayer(x, name='a') --> Scale(name='a')(x)")


# spatial_transformer.py
__all__ += ['SpatialTransformer2dAffineLayer']


def SpatialTransformer2dAffineLayer(*args, **kwargs):
    raise NonExistingLayerError(
        "SpatialTransformer2dAffineLayer(x1, x2, name='a') --> SpatialTransformer2dAffine(name='a')(x1, x2)"
    )


# stack.py
__all__ += [
    'StackLayer',
    'UnStackLayer',
]


def StackLayer(*args, **kwargs):
    raise NonExistingLayerError("StackLayer(x1, x2, name='a') --> Stack(name='a')(x1, x2)")


def UnStackLayer(*args, **kwargs):
    raise NonExistingLayerError("UnStackLayer(x1, x2, name='a') --> UnStack(name='a')(x1, x2)")


# time_distributed.py
__all__ += [
    'TimeDistributedLayer',
]


def TimeDistributedLayer(*args, **kwargs):
    raise NonExistingLayerError("TimeDistributedLayer(x1, x2, name='a') --> TimeDistributed(name='a')(x1, x2)")
