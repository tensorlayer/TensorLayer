#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers.core import Module

__all__ = [
    'BatchNorm',
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
]
# TODO Layers that needs to be updated
# ['InstanceNorm',
#     'InstanceNorm1d',
#     'InstanceNorm2d',
#     'InstanceNorm3d',
#     'LayerNorm',
#     'GroupNorm',
#     'SwitchNorm',
# ]


class BatchNorm(Module):
    """
    The :class:`BatchNorm` is a batch normalization layer for both fully-connected and convolution outputs.
    See ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Parameters
    ----------
    decay : float
        A decay factor for `ExponentialMovingAverage`.
        Suggest to use a large value for large dataset.
    epsilon : float
        Eplison.
    act : activation function
        The activation function of this layer.
    is_train : boolean
        Is being used for training or inference.
    beta_init : initializer or None
        The initializer for initializing beta, if None, skip beta.
        Usually you should not skip beta unless you know what happened.
    gamma_init : initializer or None
        The initializer for initializing gamma, if None, skip gamma.
        When the batch normalization layer is use instead of 'biases', or the next layer is linear, this can be
        disabled since the scaling can be done by the next layer. see `Inception-ResNet-v2 <https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py>`__
    moving_mean_init : initializer or None
        The initializer for initializing moving mean, if None, skip moving mean.
    moving_var_init : initializer or None
        The initializer for initializing moving var, if None, skip moving var.
    num_features: int
        Number of features for input tensor. Useful to build layer if using BatchNorm1d, BatchNorm2d or BatchNorm3d,
        but should be left as None if using BatchNorm. Default None.
    data_format : str
        channels_last 'channel_last' (default) or channels_first.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([10, 50, 50, 32], name='input')
    >>> net = tl.layers.BatchNorm()(net)

    Notes
    -----
    The :class:`BatchNorm` is universally suitable for 3D/4D/5D input in static model, but should not be used
    in dynamic model where layer is built upon class initialization. So the argument 'num_features' should only be used
    for subclasses :class:`BatchNorm1d`, :class:`BatchNorm2d` and :class:`BatchNorm3d`. All the three subclasses are
    suitable under all kinds of conditions.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`__
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`__

    """

    def __init__(
        self,
        decay=0.9,
        epsilon=0.00001,
        act=None,
        is_train=True,
        beta_init=tl.initializers.zeros(),
        gamma_init=tl.initializers.random_normal(mean=1.0, stddev=0.002),
        moving_mean_init=tl.initializers.zeros(),
        moving_var_init=tl.initializers.zeros(),
        num_features=None,
        data_format='channels_last',
        name=None,
    ):
        super(BatchNorm, self).__init__(name=name, act=act)
        self.decay = decay
        self.epsilon = epsilon
        self.data_format = data_format
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.moving_mean_init = moving_mean_init
        self.moving_var_init = moving_var_init
        self.num_features = num_features
        self.is_train = is_train

        self.axes = None

        # if self.num_features is None:
        #     raise AttributeError(
        #         "The registered layer `{}` should be built in advance. "
        #         "Do you forget to pass the keyword argument 'num_feature'? "
        #     )

        if self.num_features:
            self.build(None)
            self._built = True

        if self.decay < 0.0 or 1.0 < self.decay:
            raise ValueError("decay should be between 0 to 1")

        logging.info(
            "BatchNorm %s: decay: %f epsilon: %f act: %s is_train: %s" % (
                self.name, decay, epsilon, self.act.__class__.__name__ if self.act is not None else 'No Activation',
                is_train
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(num_features={num_features}, decay={decay}' ', epsilon={epsilon}')
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name="{name}"'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def _get_param_shape(self, inputs_shape):
        if self.data_format == 'channels_last':
            axis = -1
        elif self.data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('data_format should be either %s or %s' % ('channels_last', 'channels_first'))

        channels = inputs_shape[axis]
        params_shape = [channels]

        return params_shape

    def _check_input_shape(self, inputs):
        if inputs.ndim <= 1:
            raise ValueError('expected input at least 2D, but got {}D input'.format(inputs.ndim))

    def build(self, inputs_shape):
        params_shape = [self.num_features] if self.num_features is not None else self._get_param_shape(inputs_shape)

        self.beta, self.gamma = None, None
        if self.beta_init:
            self.beta = self._get_weights(var_name="beta", shape=params_shape, init=self.beta_init)

        if self.gamma_init:
            self.gamma = self._get_weights(var_name="gamma", shape=params_shape, init=self.gamma_init)

        self.moving_mean = self._get_weights(
            var_name="moving_mean", shape=params_shape, init=self.moving_mean_init, trainable=False
        )
        self.moving_var = self._get_weights(
            var_name="moving_var", shape=params_shape, init=self.moving_var_init, trainable=False
        )

        self.batchnorm = tl.ops.BatchNorm(
            decay=self.decay, epsilon=self.epsilon, beta=self.beta, gamma=self.gamma, moving_mean=self.moving_mean,
            moving_var=self.moving_var, num_features=self.num_features, data_format=self.data_format,
            is_train=self.is_train
        )

        self.act_init_flag = False
        if self.act:
            self.act_init_flag = True

    def forward(self, inputs):
        self._check_input_shape(inputs)
        if self._forward_state == False:
            if self._built == False:
                self.build(tl.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        if not self.is_train:
            self.batchnorm = tl.ops.BatchNorm(
                decay=self.decay, epsilon=self.epsilon, beta=self.beta, gamma=self.gamma, moving_mean=self.moving_mean,
                moving_var=self.moving_var, num_features=self.num_features, data_format=self.data_format, is_train=False
            )
        outputs = self.batchnorm(inputs=inputs)
        if self.act_init_flag:
            outputs = self.act(outputs)
        return outputs


class BatchNorm1d(BatchNorm):
    """The :class:`BatchNorm1d` applies Batch Normalization over 2D/3D input (a mini-batch of 1D
    inputs (optional) with additional channel dimension), of shape (N, C) or (N, L, C) or (N, C, L).
    See more details in :class:`BatchNorm`.

    Examples
    ---------
    With TensorLayer

    >>> # in static model, no need to specify num_features
    >>> net = tl.layers.Input([10, 50, 32], name='input')
    >>> net = tl.layers.BatchNorm1d()(net)
    >>> # in dynamic model, build by specifying num_features
    >>> conv = tl.layers.Conv1d(32, 5, 1, in_channels=3)
    >>> bn = tl.layers.BatchNorm1d(num_features=32)

    """

    def _check_input_shape(self, inputs):
        if inputs.ndim != 2 and inputs.ndim != 3:
            raise ValueError('expected input to be 2D or 3D, but got {}D input'.format(inputs.ndim))


class BatchNorm2d(BatchNorm):
    """The :class:`BatchNorm2d` applies Batch Normalization over 4D input (a mini-batch of 2D
    inputs with additional channel dimension) of shape (N, H, W, C) or (N, C, H, W).
    See more details in :class:`BatchNorm`.

    Examples
    ---------
    With TensorLayer

    >>> # in static model, no need to specify num_features
    >>> net = tl.layers.Input([10, 50, 50, 32], name='input')
    >>> net = tl.layers.BatchNorm2d()(net)
    >>> # in dynamic model, build by specifying num_features
    >>> conv = tl.layers.Conv2d(32, (5, 5), (1, 1), in_channels=3)
    >>> bn = tl.layers.BatchNorm2d(num_features=32)

    """

    def _check_input_shape(self, inputs):
        if inputs.ndim != 4:
            raise ValueError('expected input to be 4D, but got {}D input'.format(inputs.ndim))


class BatchNorm3d(BatchNorm):
    """The :class:`BatchNorm3d` applies Batch Normalization over 5D input (a mini-batch of 3D
    inputs with additional channel dimension) with shape (N, D, H, W, C) or (N, C, D, H, W).
    See more details in :class:`BatchNorm`.

    Examples
    ---------
    With TensorLayer

    >>> # in static model, no need to specify num_features
    >>> net = tl.layers.Input([10, 50, 50, 50, 32], name='input')
    >>> net = tl.layers.BatchNorm3d()(net)
    >>> # in dynamic model, build by specifying num_features
    >>> conv = tl.layers.Conv3d(32, (5, 5, 5), (1, 1), in_channels=3)
    >>> bn = tl.layers.BatchNorm3d(num_features=32)

    """

    def _check_input_shape(self, inputs):
        if inputs.ndim != 5:
            raise ValueError('expected input to be 5D, but got {}D input'.format(inputs.ndim))
