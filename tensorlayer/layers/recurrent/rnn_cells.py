#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import LayersConfig

__all__ = ['ConvRNNCell']

class ConvRNNCell(object):
    """Abstract object representing an Convolutional RNN Cell."""

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state."""
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell."""
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype=LayersConfig.tf_dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros

        """
        shape = self.shape
        num_features = self.num_features
        # TODO : TypeError: 'NoneType' object is not subscriptable
        zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * 2], dtype=dtype)
        return zeros
