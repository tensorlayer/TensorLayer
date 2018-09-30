#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

from tensorlayer.lazy_imports import LazyImport

try:
    roi_pooling = LazyImport("tensorlayer.layers.contrib.roi_pooling.sources.roi_pooling.roi_pooling_ops")
except Exception as e:
    tl.logging.error(e)
    tl.logging.error("HINT: 1. https://github.com/deepsense-ai/roi-pooling  2. tensorlayer/third_party/roi_pooling")

__all__ = [
    'ROIPoolingLayer',
]


class ROIPoolingLayer(Layer):
    """
    The region of interest pooling layer.

    Parameters
    -----------
    rois : tuple of int
        Regions of interest in the format of (feature map index, upper left, bottom right).
    pool_width : int
        The size of the pooling sections.
    pool_width : int
        The size of the pooling sections.
    name : str
        A unique layer name.

    Notes
    -----------
    - This implementation is imported from `Deepsense-AI <https://github.com/deepsense-ai/roi-pooling>`__ .
    - Please install it by the instruction `HERE <https://github.com/tensorlayer/tensorlayer/blob/master/tensorlayer/third_party/roi_pooling/README.md>`__.

    """

    def __init__(
        self,
        rois,
        pool_height=2,
        pool_width=2,
        name='roipooling_layer',
    ):

        if not isinstance(rois, tuple):
            raise ValueError('`rois` should be of type `tuple`')

        self.rois = rois
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.name = name

        super(ROIPoolingLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("pool_shape: (%d, %d)" % (self.pool_height, self.pool_width))
        except AttributeError:
            pass

        return self._str(additional_str)

    def compile(self):

        with tf.variable_scope(self.name):
            self._temp_data['outputs'] = roi_pooling(
                self._temp_data['inputs'], self.rois, self.pool_height, self.pool_width
            )
