#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.layers.core import Layer

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

from tensorlayer.lazy_imports import LazyImport

try:
    roi_pooling = LazyImport("tensorlayer.third_party.roi_pooling.roi_pooling.roi_pooling_ops")
except Exception as e:
    logging.error(e)
    logging.error("HINT: 1. https://github.com/deepsense-ai/roi-pooling  2. tensorlayer/third_party/roi_pooling")

__all__ = [
    'ROIPoolingLayer',
]


class ROIPoolingLayer(Layer):
    """
    The region of interest pooling layer.

    Parameters
    -----------
    prev_layer : :class:`Layer`
        The previous layer.
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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            rois,
            pool_height=2,
            pool_width=2,
            name='roipooling_layer',
    ):
        super(ROIPoolingLayer, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("ROIPoolingLayer %s: (%d, %d)" % (self.name, pool_height, pool_width))

        self.outputs = roi_pooling(self.inputs, rois, pool_height, pool_width)

        self._add_layers(self.outputs)
