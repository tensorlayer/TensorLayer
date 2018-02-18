# -*- coding: utf-8 -*-

from .core import *


class ROIPoolingLayer(Layer):
    """
    The region of interest pooling layer.

    Parameters
    -----------
    layer : :class:`Layer`
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
    - Please install it by the instruction `HERE <https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/third_party/roi_pooling/README.md>`__.

    """

    def __init__(
            self,
            layer,
            rois,
            pool_height=2,
            pool_width=2,
            name='roipooling_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        logging.info("ROIPoolingLayer %s: (%d, %d)" % (self.name, pool_height, pool_width))
        try:
            from tensorlayer.third_party.roi_pooling.roi_pooling.roi_pooling_ops import roi_pooling
        except Exception as e:
            logging.info(e)
            logging.info("HINT: 1. https://github.com/deepsense-ai/roi-pooling  2. tensorlayer/third_party/roi_pooling")
        self.outputs = roi_pooling(self.inputs, rois, pool_height, pool_width)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
