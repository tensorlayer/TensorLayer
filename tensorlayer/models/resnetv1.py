#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
ResNet for ImageNet.
Introduction
----------------
The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.

Download Pre-trained Model
----------------------------
- Model weights in this example - resnet_v1_XX_2016_08_28.tar.gz : http://download.tensorflow.org/models/resnet_v1_XX_2016_08_28.tar.gz

Note
------
- For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.
- When feeding other images to the model be sure to properly resize or crop them
beforehand. Distorted images might end up being misclassified. One way of safely
feeding images of multiple sizes is by doing center cropping.
"""

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from tensorlayer import tl_logging as logging

from tensorflow.contrib.slim.nets import resnet_utils
from tensorlayer.layers import Layer

from tensorlayer.files import maybe_download_and_extract, assign_params, load_ckpt

resnet_arg_scope = resnet_utils.resnet_arg_scope

__all__ = [
    'ResNetV1',
]

class ResNetV1_50(Layer):
    """Pre-trained ResNetV1_50 model.

       Parameters
       ------------
       x : placeholder
           shape [None, 224, 224, 3], value range [0, 1].
       end_with : str
           The end point of the model [conv, depth1, depth2 ... depth13, globalmeanpool, out]. Default ``out`` i.e. the whole model.
       is_train : boolean
           Whether the model is used for training i.e. enable dropout.
       reuse : boolean
           Whether to reuse the model.

       Examples
       ---------
       Classify ImageNet classes, see `tutorial_models_resnetv1.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_resnetv1.py>`__

       >>> x = tf.placeholder(tf.float32, [None, 224, 224, 3])
       >>> # get the whole model
       >>> net = tl.models.ResNetV1(x)
       >>> # restore pre-trained parameters
       >>> sess = tf.InteractiveSession()
       >>> net.restore_params(sess)
       >>> # use for inferencing
       >>> probs = tf.nn.softmax(net.outputs)

       Extract features and Train a classifier with 100 classes

       >>> x = tf.placeholder(tf.float32, [None, 224, 224, 3])
       >>> # get model without the last layer
       >>> cnn = tl.models.ResNetV1(x, end_with='reshape')
       >>> # add one more layer
       >>> net = Conv2d(cnn, 100, (1, 1), (1, 1), name='out')
       >>> net = FlattenLayer(net, name='flatten')
       >>> # initialize all parameters
       >>> sess = tf.InteractiveSession()
       >>> tl.layers.initialize_global_variables(sess)
       >>> # restore pre-trained parameters
       >>> cnn.restore_params(sess)
       >>> # train your own classifier (only update the last layer)
       >>> train_params = tl.layers.get_variables_with_name('out')

       Reuse model

       >>> x1 = tf.placeholder(tf.float32, [None, 224, 224, 3])
       >>> x2 = tf.placeholder(tf.float32, [None, 224, 224, 3])
       >>> # get model without the last layer
       >>> net1 = tl.models.ResNetV1(x1, end_with='reshape')
       >>> # reuse the parameters with different input
       >>> net2 = tl.models.ResNetV1(x2, end_with='reshape', reuse=True)
       >>> # restore pre-trained parameters (as they share parameters, we don’t need to restore net2)
       >>> sess = tf.InteractiveSession()
       >>> net1.restore_params(sess)

       """

    def __init__(self, prev_layer, end_with='conv1', num_classes=None, is_train=False, global_pool=True, output_stride=None, reuse=None):

        # super(ResNetV1_50, self
        #      ).__init__(prev_layer=prev_layer, name=name)

        self.net, self.end_points = self._resnet_v1_50(x,
                 num_classes,
                 is_train,
                 global_pool,
                 output_stride,
                 reuse,
                 scope='resnet_v1_50')

        self.outputs = self.net  # self.end_points[end_with]

        self._add_layers(self.outputs)  # TODO: all tensors of all layers

        all_params = # TODO

        self._add_params(all_params) #

        # self.all_params = list(self.net.all_params)
        # self.all_layers = list(self.net.all_layers)
        # self.all_drop = dict(self.net.all_drop)
        #
        # self.print_layers = self.net.print_layers
        # self.print_params = self.net.print_params

    # @classmethod
    # inputs has shape [batch, 224, 224, 3]
    def _resnet_v1_50(self,
                     inputs,
                     num_classes=None,
                     is_training=True,
                     global_pool=True,
                     output_stride=None,
                     reuse=None,
                     scope='resnet_v1_50'):

        # ref: https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py

        """Generator for v1 ResNet models.
        This function generates a family of ResNet v1 models. See the resnet_v1_*()
        methods for specific model instantiations, obtained by selecting different
        block instantiations that produce ResNets of various depths.
        Training for image classification on Imagenet is usually done with [224, 224]
        inputs, resulting in [7, 7] feature maps at the output of the last ResNet
        block for the ResNets defined in [1] that have nominal stride equal to 32.
        However, for dense prediction tasks we advise that one uses inputs with
        spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
        this case the feature maps at the ResNet output will have spatial shape
        [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
        and corners exactly aligned with the input image corners, which greatly
        facilitates alignment of the features to the image. Using as input [225, 225]
        images results in [8, 8] feature maps at the output of the last ResNet block.
        For dense prediction tasks, the ResNet needs to run in fully-convolutional
        (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
        have nominal stride equal to 32 and a good choice in FCN mode is to use
        output_stride=16 in order to increase the density of the computed features at
        small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.
        Args:
          inputs: A tensor of size [batch, height_in, width_in, channels].
          blocks: A list of length equal to the number of ResNet blocks. Each element
            is a resnet_utils.Block object describing the units in the block.
          num_classes: Number of predicted classes for classification tasks.
            If 0 or None, we return the features before the logit layer.
          is_training: whether batch_norm layers are in training mode. If this is set
            to None, the callers can specify slim.batch_norm's is_training parameter
            from an outer slim.arg_scope.
          global_pool: If True, we perform global average pooling before computing the
            logits. Set to True for image classification, False for dense prediction.
          output_stride: If None, then the output will be computed at the nominal
            network stride. If output_stride is not None, it specifies the requested
            ratio of input to output spatial resolution.
          include_root_block: If True, include the initial convolution followed by
            max-pooling, if False excludes it.
          spatial_squeeze: if True, logits is of shape [B, C], if false logits is
              of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
              To use this parameter, the input images must be smaller than 300x300
              pixels, in which case the output logit layer does not contain spatial
              information and can be removed.
          store_non_strided_activations: If True, we compute non-strided (undecimated)
            activations at the last unit of each block and store them in the
            `outputs_collections` before subsampling them. This gives us access to
            higher resolution intermediate activations which are useful in some
            dense prediction problems but increases 4x the computation and memory cost
            at the last unit of each block.
          reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
          scope: Optional variable_scope.
        Returns:
          net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
            If global_pool is False, then height_out and width_out are reduced by a
            factor of output_stride compared to the respective height_in and width_in,
            else both height_out and width_out equal one. If num_classes is 0 or None,
            then net is the output of the last ResNet block, potentially after global
            average pooling. If num_classes a non-zero integer, net contains the
            pre-softmax activations.
          end_points: A dictionary from components of the network to the corresponding
            activation.
        Raises:
          ValueError: If the target output_stride is not valid.
        """
        self.net, self.end_points = nets.resnet_v1.resnet_v1_50(inputs,
                                                      num_classes,
                                                      is_training,
                                                      global_pool,
                                                      output_stride,
                                                      reuse,
                                                      scope
                                                      )

        return self.net, self.end_points

    _resnet_v1_50.default_image_size = 224

    def restore_params(self, sess, path='models'):
        logging.info("Restore pre-trained parameters for")
        maybe_download_and_extract(
            'resnet_v1_50_2016_08_28.tar.gz', path, 'http://download.tensorflow.org/models/',
            expected_bytes=95073259, extract=True
        )  # Download the file and extract
        params = load_ckpt(sess=sess, mode_name='resnet_v1_50.ckpt', var_list=self.net.all_params, save_dir=path, is_latest=False, printable=True)
        assign_params(sess, params[:len(self.net.all_params)], self.net)

