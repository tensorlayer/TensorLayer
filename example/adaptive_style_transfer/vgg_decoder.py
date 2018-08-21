# Decoder mostly mirrors the encoder with all pooling layers replaced by nearest
# up-sampling to reduce checker-board effects.
# Decoder has no BN/IN layers.

# DECODER_LAYERS = (
#     'conv4_1', 'relu4_1', 'upsample', 'conv3_4', 'relu3_4',
#
#     'conv3_3', 'relu3_3', 'conv3_2',  'relu3_2', 'conv3_1',
#
#     'relu3_1', 'upsample', 'conv2_2', 'relu2_2', 'conv2_1',
#
#     'relu2_1', 'upsample', 'conv1_2',  'relu1_2', 'conv1_1',
# )

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

WEIGHT_INIT_STDDEV = 0.1


class Decoder(object):

    @classmethod
    def decode(self, image, prefix):
        """
        Build the VGG 19 Model
        Parameters
        -----------
        rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
        """
        # conv4_1
        net_in = InputLayer(image, name='input')
        net = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        net = Conv2dLayer(
            net, act=tf.nn.relu, shape=[3, 3, 512, 256], strides=[1, 1, 1, 1], padding='VALID', name=prefix + 'conv4_1'
        )
        net = UpSampling2dLayer(net, [2, 2], method=1)

        # conv3
        net = PadLayer(net, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        net = Conv2dLayer(
            net, act=tf.nn.relu, shape=[3, 3, 256, 256], strides=[1, 1, 1, 1], padding='VALID', name=prefix + 'conv3_4'
        )
        net = PadLayer(net, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        net = Conv2dLayer(
            net, act=tf.nn.relu, shape=[3, 3, 256, 256], strides=[1, 1, 1, 1], padding='VALID', name=prefix + 'conv3_3'
        )
        net = PadLayer(net, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        net = Conv2dLayer(
            net, act=tf.nn.relu, shape=[3, 3, 256, 256], strides=[1, 1, 1, 1], padding='VALID', name=prefix + 'conv3_2'
        )
        net = PadLayer(net, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        net = Conv2dLayer(
            net, act=tf.nn.relu, shape=[3, 3, 256, 128], strides=[1, 1, 1, 1], padding='VALID', name=prefix + 'conv3_1'
        )
        net = UpSampling2dLayer(net, [2, 2], method=1)

        # conv2
        net = PadLayer(net, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        net = Conv2dLayer(
            net, act=tf.nn.relu, shape=[3, 3, 128, 128], strides=[1, 1, 1, 1], padding='VALID', name=prefix + 'conv2_2'
        )
        net = PadLayer(net, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        net = Conv2dLayer(
            net, act=tf.nn.relu, shape=[3, 3, 128, 64], strides=[1, 1, 1, 1], padding='VALID', name=prefix + 'conv2_1'
        )
        net = UpSampling2dLayer(net, [2, 2], method=1)

        # conv1
        net = PadLayer(net, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        net = Conv2dLayer(
            net, act=tf.nn.relu, shape=[3, 3, 64, 64], strides=[1, 1, 1, 1], padding='VALID', name=prefix + 'conv1_2'
        )
        net = PadLayer(net, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        net = Conv2dLayer(
            net, act=None, shape=[3, 3, 64, 3], strides=[1, 1, 1, 1], padding='VALID', name=prefix + 'conv1_1'
        )

        print("build Decoder model finished:")

        return net

    @classmethod
    def restore_model(self, sess, weight_path, net):
        tl.files.load_and_assign_npz(sess, weight_path, net)
        print("Restored decoder model from npy file")
