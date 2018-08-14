import tensorflow as tf
from tensorlayer.layers import (Conv2d, ConcatLayer, DropoutLayer)
import tensorlayer as tl
from tensorlayer.layers import *

__all__ = ['model']
# _init_norm = tf.truncated_normal_initializer(stddev=0.01)
_init_xavier = tf.contrib.layers.xavier_initializer()
_init_norm = tf.truncated_normal_initializer(stddev=0.01)
_b_initer = tf.constant_initializer(value=0.0)

# _init_norm = tf.contrib.layers.xavier_initializer()
# _b_initer = tf.contrib.layers.xavier_initializer()


def _stage(cnn, b1, b2, n_pos, maskInput1, maskInput2, name='stageX'):
    with tf.variable_scope(name):
        net = ConcatLayer([cnn, b1, b2], -1, name='concat')
        with tf.variable_scope("branch1"):
            b1 = Conv2d(net, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c1')
            b1 = Conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c2')
            b1 = Conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c3')
            b1 = Conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c4')
            b1 = Conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c5')
            b1 = Conv2d(b1, 128, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=_init_norm, b_init=_b_initer, name='c6')
            b1 = Conv2d(b1, n_pos, (1, 1), (1, 1), None, 'VALID', W_init=_init_norm, b_init=_b_initer, name='conf')
            b1.outputs = b1.outputs * maskInput1
        with tf.variable_scope("branch2"):
            b2 = Conv2d(net, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c1')
            b2 = Conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c2')
            b2 = Conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c3')
            b2 = Conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c4')
            b2 = Conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c5')
            b2 = Conv2d(b2, 128, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=_init_norm, b_init=_b_initer, name='c6')
            b2 = Conv2d(b2, 38, (1, 1), (1, 1), None, 'VALID', W_init=_init_norm, b_init=_b_initer, name='pafs')
            b2.outputs = b2.outputs * maskInput2
    return b1, b2


def vgg_network(x):
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x)
    bgr = tf.concat(axis=3, values=[blue, green, red])
    bgr = bgr - 0.5
    # input layer
    net_in = InputLayer(bgr, name='input')
    # conv1
    net = Conv2d(net_in, 64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
    net = Conv2d(net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
    # conv2
    net = Conv2d(net, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
    net = Conv2d(net, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
    # conv3
    net = Conv2d(net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
    net = Conv2d(net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
    net = Conv2d(net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
    net = Conv2d(net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
    net = MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
    # conv4
    net = Conv2d(net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
    net = Conv2d(net, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
    net = Conv2d(
        net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=_init_norm,
        b_init=_b_initer, name='conv4_3'
    )
    net = Conv2d(
        net, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', W_init=_init_norm,
        b_init=_b_initer, name='conv4_4'
    )

    return net


def model(
        x,
        n_pos,
        mask_miss1,
        mask_miss2,
        is_train=False,
        reuse=None,
):
    b1_list = []
    b2_list = []
    with tf.variable_scope('model', reuse):
        # feature extraction part
        # cnn = tl.models.MobileNetV1(x, end_with='depth5', is_train=is_train, reuse=reuse)  # i.e. vgg16 conv4_2 ~ 4_4
        cnn = vgg_network(x)
        with tf.variable_scope('cpm', reuse):
            # stage 1
            with tf.variable_scope("stage1"):
                with tf.variable_scope("branch1"):
                    b1 = Conv2d(
                        cnn, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c1'
                    )
                    b1 = Conv2d(
                        b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c2'
                    )
                    b1 = Conv2d(
                        b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c3'
                    )
                    b1 = Conv2d(
                        b1, 512, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=_init_norm, b_init=_b_initer, name='c4'
                    )
                    b1 = Conv2d(
                        b1, n_pos, (1, 1), (1, 1), None, 'VALID', W_init=_init_norm, b_init=_b_initer, name='confs'
                    )
                    b1.outputs = b1.outputs * mask_miss1
                with tf.variable_scope("branch2"):
                    b2 = Conv2d(
                        cnn, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c1'
                    )
                    b2 = Conv2d(
                        b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c2'
                    )
                    b2 = Conv2d(
                        b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=_init_norm, b_init=_b_initer, name='c3'
                    )
                    b2 = Conv2d(
                        b2, 512, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=_init_norm, b_init=_b_initer, name='c4'
                    )
                    b2 = Conv2d(b2, 38, (1, 1), (1, 1), None, 'VALID', W_init=_init_norm, b_init=_b_initer, name='pafs')
                    b2.outputs = b2.outputs * mask_miss2
                b1_list.append(b1)
                b2_list.append(b2)
            # stage 2~6
            for i in range(2, 7):
                b1, b2 = _stage(cnn, b1_list[-1], b2_list[-1], n_pos, mask_miss1, mask_miss2, name='stage%d' % i)
                b1_list.append(b1)
                b2_list.append(b2)
        net = tl.layers.merge_networks([b1_list[-1], b2_list[-1]])
        return cnn, b1_list, b2_list, net
