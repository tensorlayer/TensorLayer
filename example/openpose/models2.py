import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

__all__ = ['model']

W_init = tf.truncated_normal_initializer(stddev=0.01)
b_init = tf.constant_initializer(value=0.0)


def stage(cnn, b1, b2, n_pos, maskInput1, maskInput2, is_train, name='stageX'):
    """ Define the archuecture of stage 2 to 6 """
    with tf.variable_scope(name):
        net = ConcatLayer([cnn, b1, b2], -1, name='concat')
        with tf.variable_scope("branch1"):
            b1 = Conv2d(net, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c1')
            b1 = Conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c2')
            b1 = Conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c3')
            b1 = Conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c4')
            b1 = Conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c5')
            b1 = Conv2d(b1, 128, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c6')
            b1 = Conv2d(b1, n_pos, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='conf')
            if is_train:
                b1.outputs = b1.outputs * maskInput1
        with tf.variable_scope("branch2"):
            b2 = Conv2d(net, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c1')
            b2 = Conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c2')
            b2 = Conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c3')
            b2 = Conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c4')
            b2 = Conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c5')
            b2 = Conv2d(b2, 128, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c6')
            b2 = Conv2d(b2, 38, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='pafs')
            if is_train:
                b2.outputs = b2.outputs * maskInput2
    return b1, b2

def ibn_b_block(nin, n_filter1, n_filter2, is_train, is_increase=False, use_in=True, name='x'):
    with tf.variable_scope(name+'_ibn_b'):
        n = Conv2d(nin, n_filter1, (1, 1), (1, 1), None, 'SAME', W_init=W_init, b_init=b_init, name='1x1conv')
        n = BatchNormLayer(n, is_train, act=tf.nn.relu, name='bn1')

        n = Conv2d(n, n_filter1, (3, 3), (1, 1), None, 'SAME', W_init=W_init, b_init=b_init, name='3x3conv')
        n = BatchNormLayer(n, is_train, act=tf.nn.relu, name='bn2')

        n = Conv2d(n, n_filter2, (1, 1), (1, 1), None, 'SAME', W_init=W_init, b_init=b_init, name='1x1conv_full')
        n = BatchNormLayer(n, is_train, act=None, name='bn3')

        if is_increase:
            nin = Conv2d(nin, n_filter2, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='conv_increase')
            nin = BatchNormLayer(nin, is_train, act=None, name='bn_increase')

        n = ElementwiseLayer([n, nin], tf.add, name='add')
        if use_in:
            n = InstanceNormLayer(n, act=tf.nn.relu, epsilon=1e-05, name='insnorm')
        else:
            n = LambdaLayer(n, tf.nn.relu, name='relu')
    return n

def model(x, n_pos, mask_miss1, mask_miss2, is_train=False, reuse=None):
    """ Defines the entire pose estimation model. """
    b1_list = []
    b2_list = []
    with tf.variable_scope('model', reuse):

        n = InputLayer(x, name='in')
        n = Conv2d(n, 32, (3, 3), (1, 1), None, 'SAME', W_init=W_init, b_init=b_init, name='conv1_1')
        n = BatchNormLayer(n, is_train, act=tf.nn.relu, name='bn1')
        n = Conv2d(n, 64, (3, 3), (1, 1), None, 'SAME', W_init=W_init, b_init=b_init, name='conv1_2')
        n = BatchNormLayer(n, is_train, act=tf.nn.relu, name='bn2')
        n = MaxPool2d(n, (3, 3), (2, 2), 'SAME', name='maxpool1')

        n = ibn_b_block(n, 64, 256, is_train, True, use_in=True, name='conv2_1')
        n = ibn_b_block(n, 64, 256, is_train, use_in=True, name='conv2_2')
        # n = ibn_b_block(n, 64, 256, is_train, use_in=True, name='conv2_3')
        n = MaxPool2d(n, (3, 3), (2, 2), 'SAME', name='maxpool2')

        n = ibn_b_block(n, 128, 512, is_train, True, use_in=True, name='conv3_1')
        n = ibn_b_block(n, 128, 512, is_train, use_in=True, name='conv3_2')
        n = ibn_b_block(n, 128, 512, is_train, use_in=True, name='conv3_3')
        n = ibn_b_block(n, 128, 512, is_train, use_in=True, name='conv3_4')
        n = MaxPool2d(n, (3, 3), (2, 2), 'SAME', name='maxpool3')

        n = ibn_b_block(n, 256, 512, is_train, use_in=False, name='conv4_1')
        # n = ibn_b_block(n, 256, 512, is_train, use_in=False, name='conv4_2')
        # n = ibn_b_block(n, 256, 512, is_train, use_in=False, name='conv4_3')
        cnn = ibn_b_block(n, 256, 512, is_train, use_in=False, name='conv4_4')

        net = Conv2d(cnn, 256, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='conv4_3')
        net = Conv2d(net, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='conv4_4')

        with tf.variable_scope('cpm', reuse):
            # stage 1
            with tf.variable_scope("stage1/branch1"):
                b1 = Conv2d(net, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c1')
                b1 = Conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c2')
                b1 = Conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c3')
                b1 = Conv2d(b1, 512, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c4')
                b1 = Conv2d(b1, n_pos, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='confs')
                if is_train:
                    b1.outputs = b1.outputs * mask_miss1
            with tf.variable_scope("stage1/branch2"):
                b2 = Conv2d(net, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c1')
                b2 = Conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c2')
                b2 = Conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c3')
                b2 = Conv2d(b2, 512, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c4')
                b2 = Conv2d(b2, 38, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='pafs')
                if is_train:
                    b2.outputs = b2.outputs * mask_miss2
            b1_list.append(b1)
            b2_list.append(b2)
            # stage 2~6
            for i in range(5, 7):
                b1, b2 = stage(cnn, b1_list[-1], b2_list[-1], n_pos, mask_miss1, mask_miss2, is_train, name='stage%d' % i)
                b1_list.append(b1)
                b2_list.append(b2)
        net = tl.layers.merge_networks([b1_list[-1], b2_list[-1]])
        return cnn, b1_list, b2_list, net
