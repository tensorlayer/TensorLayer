import unittest

try:
    from tests.unittests_helper import CustomTestCase
except ImportError:
    from unittests_helper import CustomTestCase

# To be removed later
import sys
sys.path.insert(0, './')

import tensorflow as tf
import tensorlayer as tl


class Layer_Data_Format_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        x_1d = tf.placeholder(tf.float32, [None, 5, 1])
        cls.input_1d = tl.layers.InputLayer(x_1d)

        x_2d = tf.placeholder(tf.float32, [None, 5, 5, 1])
        cls.input_2d = tl.layers.InputLayer(x_2d)

        x_3d = tf.placeholder(tf.float32, [None, 5, 6, 7, 3])
        cls.input_3d = tl.layers.InputLayer(x_3d)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_Conv1dLayer_NWC(self):
        with self.assertNotRaises(Exception):
            with tf.variable_scope('test_Conv1dLayer_NWC', reuse=False):
                tl.layers.Conv1dLayer(self.input_1d, data_format="channels_last")

    def test_Conv1d_NWC(self):
        with self.assertNotRaises(Exception):
            with tf.variable_scope('test_Conv1d_NWC', reuse=False):
                tl.layers.Conv1d(self.input_1d, data_format="channels_last")

    def test_Conv2dLayer_NHWC(self):
        with self.assertNotRaises(Exception):
            with tf.variable_scope('test_Conv2dLayer_NHWC', reuse=False):
                tl.layers.Conv2dLayer(self.input_2d, data_format="NHWC")

    def test_Conv2d_NHWC(self):
        with self.assertNotRaises(Exception):
            with tf.variable_scope('test_Conv2d_NHWC', reuse=False):
                tl.layers.Conv2d(self.input_2d, data_format="channels_last")

    def test_Conv3dLayer_NDHWC(self):
        with self.assertNotRaises(Exception):
            with tf.variable_scope('test_Conv3dLayer_1', reuse=False):
                tl.layers.Conv3dLayer(self.input_3d, data_format="NDHWC")


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    unittest.main()
