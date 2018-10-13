import os
import unittest
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorlayer as tl
from tests.utils import CustomTestCase


class PReLU_Layer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        x = tf.placeholder(tf.float32, shape=[None, 30])

        in_layer = tl.layers.Input()(x)

        net = tl.layers.Dense(n_units=10, name='dense_1')(in_layer)
        cls.net1 = tl.layers.PRelu(name='prelu_1')(net)

        # cls.net1.print_layers()
        # cls.net1.print_params(False)

        net2 = tl.layers.Dense(n_units=30, name='dense_2')(cls.net1)
        cls.net2 = tl.layers.PRelu(channel_shared=True, name='prelu_2')(net2)

        # cls.net2.print_layers()
        # cls.net2.print_params(False)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):
        # self.assertEqual(len(self.net1.all_layers), 3)
        # self.assertEqual(len(self.net1.all_weights), 3)
        # self.assertEqual(self.net1.count_weights(), 320)

        self.assertEqual(self.net1.outputs.get_shape().as_list()[1:], [10])

        prelu1_param_shape = self.net1.local_weights[0].get_shape().as_list()
        self.assertEqual(prelu1_param_shape, [10])

    def test_net2(self):
        # self.assertEqual(len(self.net2.all_layers), 5)
        # self.assertEqual(len(self.net2.all_weights), 6)
        # self.assertEqual(self.net2.count_weights(), 651)

        self.assertEqual(self.net2.outputs.get_shape().as_list()[1:], [30])

        prelu2_param_shape = self.net2.local_weights[0].get_shape().as_list()
        self.assertEqual(prelu2_param_shape, [1])


class PRelu6_Layer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        x = tf.placeholder(tf.float32, shape=[None, 30])

        in_layer = tl.layers.Input()(x)

        net = tl.layers.Dense(n_units=10, name='dense_1')(in_layer)
        cls.net1 = tl.layers.PRelu6(name='prelu6_1')(net)

        # cls.net1.print_layers()
        # cls.net1.print_params(False)

        net2 = tl.layers.Dense(n_units=30, name='dense_2')(cls.net1)
        cls.net2 = tl.layers.PRelu6(channel_shared=True, name='prelu6_2')(net2)

        # cls.net2.print_layers()
        # cls.net2.print_params(False)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):
        # self.assertEqual(len(self.net1.all_layers), 3)
        # self.assertEqual(len(self.net1.all_weights), 3)
        # self.assertEqual(self.net1.count_weights(), 320)

        self.assertEqual(self.net1.outputs.get_shape().as_list()[1:], [10])

        prelu1_param_shape = self.net1.local_weights[0].get_shape().as_list()
        self.assertEqual(prelu1_param_shape, [10])

    def test_net2(self):
        # self.assertEqual(len(self.net2.all_layers), 5)
        # self.assertEqual(len(self.net2.all_weights), 6)
        # self.assertEqual(self.net2.count_weights(), 651)

        self.assertEqual(self.net2.outputs.get_shape().as_list()[1:], [30])

        prelu2_param_shape = self.net2.local_weights[0].get_shape().as_list()
        self.assertEqual(prelu2_param_shape, [1])


class PTRelu6_Layer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        x = tf.placeholder(tf.float32, shape=[None, 30])

        in_layer = tl.layers.Input()(x)

        net = tl.layers.Dense(n_units=10, name='dense_1')(in_layer)
        cls.net1 = tl.layers.PTRelu6(name='ptrelu6_1')(net)

        # cls.net1.print_layers()
        # cls.net1.print_params(False)

        net2 = tl.layers.Dense(n_units=30, name='dense_2')(cls.net1)
        cls.net2 = tl.layers.PTRelu6(channel_shared=True, name='ptrelu6_2')(net2)

        # cls.net2.print_layers()
        # cls.net2.print_params(False)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):
        # self.assertEqual(len(self.net1.all_layers), 3)
        # self.assertEqual(len(self.net1.all_weights), 4)
        # self.assertEqual(self.net1.count_weights(), 330)

        self.assertEqual(self.net1.outputs.get_shape().as_list()[1:], [10])

        prelu1_param_shape = self.net1.local_weights[0].get_shape().as_list()
        self.assertEqual(prelu1_param_shape, [10])

    def test_net2(self):
        # self.assertEqual(len(self.net2.all_layers), 5)
        # self.assertEqual(len(self.net2.all_weights), 8)
        # self.assertEqual(self.net2.count_weights(), 662)

        self.assertEqual(self.net2.outputs.get_shape().as_list()[1:], [30])

        prelu2_param_shape = self.net2.local_weights[0].get_shape().as_list()
        self.assertEqual(prelu2_param_shape, [1])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
