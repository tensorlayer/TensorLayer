#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tensorlayer.decorators import private_method

from tests.utils import CustomTestCase


class Layer_Pooling_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        class MyClass(object):

            @private_method
            def _private_func(self):
                tl.logging.debug("I am private")

            def public_func(self):
                tl.logging.debug("I am public and calling now the private func")
                self._private_func()

        cls.my_object = MyClass()

    def test_call_from_public_method(self):
        with self.assertNotRaises(RuntimeError):
            self.my_object.public_func()

    def test_call_private_method(self):
        with self.assertRaises(RuntimeError):
            self.my_object._private_func()


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
