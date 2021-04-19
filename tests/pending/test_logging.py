#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class TL_Logger_Test(CustomTestCase):

    def test_debug(self):
        with self.assertNotRaises(Exception):
            tl.logging.debug("This is a debug message")

    def test_error(self):
        with self.assertNotRaises(Exception):
            tl.logging.error("This is an error message")

    def test_fatal(self):
        with self.assertNotRaises(Exception):
            tl.logging.fatal("This is a fatal error message")

    def test_info(self):
        with self.assertNotRaises(Exception):
            tl.logging.info("This is an information message")

    def test_warn(self):
        with self.assertNotRaises(Exception):
            tl.logging.warn("This is a warning message")

    def test_set_verbosity(self):
        with self.assertNotRaises(Exception):
            tl.logging.set_verbosity(tl.logging.DEBUG)
            tl.logging.set_verbosity(tl.logging.INFO)
            tl.logging.set_verbosity(tl.logging.WARN)
            tl.logging.set_verbosity(tl.logging.ERROR)
            tl.logging.set_verbosity(tl.logging.FATAL)

    def test_get_verbosity(self):
        with self.assertNotRaises(Exception):
            tl.logging.get_verbosity()


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
