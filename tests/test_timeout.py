#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import WindowsError
from tests.utils import TimeoutError

from tests.utils import TimeoutContext
from tests.utils import CustomTestCase

from tests.utils.custom_networks import InceptionV4_Network

if os.getenv("TRAVIS", None) is not None:
    NETWORK_CREATION_TIMEOUT = 120  # Seconds before timeout
else:
    NETWORK_CREATION_TIMEOUT = 40  # Seconds before timeout

######################################################################################
#                                                                                    #
#                                UNITTEST TIMEOUT                                    #
#                                                                                    #
######################################################################################


class Layer_Timeoutt_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        #######################################################################
        ####  =============    Placeholders Declaration      ============= ####
        #######################################################################

        cls.input_plh = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_placeholder')

        #######################################################################
        ####  =============        Model Declaration         ============= ####
        #######################################################################

        cls.inception_v4_net = InceptionV4_Network(include_FC_head=True, flatten_output=False)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_timeout_not_reuse(self):

        with self.assertNotRaises(TimeoutError):
            try:
                with TimeoutContext(NETWORK_CREATION_TIMEOUT):
                    start_time = time.time()

                    _ = self.inception_v4_net(self.input_plh, reuse=False, is_train=False)

                    tl.logging.info("Seconds Elapsed [Not Reused]: %d" % int(time.time() - start_time))

            except WindowsError:
                tl.logging.warning("This unittest can not run on Windows")

    def test_timeout_reuse(self):

        with self.assertNotRaises(TimeoutError):
            try:
                with TimeoutContext(NETWORK_CREATION_TIMEOUT):
                    start_time = time.time()

                    _ = self.inception_v4_net(self.input_plh, reuse=True, is_train=False)

                    tl.logging.info("Seconds Elapsed [Reused Model]: %d" % int(time.time() - start_time))

            except WindowsError:
                tl.logging.warning("This unittest can not run on Windows")


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
