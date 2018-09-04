#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

import numpy as np

from tests.utils import CustomTestCase


class Array_Op_Alphas_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        b1 = tl.alphas([4, 3, 2, 1], 0.5431)
        b2 = tl.alphas([4, 3, 2], 5)
        b3 = tl.alphas([1, 2, 3, 4], -5)
        b4 = tl.alphas([2, 3, 4], True)

        with tf.Session() as sess:
            cls._b1, cls._b2, cls._b3, cls._b4 = sess.run([b1, b2, b3, b4])

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_b1(self):
        self.assertEqual(self._b1.shape, (4, 3, 2, 1))

        b1 = np.array(
            [
                [
                    [
                        [0.5431],
                        [0.5431],
                    ],
                    [
                        [0.5431],
                        [0.5431],
                    ],
                    [
                        [0.5431],
                        [0.5431],
                    ],
                ], [
                    [
                        [0.5431],
                        [0.5431],
                    ],
                    [
                        [0.5431],
                        [0.5431],
                    ],
                    [
                        [0.5431],
                        [0.5431],
                    ],
                ], [
                    [
                        [0.5431],
                        [0.5431],
                    ],
                    [
                        [0.5431],
                        [0.5431],
                    ],
                    [
                        [0.5431],
                        [0.5431],
                    ],
                ], [
                    [
                        [0.5431],
                        [0.5431],
                    ],
                    [
                        [0.5431],
                        [0.5431],
                    ],
                    [
                        [0.5431],
                        [0.5431],
                    ],
                ]
            ]
        )

        np.array_equal(self._b1, b1)

    def test_b2(self):
        self.assertEqual(self._b2.shape, (4, 3, 2))

        b2 = np.array(
            [
                [
                    [
                        5,
                        5,
                    ],
                    [
                        5,
                        5,
                    ],
                    [
                        5,
                        5,
                    ],
                ], [
                    [
                        5,
                        5,
                    ],
                    [
                        5,
                        5,
                    ],
                    [
                        5,
                        5,
                    ],
                ], [
                    [
                        5,
                        5,
                    ],
                    [
                        5,
                        5,
                    ],
                    [
                        5,
                        5,
                    ],
                ], [
                    [
                        5,
                        5,
                    ],
                    [
                        5,
                        5,
                    ],
                    [
                        5,
                        5,
                    ],
                ]
            ]
        )

        np.array_equal(self._b2, b2)

    def test_b3(self):
        self.assertEqual(self._b3.shape, (1, 2, 3, 4))

        b3 = np.array(
            [
                [
                    [[-5, -5, -5, -5], [-5, -5, -5, -5], [-5, -5, -5, -5]],
                    [[-5, -5, -5, -5], [-5, -5, -5, -5], [-5, -5, -5, -5]],
                ]
            ]
        )

        np.array_equal(self._b3, b3)

    def test_b4(self):
        self.assertEqual(self._b4.shape, (2, 3, 4))

        b4 = np.array(
            [
                [[True, True, True, True], [True, True, True, True], [True, True, True, True]],
                [[True, True, True, True], [True, True, True, True], [True, True, True, True]],
            ]
        )

        np.array_equal(self._b4, b4)


class Array_Op_Alphas_Like_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        a = tf.constant([[[4, 5, 6], [1, 2, 3]], [[4, 5, 6], [1, 2, 3]]])

        b1 = tl.alphas_like(a, 0.5431)
        b2 = tl.alphas_like(a, 5)
        b3 = tl.alphas_like(a, -5)
        b4 = tl.alphas_like(a, True)

        with tf.Session() as sess:
            cls._b1, cls._b2, cls._b3, cls._b4 = sess.run([b1, b2, b3, b4])

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_b1(self):
        self.assertEqual(self._b1.shape, (2, 2, 3))

        b1 = np.array(
            [
                [[0.5431, 0.5431, 0.5431], [0.5431, 0.5431, 0.5431]],
                [[0.5431, 0.5431, 0.5431], [0.5431, 0.5431, 0.5431]]
            ]
        )

        np.array_equal(self._b1, b1)

    def test_b2(self):
        self.assertEqual(self._b2.shape, (2, 2, 3))

        b2 = np.array([[[5, 5, 5], [5, 5, 5]], [[5, 5, 5], [5, 5, 5]]])

        np.array_equal(self._b2, b2)

    def test_b3(self):
        self.assertEqual(self._b3.shape, (2, 2, 3))

        b3 = np.array([[[-5, -5, -5], [-5, -5, -5]], [[-5, -5, -5], [-5, -5, -5]]])

        np.array_equal(self._b3, b3)

    def test_b4(self):
        self.assertEqual(self._b4.shape, (2, 2, 3))

        b4 = np.array([[[True, True, True], [True, True, True]], [[True, True, True], [True, True, True]]])

        np.array_equal(self._b4, b4)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
