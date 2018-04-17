#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from contextlib import contextmanager


class CustomTestCase(unittest.TestCase):
    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type:
            raise self.failureException('{} raised'.format(exc_type.__name__))
