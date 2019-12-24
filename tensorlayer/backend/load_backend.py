#!/usr/bin/env python3
# -*- coding: utf-8 -*-

BACKEND = 'tensorflow'

if BACKEND == 'tensorflow':
    from .tensorflow_backend import *
    print('Using TensorFlow backend.\n')