#! /usr/bin/python
# -*- coding: utf-8 -*-

from .load_data_backend import *


class Dataset(object):

    def __init__(self):
        pass

    @staticmethod
    def from_generator(generator, output_types, output_shapes=None, args=None):
        return FromGenerator(generator, output_types, output_shapes=output_shapes, args=args)

    @staticmethod
    def map(ds, map_func, num_parallel_calls=None):
        return Map(ds=ds, map_func=map_func, num_parallel_calls=num_parallel_calls)

    @staticmethod
    def shuffle(ds, buffer_size, seed=None, reshuffle_each_iteration=None):
        return Shuffle(ds=ds, buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)

    @staticmethod
    def prefetch(ds, buffer_size):
        return Prefetch(ds=ds, buffer_size=buffer_size)

    @staticmethod
    def batch(ds, batch_size, drop_remainder=False):
        return Batch(ds=ds, batch_size=batch_size, drop_remainder=drop_remainder)

    @staticmethod
    def repeat(ds, count):
        return Repeat(ds, count=count)
