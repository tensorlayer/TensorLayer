#! /usr/bin/python
# -*- coding: utf-8 -*-

import mindspore.dataset as dataset

__all__ = ['FromGenerator', 'Map', 'Shuffle', 'Prefetch', 'Batch', 'Repeat']


def FromGenerator(generator, output_types, output_shapes=None, args=None):
    pass


def Map(ds, map_func, num_parallel_calls=None):
    """ Maps map_func across the elements of this dataset.

    Parameters
    ----------
    ds : DataFlow
        input DataFlow
    map_func : function
        A function mapping a dataset element to another dataset element.
    num_parallel_calls

    Returns
    -------

    """
    pass


def Shuffle(ds, buffer_size, seed=None, reshuffle_each_iteration=None):
    pass


def Prefetch(ds, buffer_size):
    pass


def Batch(ds, batch_size, drop_remainder=False):
    pass


def Repeat(ds, count=None):
    pass
