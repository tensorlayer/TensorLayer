#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

__all__ = ['FromGenerator', 'Map', 'Shuffle', 'Prefetch', 'Batch', 'Repeat']


def FromGenerator(generator, output_types, output_shapes=None, args=None):
    return tf.data.Dataset.from_generator(generator, output_types, output_shapes=output_shapes, args=args)


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
    return ds.map(map_func, num_parallel_calls=num_parallel_calls)


def Shuffle(ds, buffer_size, seed=None, reshuffle_each_iteration=None):
    return ds.shuffle(buffer_size, seed=seed, reshuffle_each_iteration=reshuffle_each_iteration)


def Prefetch(ds, buffer_size):
    return ds.prefetch(buffer_size=buffer_size)


def Batch(ds, batch_size, drop_remainder=False):
    return ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)


def Repeat(ds, count=None):
    return ds.repeat(count=count)
