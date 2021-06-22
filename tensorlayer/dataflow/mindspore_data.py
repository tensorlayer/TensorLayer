#! /usr/bin/python
# -*- coding: utf-8 -*-

import mindspore.dataset as ds
import mindspore as ms
from enum import Enum
__all__ = [
    'Apply',
    'Batch',
    'Concat',
    'Filter',
    'Flat_map',
    'FromGenerator',
    'FromSlices',
    'Map',
    'Prefetch',
    'Repeat',
    'Shuffle',
    'Skip',
    'Take',
    'Dataloader',
]


class shuffle_str(str, Enum):
    GLOBAL: str = "global"
    FILES: str = "file"


def Apply(dataset, transformation_func):

    return dataset.apply(transformation_func)


def Batch(dataset, batch_size, drop_last=False):
    '''

    Parameters
    ----------
    dataset
    batch_size
    drop_last
    Returns
    -------

    '''
    return dataset.batch(batch_size=batch_size, drop_remainder=drop_last)


def Concat(dataset_1, dataset_2):

    return dataset_1.concat(dataset_2)


def Filter(dataset, predicate):

    return dataset.filter(predicate)


def Flat_map(dataset, map_func):

    return dataset.flat_map(map_func)


def FromGenerator(generator, transform = None):

    return ds.GeneratorDataset(source=generator, column_names=["data", "label"])


def FromSlices(
    tensor, column_names=None, num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None,
    shard_id=None
):

    return ds.NumpySlicesDataset(
        data=tensor, column_names=column_names, num_samples=num_samples, num_parallel_workers=num_parallel_workers,
        shuffle=shuffle, sampler=sampler, num_shards=num_shards, shard_id=shard_id
    )


def Map(
    dataset, map_func, input_columns=None):
    """ Maps map_func across the elements of this dataset.

    Parameters
    ----------
    dataset : DataFlow
        input DataFlow
    map_func : function
        A function mapping a dataset element to another dataset element.
    num_parallel_calls

    Returns
    -------

    """
    return dataset.map(
        operations=map_func, input_columns=input_columns
    )


def Prefetch(dataset, buffer_size):

    return dataset.config.set_prefetch_size(buffer_size)


def Repeat(dataset, count=None):

    return dataset.repeat(count)


def Shuffle(dataset, buffer_size, seed=None, reshuffle_each_iteration=None):

    #dataset.config.set_seed(seed)

    return dataset.shuffle(buffer_size)


def Skip(dataset, count):
    '''
    Creates a Dataset that skips count elements from this dataset.
    Parameters
    ----------
    dataset:
        A dataset
    count:
        A tf.int64 scalar tf.Tensor, representing the number of elements of this dataset that should be skipped to form the new dataset.


    Returns
    -------

    '''
    return dataset.skip(count)


def Take(dataset, count):
    '''
    Creates a Dataset with at most count elements from this dataset.
    Parameters
    ----------
    dataset:
        A dataset
    count:
        A tf.int64 scalar tf.Tensor, representing the number of elements of this dataset that should be taken to form the new dataset.
         If count is -1, or if count is greater than the size of this dataset, the new dataset will contain all elements of this dataset.
    Returns
    -------

    '''
    return dataset.take(count)

def Zip(datasets):
    '''
    Creates a Dataset by zipping together the given datasets.
    Parameters
    ----------
    datasets:
        A tuple of datasets to be zipped together.
    Returns
    -------

    '''
    return ds.zip(datasets)


def Dataloader(dataset, batch_size, shuffle=False, drop_last=False, prefetch=2, shuffle_buffer_size=10000):


    if shuffle:
        dataset = Shuffle(dataset, buffer_size=shuffle_buffer_size)
    dataset = Batch(dataset, batch_size=batch_size, drop_last=drop_last)


    return dataset
