#! /usr/bin/python
# -*- coding: utf-8 -*-

import mindspore.dataset as ds
import mindspore as ms
from enum import Enum
__all__ = [
    'Batch',
    'Concat',
    'FromGenerator',
    'FromSlices',
    'Map',
    'Repeat',
    'Shuffle',
    'Dataloader',
    'Dataset',
    'IterableDataset',
]


class Dataset(object):

    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__len__', self.__class__.__name__))


class IterableDataset(object):

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError("'{}' not implement in class " \
                                  "{}".format('__iter__', self.__class__.__name__))


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


def Concat(datasets):

    datasets = list(datasets)
    dataset = ds.Dataset.concat(datasets)
    return dataset


def FromGenerator(generator, output_types, column_names):

    output_types = list(output_types)
    column_names = list(column_names)
    return ds.GeneratorDataset(source=generator, column_names=column_names, column_types=output_types)


def FromSlices(datas, column_names):

    return ds.NumpySlicesDataset(data=datas, column_names=column_names)


def Map(dataset, map_func, input_columns=None):
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
    return dataset.map(operations=map_func, input_columns=input_columns)


def Repeat(dataset, count=None):

    return dataset.repeat(count)


def Shuffle(dataset, buffer_size):

    return dataset.shuffle(buffer_size)


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
    datasets = tuple(datasets)
    return ds.zip(datasets)


def Dataloader(dataset, batch_size, shuffle=False, drop_last=False, shuffle_buffer_size=10000):

    if shuffle:
        dataset = Shuffle(dataset, buffer_size=shuffle_buffer_size)
    dataset = Batch(dataset, batch_size=batch_size, drop_last=drop_last)

    return dataset
