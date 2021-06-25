#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import paddle
from paddle.io import Dataset as dataset
from paddle.io import IterableDataset as iterabledataset
from paddle.io import DataLoader
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


class Dataset(dataset):

    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__len__', self.__class__.__name__))


class IterableDataset(iterabledataset):

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__iter__', self.__class__.__name__))

    def __getitem__(self, idx):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                "{}".format('__len__', self.__class__.__name__))


def FromGenerator(generator, output_types=None, column_names=None):

    return generator


def FromSlices(datas, column_names=None):

    datas = list(datas)
    return paddle.io.TensorDataset(datas)


def Concat(datasets):

    return paddle.io.ChainDataset(list(datasets))


def Zip(datasets):

    return paddle.io.ComposeDataset(list(datasets))


def Dataloader(dataset, batch_size=None, shuffle=False, drop_last=False, shuffle_buffer_size=0):

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, return_list=True)


def Batch(dataset, batch_size, drop_last=False):

    raise NotImplementedError('This function not implement in paddle backend.')


def Shuffle(dataset, buffer_size, seed=None):

    raise NotImplementedError('This function not implement in paddle backend.')


def Repeat(dataset, count=None):

    raise NotImplementedError('This function not implement in paddle backend.')


def Map(dataset, map_func, input_columns=None):

    raise NotImplementedError('This function not implement in paddle backend.')
