#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import paddle
from paddle.io import Dataset, BatchSampler, DataLoader, IterableDataset
__all__ = [
    'Concat',
    'FromGenerator',
    'FromSlices',
    'Map',
    # 'Shuffle',
    # 'Batch',
    'Dataloader',
]


def to_list(value):
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class FromGenerator(Dataset):

    def __init__(self, generator):

        if not callable(generator):
            raise TypeError("'generator' must be callable")
        self.generator = generator()
        self.datas = []
        self.labels = []
        for data, label in self.generator:
            self.datas.append(data)
            self.labels.append(label)

    def __getitem__(self, idx):

        x = self.datas[idx]
        y = self.labels[idx]

        return x, y

    def __len__(self):

        return self.datas.shape[0]


class FromSlices(Dataset):

    def __init__(self, datas, transform=None):
        self.datas = datas[0]
        self.labels = datas[1]
        self.transform = transform

        if len(self.datas) != len(self.labels):
            raise ValueError('Datas and labels not have same shape of the 1st dimension.')

    def __getitem__(self, idx):
        data = paddle.to_tensor(self.datas[idx], dtype='float32')
        label = paddle.to_tensor(self.labels[idx], dtype='int64')
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):

        return len(self.datas)


class Concat(IterableDataset):

    def __init__(self, datasets):
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "input datasets shoule not be empty"
        for i, dataset in enumerate(self.datasets):
            assert isinstance(dataset, IterableDataset), \
                "ChainDataset only support paddle.io.IterableDataset"

    def __iter__(self):
        for dataset in self.datasets:
            for sample in dataset:
                yield sample


class Map(Dataset):

    def __init__(self, dataset, transform):
        self.isDataset = False
        self.transform = transform
        if isinstance(dataset, Dataset):
            self.isDataset = True
            self.dataset = dataset
        elif isinstance(dataset, list) or isinstance(dataset, tuple):
            self.datas = dataset[0]
            self.labels = dataset[1]
        else:
            raise TypeError(
                " 'dataset' should be subclass instance of paddle.io.Dataset "
                "or a [data, label] list/tulpe, not a {}".format(type(dataset))
            )

    def __getitem__(self, idx):
        if self.isDataset:
            x = self.dataset[idx][0]
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            x = self.transform(x)
            y = self.dataset[idx][1]
        else:
            x = self.datas[idx]
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            x = self.transform(x)
            y = self.labels[idx]

        return x, y

    def __len__(self):

        if self.isDataset:
            return len(self.dataset[0])
        else:
            return len(self.datas)


def Dataloader(dataset, batch_size=None, shuffle=False, drop_last=False, prefetch=0, shuffle_buffer_size=0):

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
