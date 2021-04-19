#! /usr/bin/python
# -*- coding: utf-8 -*-

import mindspore.dataset as ds
import mindspore as ms
from enum import Enum
__all__ = [
    'Apply',
    'Batch',
    'Concat',
    'CsvDataset',
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
    'TextFlieDataset',
    'TFRecordDataset',
]


class Shuffle(str, Enum):
    GLOBAL: str = "global"
    FILES: str = "file"


def Apply(dataset, transformation_func):

    return dataset.apply(transformation_func)


def Batch(
    dataset, batch_size, drop_remainder=False, num_parallel_workers=None, per_batch_map=None, inut_columns=None,
    output_columns=None, column_order=None, pad_info=None
):
    '''
    Combine batch_size number of consecutive rows into batches.
    Parameters
    ----------
    dataset
    batch_size
    drop_remainder
    num_parallel_workers
    per_batch_map
    inut_columns
    output_columns
    column_order
    pad_info

    Returns
    -------

    '''
    return dataset.batch(
        batch_size=batch_size, drop_remainder=drop_remainder, num_parallel_workers=num_parallel_workers,
        per_batch_map=per_batch_map, input_columns=inut_columns, output_columns=output_columns,
        column_order=column_order, pad_info=pad_info
    )


def Concat(dataset_1, dataset_2):

    return dataset_1.concat(dataset_2)


def CsvDataset(
    file_pattern, batch_size=1, column_names=None, column_defaults=None, label_name=None, select_columns=None,
    field_delim=',', use_quote_delim=True, na_value='', header=True, num_epochs=None, shuffle=Shuffle.GLOBAL,
    shuffle_buffer_size=10000, shuffle_seed=None, prefetch_buffer_size=None, num_parallel_reads=None, sloppy=False,
    num_rows_for_inference=100, compression_type=None, ignore_errors=False, numples_samples=None, num_shards=None,
    shard_id=None, cache=None
):
    """
        A source dataset that reads and parses comma-separated values (CSV) datasets.

     Examples:
        >>> import mindspore.dataset as dataset
        >>>
        >>> dataset_files = ["/path/to/1", "/path/to/2"] # contains 1 or multiple text files
        >>> dataset = dataset.CSVDataset(dataset_files=dataset_files, column_names=['col1', 'col2', 'col3', 'col4'])
    """
    return ds.CSVDataset(
        dataset_files=file_pattern, field_delim=field_delim, column_defaults=column_defaults, column_names=column_names,
        num_samples=numples_samples, num_parallel_workers=num_parallel_reads, shuffle=shuffle, num_shards=num_shards,
        shard_id=shard_id, cache=cache
    )


def Filter(dataset, predicate):

    return dataset.filter(predicate)


def Flat_map(dataset, map_func):

    return dataset.flat_map(map_func)


def FromGenerator(
    generator, output_types, output_shapes=None, args=None, column_names=None, column_types=None, schema=None,
    num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None, shard_id=None,
    python_multiprocessing=True
):

    return ds.GeneratorDataset(
        source=generator, column_names=column_names, column_types=column_types, schema=schema, num_samples=num_samples,
        num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler, num_shards=num_shards,
        shard_id=shard_id, python_multiprocessing=python_multiprocessing
    )


def FromSlices(
    tensor, column_names=None, num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None,
    shard_id=None
):

    return ds.NumpySlicesDataset(
        data=tensor, column_names=column_names, num_samples=num_samples, num_parallel_workers=num_parallel_workers,
        shuffle=shuffle, sampler=sampler, num_shards=num_shards, shard_id=shard_id
    )


def Map(
    dataset, map_func, num_parallel_calls=None, input_columns=None, output_columns=None, column_order=None,
    num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None
):
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
        operations=map_func, input_columns=input_columns, output_columns=output_columns, column_order=column_order,
        num_parallel_workers=num_parallel_workers, python_multiprocessing=python_multiprocessing, cache=cache,
        callbacks=callbacks
    )


def Prefetch(dataset, buffer_size):

    batch_size = dataset.get_batch_size()
    prefetch_size = batch_size * buffer_size

    return dataset.config.set_prefetch_size(prefetch_size)



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


def TextFlieDataset(
    filenames, compression_type=None, buffer_size=None, num_parallel_reads=None, num_samples=None, shuffle=None,
    num_shards=None, shard_id=None, cache=None
):
    """
    A source dataset that reads and parses datasets stored on disk in text format.
    The generated dataset has one column ['text'].

        Examples:
        >>> import mindspore.dataset as dataset
        >>>
        >>> dataset_files = ["/path/to/1", "/path/to/2"] # contains 1 or multiple text files
        >>> dataset = dataset.TextFileDataset(dataset_files=dataset_files)
    """
    if shuffle is None:
        shuffle = Shuffle.GLOBAL
    return ds.TextFileDataset(
        dataset_files=filenames, num_samples=num_samples, num_parallel_workers=num_parallel_reads, shuffle=shuffle,
        num_shards=num_shards, shard_id=shard_id, cache=cache
    )


def TFRecordDataset(
    filenames, compression_type=None, buffer_size=None, num_parallel_reads=None, schema=None, columns_list=None,
    num_samples=None, shuffle=None, num_shards=None, shard_id=None, shard_equal_rows=False, cache=None
):
    """
        A source dataset that reads and parses datasets stored on disk in TFData format.

 Examples:
        >>> import mindspore.dataset as dataset
        >>> import mindspore.common.dtype as mstype
        >>>
        >>> dataset_files = ["/path/to/1", "/path/to/2"] # contains 1 or multiple tf data files
        >>>
        >>> # 1) Get all rows from dataset_files with no explicit schema
        >>> # The meta-data in the first row will be used as a schema.
        >>> tfdataset = dataset.TFRecordDataset(dataset_files=dataset_files)
        >>>
        >>> # 2) Get all rows from dataset_files with user-defined schema
        >>> schema = dataset.Schema()
        >>> schema.add_column('col_1d', de_type=mindspore.int64, shape=[2])
        >>> tfdataset = dataset.TFRecordDataset(dataset_files=dataset_files, schema=schema)
        >>>
        >>> # 3) Get all rows from dataset_files with schema file "./schema.json"
        >>> tfdataset = dataset.TFRecordDataset(dataset_files=dataset_files, schema="./schema.json")
    """
    if shuffle is None:
        shuffle = Shuffle.GLOBAL
    return ds.TFRecordDataset(
        dataset_files=filenames, schema=schema, columns_list=columns_list, num_samples=num_samples,
        num_parallel_workers=num_parallel_reads, shuffle=shuffle, num_shards=num_shards, shard_id=shard_id,
        shard_equal_rows=shard_equal_rows, cache=cache
    )


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
