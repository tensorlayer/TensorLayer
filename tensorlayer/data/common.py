import math
import multiprocessing
import os

import tensorflow as tf
import zmq

import numpy as np

from .base import IndexableDatasetWrapper, DatasetWrapper, _Transforms_for_tf_dataset
from .parallel import _get_pipe_name, ZMQMultiprocessDataset, MultiprocessDataset
from .utils import clean_up_socket_files
from .serialize import convert_to_bytes, load_from_bytes

__all__ = ['PrefetchBatchedDataset', 'TransformedDataset', 'ShuffledDataset',
           'AugmentedDataset']


class BatchedDataset(DatasetWrapper):
    def __init__(self,
                 ds,
                 batch_size,
                 drop_remainder=True,
                 return_numpy=True,
                 output_types=None):
        super(BatchedDataset, self).__init__(ds)
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.return_numpy = return_numpy
        self.output_types = output_types

    def __iter__(self):
        dp_buffer = []
        for dp in self.ds:
            dp_buffer.append(dp)
            if len(dp_buffer) == self.batch_size:
                yield self._batch_datapoints(dp_buffer, self.return_numpy, self.output_types)
                del dp_buffer[:]
        if not self.drop_remainder:
            self._batch_datapoints(dp_buffer, self.return_numpy, self.output_types)

    def __len__(self):
        ds_len = len(self.ds)
        if self.drop_remainder:
            return ds_len // self.batch_size
        else:
            return math.ceil(ds_len / self.batch_size)

    @staticmethod
    def _batch_datapoints(dp_buffer, return_numpy, output_types):
        """

        :param dp_buffer: a list of datapoints
        :return:
        """
        first_dp = dp_buffer[0]
        if isinstance(first_dp, (tuple, list)):
            dp_batch = [None] * len(first_dp)
            for i in range(len(first_dp)):
                dp_element_batch = []
                for j in range(len(dp_buffer)):
                    dp_element_batch.append(dp_buffer[j][i])
                if return_numpy:
                    dp_batch[i] = BatchedDataset._batch_ndarray(dp_element_batch,
                                                                dtype=BatchedDataset._get_element_dtype(output_types,
                                                                                                        i))
                else:
                    dp_batch[i] = dp_element_batch
            return dp_batch
        elif isinstance(first_dp, dict):
            dp_batch = {}
            for key in first_dp.keys():
                dp_element_batch = []
                for j in range(len(dp_buffer)):
                    dp_element_batch.append(dp_buffer[j][key])
                if return_numpy:
                    dp_batch[key] = BatchedDataset._batch_ndarray(dp_element_batch, dtype=None)
                else:
                    dp_batch[key] = dp_element_batch
            return dp_batch
        elif isinstance(first_dp, np.ndarray):
            return BatchedDataset._batch_ndarray(dp_buffer)
        # single elements
        else:
            if return_numpy:
                return BatchedDataset._batch_ndarray(dp_buffer,
                                                     dtype=BatchedDataset._get_element_dtype(output_types, 0))
            else:
                return dp_buffer

    @staticmethod
    def _batch_ndarray(dp_element_batch, dtype):
        """

        :param dp_element_batch: a list of datapoint element, an element can be np.ndarray / list
        :return: np.ndarray, type is the same as input
        """
        try:
            if dtype is not None:
                ret = np.asarray(dp_element_batch, dtype=dtype)
            else:
                ret = np.asarray(dp_element_batch)
            return ret
        except:
            raise ValueError("Unsupported type for batching.")

    @staticmethod
    def _get_element_dtype(output_types, i):
        if output_types is None:
            return None
        if not isinstance(output_types, (tuple, list)):
            return output_types
        if len(output_types) == 1:
            return output_types[0]
        return output_types[i]


class PrefetchBatchedDataset(DatasetWrapper):
    def __init__(self,
                 ds,
                 batch_size,
                 drop_remainder=True,
                 return_numpy=True,
                 output_types=None,
                 use_zmq=True):
        super(PrefetchBatchedDataset, self).__init__(ds)
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.return_numpy = return_numpy
        self.output_types = output_types
        self.use_zmq = use_zmq

        if self.use_zmq:
            self.data_pipename = _get_pipe_name('batch_prefetch')
            context = zmq.Context()
            self.fetch_data_socket = context.socket(zmq.PULL)
            self.fetch_data_socket.set_hwm(1)
            self.fetch_data_socket.bind(self.data_pipename)
            self.worker = multiprocessing.Process(target=self._ZMQ_BatchedDataset_worker,
                                                  args=(self.ds,))
            self.worker.daemon = True
            self.worker.start()
            clean_up_socket_files(self.data_pipename)
        else:
            pipe_output, pipe_input = multiprocessing.Pipe()
            self.worker = multiprocessing.Process(target=self._BatchedDataset_worker,
                                                  args=(self.ds, (pipe_output, pipe_input)))
            self.worker.daemon = True
            self.worker.start()
            # main process only reads (gets output)
            pipe_input.close()
            self.pipe_output = pipe_output

    def _ZMQ_BatchedDataset_worker(self, ds):
        context = zmq.Context()
        prepare_data_socket = context.socket(zmq.PUSH)
        prepare_data_socket.set_hwm(1)
        prepare_data_socket.connect(self.data_pipename)
        while True:
            dp_buffer = []
            for dp in ds:
                dp_buffer.append(dp)
                if len(dp_buffer) == self.batch_size:
                    prepare_data_socket.send(convert_to_bytes(
                        BatchedDataset._batch_datapoints(dp_buffer, self.return_numpy, self.output_types)), copy=False)
                    del dp_buffer[:]
            if not self.drop_remainder:
                prepare_data_socket.send(
                    convert_to_bytes(BatchedDataset._batch_datapoints(dp_buffer, self.return_numpy, self.output_types)),
                    copy=False)

    def _BatchedDataset_worker(self, ds, pipe):
        pipe_output, pipe_input = pipe
        # worker process only writes (puts input)
        pipe_output.close()
        while True:
            dp_buffer = []
            for dp in ds:
                dp_buffer.append(dp)
                if len(dp_buffer) == self.batch_size:
                    pipe_input.send(BatchedDataset._batch_datapoints(dp_buffer, self.return_numpy, self.output_types))
                    del dp_buffer[:]
            if not self.drop_remainder:
                pipe_input.send(BatchedDataset._batch_datapoints(dp_buffer, self.return_numpy, self.output_types))

    def __iter__(self):
        for _ in range(self.__len__()):
            # yield self.q.get()
            if self.use_zmq:
                yield load_from_bytes(self.fetch_data_socket.recv(copy=False))
            else:
                yield self.pipe_output.recv()

    def __len__(self):
        ds_len = len(self.ds)
        if self.drop_remainder:
            return ds_len // self.batch_size
        else:
            return math.ceil(ds_len / self.batch_size)


class ShuffledDataset(DatasetWrapper):
    def __init__(self, ds):
        super(ShuffledDataset, self).__init__(ds)

    def __iter__(self):
        self.shuffled_idxs = np.random.permutation(len(self.ds))
        for index, data in enumerate(self.ds):
            yield self.ds[self.shuffled_idxs[index]]


class TransformedDataset(IndexableDatasetWrapper):
    """

    """

    def __init__(self, ds, transforms):
        super(TransformedDataset, self).__init__(ds)
        self.transforms = transforms

    def __getitem__(self, index):
        dp = self.ds[index]
        for transform in self.transforms:
            assert callable(transform)
            if isinstance(dp, (list, tuple)):
                dp = transform(*dp)
            else:
                dp = transform(dp)
        return dp


class AugmentedDataset(IndexableDatasetWrapper):
    def __init__(self, ds, augmentations):
        super(AugmentedDataset, self).__init__(ds)
        self.augmentations = augmentations
        self.num_augmentations = len(self.augmentations)

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError
        dp = self.ds[index % self.ds_len]
        if index < self.ds_len:
            return dp
        augmentation = self.augmentations[(index // self.ds_len) - 1]
        assert callable(augmentation)
        if isinstance(dp, (list, tuple)):
            return augmentation(*dp)
        else:
            return augmentation(dp)

    def __len__(self):
        # every augmentation gives one more duplication of dataset
        return self.ds_len * (1 + self.num_augmentations)


class Dataloader(DatasetWrapper):
    def __init__(self,
                 ds,
                 augmentations=None,
                 shuffle=False,
                 batch_size=1,
                 drop_remainder=True,
                 output_types=None,
                 num_worker=os.cpu_count(),
                 use_zmq=True,
                 prefetch_batch=True,
                 num_prefetch=None,
                 transforms=None):

        super(Dataloader, self).__init__(ds)
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.output_types = output_types
        self.num_worker = num_worker
        self.use_zmq = use_zmq
        self.prefetch_batch = prefetch_batch
        self.num_prefetch = num_worker if num_prefetch is None else num_prefetch
        self.transforms = transforms

        if self.augmentations is not None:
            self.ds = AugmentedDataset(self.ds, self.augmentations)

        if self.transforms is not None:
            self.ds = TransformedDataset(self.ds, self.transforms)
            # self.tfds = self.tfds.map(map_func=_Transforms(self.transforms), num_parallel_calls=num_map_worker)

        # TODO: auto adjust num_prefetch
        if self.num_worker > 1:
            if self.use_zmq:
                self.ds = ZMQMultiprocessDataset(self.ds, num_worker=self.num_worker, hwm=self.num_prefetch,
                                                 shuffle=self.shuffle)
            else:
                self.ds = MultiprocessDataset(self.ds, num_worker=self.num_worker, num_prefetch=self.num_prefetch,
                                              shuffle=self.shuffle)
        elif self.shuffle:
            self.ds = ShuffledDataset(self.ds)

        if self.prefetch_batch:
            self.ds = PrefetchBatchedDataset(self.ds, self.batch_size, drop_remainder=self.drop_remainder,
                                             output_types=self.output_types, use_zmq=self.use_zmq)
        else:
            self.ds = BatchedDataset(self.ds, self.batch_size, drop_remainder=self.drop_remainder,
                                     output_types=self.output_types)

    def __iter__(self):
        for dp in self.ds:
            yield dp


class TFDataloader(DatasetWrapper):
    def __init__(self,
                 ds,
                 output_types,
                 augmentations=None,
                 shuffle=False,
                 shuffle_buffer_size=None,
                 batch_size=1,
                 drop_remainder=True,
                 num_worker=tf.data.experimental.AUTOTUNE,
                 transforms=None):

        super(TFDataloader, self).__init__(ds)
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_buffer_size = 2 * batch_size if shuffle_buffer_size is None else shuffle_buffer_size
        self.drop_remainder = drop_remainder
        self.transforms = transforms

        self.ds = tf.data.Dataset.from_generator(self.ds, output_types=output_types)

        if self.shuffle:
            self.ds = self.ds.shuffle(buffer_size=self.shuffle_buffer_size)

        if self.transforms is not None:
            self.ds = self.ds.map(map_func=_Transforms_for_tf_dataset(self.transforms),
                                  num_parallel_calls=num_worker)

        if self.batch_size > 1:
            self.ds = self.ds.batch(batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        self.ds = self.ds.prefetch(tf.data.experimental.AUTOTUNE)

    def __iter__(self):
        for dp in self.ds:
            yield dp
