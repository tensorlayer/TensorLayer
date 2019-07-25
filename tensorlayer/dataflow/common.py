import atexit
import math
import multiprocessing
import os

import tensorflow as tf
import zmq

import numpy as np

from .base import IndexableDatasetWrapper, DatasetWrapper, _Transforms_for_tf_dataset
from .parallel import _get_pipe_name, ZMQMultiprocessDataset, MultiprocessDataset
from .utils import ensure_proc_terminate
from .serialize import convert_to_bytes, load_from_bytes

__all__ = ['BatchedDataset', 'TransformedDataset', 'ShuffledDataset',
           'AugmentedDataset', 'Dataloader', 'TFDataloader']


class BatchedDataset(DatasetWrapper):
    def __init__(self,
                 ds,
                 batch_size,
                 drop_remainder=True,
                 return_numpy=True,
                 keep_dims=False,
                 output_types=None,
                 use_zmq=True):
        super(BatchedDataset, self).__init__(ds)
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.return_numpy = return_numpy
        self.keep_dims = keep_dims
        self.output_types = output_types
        self.use_zmq = use_zmq

        # self.q = multiprocessing.Queue(maxsize=1)
        # self.worker = multiprocessing.Process(target=self._BatchedDataset_worker,
        #                                       args=(self.ds, self.q))
        # self.worker.start()
        # ensure_proc_terminate(self.worker)

        if self.use_zmq:
            self.data_pipename = _get_pipe_name('batch_prefetch')
            context = zmq.Context()
            self.fetch_data_socket = context.socket(zmq.PULL)
            self.fetch_data_socket.bind(self.data_pipename)
            self.worker = multiprocessing.Process(target=self._ZMQ_BatchedDataset_worker,
                                                  args=(self.ds,))
            self.worker.start()
        else:
            pipe_output, pipe_input = multiprocessing.Pipe()
            self.worker = multiprocessing.Process(target=self._BatchedDataset_worker,
                                                  args=(self.ds, (pipe_output, pipe_input)))
            self.worker.start()
            # main process only reads (gets output)
            pipe_input.close()
            self.pipe_output = pipe_output

        ensure_proc_terminate(self.worker)

    def _ZMQ_BatchedDataset_worker(self, ds):
        context = zmq.Context()
        prepare_data_socket = context.socket(zmq.PUSH)
        prepare_data_socket.connect(self.data_pipename)
        while True:
            dp_buffer = []
            for dp in ds:
                dp_buffer.append(dp)
                if len(dp_buffer) == self.batch_size:
                    # q.put(self._batch_datapoints(dp_buffer))
                    prepare_data_socket.send(convert_to_bytes(self._batch_datapoints(dp_buffer)), copy=False)
                    del dp_buffer[:]
            if not self.drop_remainder:
                # q.put(self._batch_datapoints(dp_buffer))
                prepare_data_socket.send(convert_to_bytes(self._batch_datapoints(dp_buffer)), copy=False)

    def _BatchedDataset_worker(self, ds, pipe):
        pipe_output, pipe_input = pipe
        # worker process only writes (puts input)
        pipe_output.close()
        while True:
            dp_buffer = []
            for dp in ds:
                dp_buffer.append(dp)
                if len(dp_buffer) == self.batch_size:
                    # q.put(self._batch_datapoints(dp_buffer))
                    pipe_input.send(self._batch_datapoints(dp_buffer))
                    del dp_buffer[:]
            if not self.drop_remainder:
                # q.put(self._batch_datapoints(dp_buffer))
                pipe_input.send(self._batch_datapoints(dp_buffer))

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

    def _batch_datapoints(self, dp_buffer):
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
                if self.return_numpy:
                    dp_batch[i] = self._batch_ndarray(dp_element_batch, dtype=self._get_element_dtype(i))
                else:
                    dp_batch[i] = dp_element_batch
            return dp_batch
        elif isinstance(first_dp, dict):
            dp_batch = {}
            for key in first_dp.keys():
                dp_element_batch = []
                for j in range(len(dp_buffer)):
                    dp_element_batch.append(dp_buffer[j][key])
                if self.return_numpy:
                    dp_batch[key] = self._batch_ndarray(dp_element_batch, dtype=None)
                else:
                    dp_batch[key] = dp_element_batch
            return dp_batch
        elif isinstance(first_dp, np.ndarray):
            return self._batch_ndarray(dp_buffer)
        # single elements
        else:
            if self.return_numpy:
                return self._batch_ndarray(dp_buffer, dtype=self._get_element_dtype(0))
            else:
                return dp_buffer

    def _batch_ndarray(self, dp_element_batch, dtype):
        """

        :param dp_element_batch: a list of datapoint element, an element can be np.ndarray / list
        :return: np.ndarray, type is the same as input
        """
        try:
            if dtype is not None:
                ret = np.asarray(dp_element_batch, dtype=dtype)
            else:
                ret = np.asarray(dp_element_batch)
            if self.keep_dims and len(ret.shape) == 1:
                ret = np.expand_dims(ret, 1)
            return ret
        except:
            raise ValueError("Unsupported type for batching.")

    def _get_element_dtype(self, i):
        if self.output_types is None:
            return None
        if not isinstance(self.output_types, (tuple, list)):
            return self.output_types
        if len(self.output_types) == 1:
            return self.output_types[0]
        return self.output_types[i]


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
                 batch_keep_dims=False,
                 output_types=None,
                 num_worker=os.cpu_count(),
                 use_zmq=True,
                 num_prefetch=None,
                 transforms=None):

        super(Dataloader, self).__init__(ds)
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.batch_keep_dims = batch_keep_dims
        self.output_types = output_types
        self.num_worker = num_worker
        self.use_zmq = use_zmq
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

        self.ds = BatchedDataset(self.ds, self.batch_size, drop_remainder=self.drop_remainder,
                                 output_types=self.output_types, keep_dims=self.batch_keep_dims,
                                 use_zmq=self.use_zmq)

        # self.tfds = tf.data.Dataset.from_generator(self.ds, output_types=output_types)

        # if self.num_prefetch > 1:
        #     self.tfds = self.tfds.prefetch(num_prefetch)
        atexit.register(self._clean_up_socket_files)

    def __iter__(self):
        for dp in self.ds:
            yield dp

    def _clean_up_socket_files(self):
        # remove all ipc socket files
        # the environment variable starts with 'ipc://', so file name starts from 6
        try:
            os.remove(os.environ['put_idx'][6:])
        except FileNotFoundError:
            pass
        try:
            os.remove(os.environ['collect_data'][6:])
        except FileNotFoundError:
            pass
        try:
            os.remove(os.environ['batch_prefetch'][6:])
        except FileNotFoundError:
            pass


class TFDataloader(DatasetWrapper):
    def __init__(self,
                 ds,
                 output_types,
                 augmentations=None,
                 shuffle=False,
                 shuffle_buffer_size=None,
                 batch_size=1,
                 drop_remainder=True,
                 # num_extract_worker=os.cpu_count(),
                 # num_map_worker=os.cpu_count(),
                 # num_prefetch=None,
                 transforms=None):

        super(TFDataloader, self).__init__(ds)
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_buffer_size = 2 * batch_size if shuffle_buffer_size is None else shuffle_buffer_size
        self.drop_remainder = drop_remainder
        # self.num_map_worker = num_map_worker
        # self.num_extract_worker = num_extract_worker
        # self.num_prefetch = num_extract_worker if num_prefetch is None else num_prefetch
        self.transforms = transforms

        self.ds = tf.data.Dataset.from_generator(self.ds, output_types=output_types)

        # if self.augmentations is not None:
        #     self.ds = AugmentedDataset(self.ds, self.augmentations)

        # if self.num_extract_worker > 1:
        #     self.ds = MultiProcessDataset(self.ds, num_worker=self.num_extract_worker, num_prefetch=self.num_prefetch)

        if self.shuffle:
            self.ds = self.ds.shuffle(buffer_size=self.shuffle_buffer_size)

        if self.transforms is not None:
            self.ds = self.ds.map(map_func=_Transforms_for_tf_dataset(self.transforms),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.batch_size > 1:
            self.ds = self.ds.batch(batch_size=self.batch_size, drop_remainder=self.drop_remainder)

        # if self.num_prefetch > 1:
        self.ds = self.ds.prefetch(tf.data.experimental.AUTOTUNE)

    def __iter__(self):
        for dp in self.ds:
            yield dp
