import multiprocessing
import os
import sys
import uuid

import zmq
import numpy as np

from .base import DatasetWrapper
from .serialize import *
from .utils import clean_up_socket_files


class MultiprocessDataset(DatasetWrapper):
    def __init__(self,
                 ds,
                 num_worker,
                 num_prefetch,
                 shuffle=False):

        super(MultiprocessDataset, self).__init__(ds)
        self.num_worker = num_worker
        self.num_prefetch = num_prefetch
        self.shuffle = shuffle

        self.index_queue = multiprocessing.Queue(self.num_worker)
        self.data_queue = multiprocessing.Queue(self.num_prefetch)
        self.put_idx_worker = None
        for _ in range(num_worker):
            worker = multiprocessing.Process(target=self._data_worker,
                                             args=(self.ds, self.index_queue, self.data_queue))
            worker.daemon = True
            worker.start()

    def _data_worker(self, ds, index_q, data_q):
        while True:
            idx = index_q.get()
            data_q.put((idx, ds[idx]))

    def _put_idx(self, index_q, idx):
        index_q.put(idx)

    def __iter__(self):
        # clear queues from previous epoch
        while not self.index_queue.empty():
            self.index_queue.get()
        while not self.data_queue.empty():
            self.data_queue.get()

        # shuffle at the start of every epoch
        if self.shuffle:
            self.idxs = np.random.permutation(self.ds_len)
        else:
            self.idxs = np.arange(self.ds_len)

        send_idx_cnt = 0
        for _ in range(self.num_worker * 2):
            self._put_idx(self.index_queue, self.idxs[send_idx_cnt])
            send_idx_cnt += 1

        data_buffer = {}
        for return_idx in self.idxs:
            if return_idx in data_buffer:
                yield data_buffer.pop(return_idx)
            else:
                while True:
                    idx, dp = self.data_queue.get()
                    # put new idx after collecting data
                    if send_idx_cnt < len(self.idxs):
                        self._put_idx(self.index_queue, self.idxs[send_idx_cnt])
                        send_idx_cnt += 1
                    if idx == return_idx:
                        yield dp
                        break
                    else:
                        data_buffer[idx] = dp


class ZMQMultiprocessDataset(DatasetWrapper):
    def __init__(self,
                 ds,
                 num_worker,
                 hwm=50,
                 shuffle=False):

        super(ZMQMultiprocessDataset, self).__init__(ds)
        self.num_worker = num_worker
        self.shuffle = shuffle
        self._hwm = hwm

        self.idx_pipename = _get_pipe_name('put_idx')
        self.data_pipename = _get_pipe_name('collect_data')

        for i in range(num_worker):
            # first worker bind the socket, others connect to the socket
            # however, zmq sockets using ipc do not care about the order of bind / connect
            if i == 0:
                worker = multiprocessing.Process(target=self._data_worker,
                                                 args=(True,))
            else:
                worker = multiprocessing.Process(target=self._data_worker,
                                                 args=())
            worker.daemon = True
            worker.start()

        clean_up_socket_files([self.idx_pipename, self.data_pipename])

    def _data_worker(self, bind=False):
        context = zmq.Context()
        worker_receive_index_socket = context.socket(zmq.PULL)
        worker_receive_index_socket.set_hwm(self._hwm)
        if bind:
            worker_receive_index_socket.bind(self.idx_pipename)
        else:
            worker_receive_index_socket.connect(self.idx_pipename)

        worker_send_data_socket = context.socket(zmq.PUSH)
        worker_send_data_socket.set_hwm(self._hwm)
        if bind:
            worker_send_data_socket.bind(self.data_pipename)
        else:
            worker_send_data_socket.connect(self.data_pipename)

        while True:
            recv_msg = worker_receive_index_socket.recv(copy=False)
            idx = load_from_bytes(recv_msg)
            send_msg = convert_to_bytes({'idx': idx, 'data': self.ds[idx]})
            worker_send_data_socket.send(send_msg, copy=False)

    def _put_idx(self, put_idx_socket, idx):
        send_msg = convert_to_bytes(idx)
        put_idx_socket.send(send_msg, copy=False)

    def __iter__(self):
        context = zmq.Context()
        collect_data_socket = context.socket(zmq.PULL)
        collect_data_socket.set_hwm(self._hwm)
        collect_data_socket.connect(self.data_pipename)

        put_idx_socket = context.socket(zmq.PUSH)
        put_idx_socket.set_hwm(self._hwm)
        put_idx_socket.connect(self.idx_pipename)

        # shutdown put_idx_worker and clear queues from previous epoch
        try:
            while True:
                collect_data_socket.recv(flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            pass

        # shuffle at the start of every epoch
        if self.shuffle:
            self.idxs = np.random.permutation(self.ds_len)
        else:
            self.idxs = np.arange(self.ds_len)

        send_idx_cnt = 0
        for _ in range(self.num_worker * 2):
            self._put_idx(put_idx_socket, self.idxs[send_idx_cnt])
            send_idx_cnt += 1

        data_buffer = {}
        for return_idx in self.idxs:
            if return_idx in data_buffer:
                yield data_buffer.pop(return_idx)
            else:
                while True:
                    recv_msg = collect_data_socket.recv(copy=False)
                    recv_msg = load_from_bytes(recv_msg)
                    idx, dp = recv_msg['idx'], recv_msg['data']
                    # put new idx after collecting data
                    if send_idx_cnt < len(self.idxs):
                        self._put_idx(put_idx_socket, self.idxs[send_idx_cnt])
                        send_idx_cnt += 1
                    if idx == return_idx:
                        yield dp
                        break
                    else:
                        data_buffer[idx] = dp


def _get_pipe_name(name):
    if sys.platform.startswith('linux'):
        # linux supports abstract sockets: http://api.zeromq.org/4-1:zmq-ipc
        pipename = "ipc://@{}-pipe-{}".format(name, str(uuid.uuid1())[:8])
    else:
        pipedir = '.'
        assert os.path.isdir(pipedir), pipedir
        filename = '{}/{}-pipe-{}'.format(pipedir.rstrip('/'), name, str(uuid.uuid1())[:6])
        assert not os.path.exists(filename), "Pipe {} exists! You may be unlucky.".format(filename)
        pipename = "ipc://{}".format(filename)
    # register in environment variable, used for cleaning up ipc socket files
    # os.environ[name] = pipename
    return pipename
