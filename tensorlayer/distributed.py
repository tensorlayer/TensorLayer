#! /usr/bin/python
# -*- coding: utf8 -*-
import tensorflow as tf
import os
import json


class TaskSpecDef(object):
    """
    Specification for the distributed task with the job name, index of the task,
    the parameter servers and the worker servers
    """

    def __init__(self, job_name='master', index=0, ps_hosts=None, worker_hosts=None, master=None):
        self.job_name = job_name
        self._index = int(index)
        self._cluster_spec = None
        self.num_workers = 1
        self.num_ps = 0
        self.shard_index = int(index)
        self._master = True

        if ps_hosts and worker_hosts:
            ps = ps_hosts if isinstance(ps_hosts, list) else ps_hosts.split(',')
            self.num_ps = len(ps)
            worker = worker_hosts if isinstance(worker_hosts, list) else worker_hosts.split(',')
            if master is not None and len(master) > 0:
                self._cluster_spec = tf.train.ClusterSpec({'ps'    : ps,
                                                           'worker': worker,
                                                           'master': master})
                # master is a worker too
                self.num_workers = len(worker) + 1
                if self.job_name == 'worker':
                    self.shard_index = self._index + 1
                self._master = self.job_name == 'master'
            else:
                self._cluster_spec = tf.train.ClusterSpec({'ps'    : ps,
                                                           'worker': worker})
                if self.job_name == 'worker':
                    self.shard_index = self._index
                self._master = self.job_name == 'worker' and self._index == 0

            # create server and join if it is a parameter server
            self._server = tf.train.Server(self._cluster_spec,
                                           job_name=self.job_name,
                                           task_index=self._index)
            if self.is_ps():
                self._server.join()
        else:
            self._server = None

    def is_ps(self):
        return self.job_name == 'ps'

    def is_worker(self):
        return self.job_name == 'worker'

    def is_master(self):
        return self._master

    def device_fn(self):
        current_device = '/job:{}/task:{}'.format(self.job_name, self._index)
        ps_devices = '/job:ps'
        return tf.train.replica_device_setter(ps_device=ps_devices,
                                              worker_device=current_device,
                                              cluster=self._cluster_spec)

    def target(self):
        if self._server is not None:
            return self._server.target
        else:
            return None


def TaskSpec():
    if 'TF_CONFIG' in os.environ:
        env = json.loads(os.environ.get('TF_CONFIG', '{}'))
        task_data = env.get('task', None) or {'type': 'master', 'index': 0}
        cluster_data = env.get('cluster', None) or {'ps': None, 'worker': None, 'master': None}
        return TaskSpecDef(job_name=task_data['type'],
                           index=task_data['index'],
                           ps_hosts=cluster_data['ps'],
                           worker_hosts=cluster_data['worker'],
                           master=cluster_data['master'] if 'master' in cluster_data else None)
    return None


def DistributedSession(task_spec=None,
                       checkpoint_dir=None,
                       scaffold=None,
                       hooks=None,
                       chief_only_hooks=None,
                       save_checkpoint_secs=600,
                       save_summaries_steps=object(),
                       save_summaries_secs=object(),
                       config=None,
                       stop_grace_period_secs=120,
                       log_step_count_steps=100):
    """Creates a distributed session. It calls MonitoredTrainingSession to create a
    :class:`MonitoredSession` for distributed training.

    Examples
    --------

    A simple example for distributed training where all the workers use the same dataset:

    >>> task_spec = TaskSpec()
    >>> with tf.device(task_spec.device_fn()):
    >>>      tensors = create_graph()
    >>> with tl.DistributedSession(task_spec=task_spec,
    ...                            checkpoint_dir='/tmp/ckpt') as session:
    >>>      while not session.should_stop():
    >>>           session.run(tensors)

    An example where the dataset is shared among the workers
    (see https://www.tensorflow.org/programmers_guide/datasets):

    >>> task_spec = TaskSpec()
    >>> # dataset is a :class:`tf.data.Dataset` with the raw data
    >>> dataset = create_dataset()
    >>> if task_spec is not None:
    >>>     dataset = dataset.shard(task_spec.num_workers, task_spec.shard_index)
    >>> # shuffle or apply a map function to the new sharded dataset, for example:
    >>> dataset = dataset.shuffle(buffer_size=10000)
    >>> dataset = dataset.batch(batch_size)
    >>> dataset = dataset.repeat(num_epochs)
    >>> # create the iterator for the dataset and the input tensor
    >>> iterator = dataset.make_one_shot_iterator()
    >>> next_element = iterator.get_next()
    >>> with tf.device(task_spec.device_fn()):
    >>>      # next_element is the input for the graph
    >>>      tensors = create_graph(next_element)
    >>> with tl.DistributedSession(task_spec=task_spec,
    ...                            checkpoint_dir='/tmp/ckpt') as session:
    >>>      while not session.should_stop():
    >>>           session.run(tensors)


    Parameters
    ----------
    task_spec : TaskSpecDef
        the task spec definition from TaskSpec()
    checkpoint_dir: A string.  Optional path to a directory where to restore
      variables.
    scaffold: A `Scaffold` used for gathering or building supportive ops. If
      not specified, a default one is created. It's used to finalize the graph.
    hooks: Optional list of `SessionRunHook` objects.
    chief_only_hooks: list of `SessionRunHook` objects. Activate these hooks if
      `is_chief==True`, ignore otherwise.
    save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
      using a default checkpoint saver. If `save_checkpoint_secs` is set to
      `None`, then the default checkpoint saver isn't used.
    save_summaries_steps: The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If both
      `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
      the default summary saver isn't used. Default 100.
    save_summaries_secs: The frequency, in secs, that the summaries are written
      to disk using a default summary saver.  If both `save_summaries_steps` and
      `save_summaries_secs` are set to `None`, then the default summary saver
      isn't used. Default not enabled.
    config: an instance of `tf.ConfigProto` proto used to configure the session.
      It's the `config` argument of constructor of `tf.Session`.
    stop_grace_period_secs: Number of seconds given to threads to stop after
      `close()` has been called.
    log_step_count_steps: The frequency, in number of global steps, that the
      global step/sec is logged.

    References
    ----------
    - `MonitoredTrainingSession <https://www.tensorflow.org/api_docs/python/tf/train
    /MonitoredTrainingSession>`_
    """
    return tf.train.MonitoredTrainingSession(master=task_spec.target,
                                             is_chief=task_spec.is_master(),
                                             checkpoint_dir=checkpoint_dir,
                                             scaffold=scaffold,
                                             save_checkpoint_secs=save_checkpoint_secs,
                                             save_summaries_steps=save_summaries_steps,
                                             save_summaries_secs=save_summaries_secs,
                                             log_step_count_steps=log_step_count_steps,
                                             stop_grace_period_secs=stop_grace_period_secs,
                                             config=config,
                                             hooks=hooks,
                                             chief_only_hooks=chief_only_hooks)
