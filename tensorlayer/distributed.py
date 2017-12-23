#! /usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.training import session_run_hook
import os
import sys
import json
import time

# Disable buffer for stdout.
# When running in container, or other environemnts where stdout is redirected,
# the default buffer behavior will seriously delay the message written by `print`.
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

class TaskSpecDef(object):
    """Specification for the distributed task with the job name, index of the task,
    the parameter servers and the worker servers. If you want to use the last worker
    for continuous evaluation you can call the method `user_last_worker_as_evaluator`
    which returns a new :class:`TaskSpecDef` object without the last worker in the
    cluster specification.

    Parameters
    ----------
    type : A string with the job name, it will be `master`, `worker` or `ps`.
    index : The zero-based index of the task. Distributed training jobs will have a single
        master task, one or more parameter servers, and one or more workers.
    trial : The identifier of the trial being run.
    ps_hosts : A string with a coma separate list of hosts for the parameter servers
        or a list of hosts.
    worker_hosts : A string with a coma separate list of hosts for the worker servers
        or a list of hosts.
    master : A string with the master hosts

    Note
    ----------
    master might not be included in TF_CONFIG and can be None. The shard_index is adjusted
    in any case to assign 0 to master and >= 1 to workers.
    This implementation doesn't support sparse arrays in the `TF_CONFIG` variable as the
    official TensorFlow documentation shows, as it is not a supported by the json
    definition.

    References
    ----------
    - `ML-engine trainer considerations <https://cloud.google.com/ml-engine/docs/trainer-considerations#use_tf_config>`_
    """

    def __init__(self, type='master', index=0, trial=None, ps_hosts=None, worker_hosts=None,
                 master=None):
        self.type = type
        self._index = int(index)
        self._cluster_spec = None
        self.num_workers = 1
        self.num_ps = 0
        self.shard_index = int(index)
        self._master = True
        self.trial = trial
        self.ps_hosts = ps_hosts
        self.worker_hosts = worker_hosts
        self.master = master

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
                if self.type == 'worker':
                    self.shard_index = self._index + 1
                self._master = self.type == 'master'
            else:
                self._cluster_spec = tf.train.ClusterSpec({'ps'    : ps,
                                                           'worker': worker})
                if self.type == 'worker':
                    self.shard_index = self._index
                self._master = self.type == 'worker' and self._index == 0

            # create server and join if it is a parameter server
            self._server = tf.train.Server(self._cluster_spec,
                                           job_name=self.type,
                                           task_index=self._index)
            if self.is_ps():
                self._server.join()
        else:
            self._server = None

    def is_ps(self):
        """Returns true if this server is a parameter server"""
        return self.type == 'ps'

    def is_worker(self):
        """Returns true if this server is a worker server"""
        return self.type == 'worker'

    def is_master(self):
        """Returns true if this server is the master server"""
        return self._master

    def is_evaluator(self):
        """Returns true if this server is the evaluator server"""
        return self.type == 'worker' and len(self.worker_hosts) == self._index

    def device_fn(self):
        """Returns the function with the specification to create the graph in this server"""
        current_device = '/job:{}/task:{}'.format(self.type, self._index)
        ps_devices = '/job:ps'
        return tf.train.replica_device_setter(ps_device=ps_devices,
                                              worker_device=current_device,
                                              cluster=self._cluster_spec)

    def target(self):
        if self._server is not None:
            return self._server.target
        else:
            return None

    def user_last_worker_as_evaluator(self):
        """ Returns a new :class:`TaskSpecDef` where the last worker has been removed from
         the list of worker_hosts, so it is not used for training anymore. You can call
         is_evaluator to know whether this server is the evaluator one or not.
         In case there is only one server for training this method raises an exception, as
         you cannot use any server for evaluation.
         """
        if self.worker_hosts is None \
                or len(self.worker_hosts) == 0 \
                or (self.master is None and len(self.worker_hosts) == 1):
            raise Exception('You need more than one worker instance to use one as evaluator')
        return TaskSpecDef(type=self.type,
                           index=self._index,
                           trial=self.trial,
                           ps_hosts=self.ps_hosts,
                           worker_hosts=self.worker_hosts[:-1],
                           master=self.master)


def TaskSpec():
    """Returns the a :class:`TaskSpecDef` based on the environment variables for distributed
    training.

    References
    ----------
    - `ML-engine trainer considerations <https://cloud.google.com/ml-engine/docs/trainer-considerations#use_tf_config>`_
    - `TensorPort Distributed Computing <https://www.tensorport.com/documentation/code-details/>`_
    """

    # TF_CONFIG is used in ML-engine
    if 'TF_CONFIG' in os.environ:
        env = json.loads(os.environ.get('TF_CONFIG', '{}'))
        task_data = env.get('task', None) or {'type': 'master', 'index': 0}
        cluster_data = env.get('cluster', None) or {'ps': None, 'worker': None, 'master': None}
        return TaskSpecDef(type=task_data['type'],
                           index=task_data['index'],
                           trial=task_data['trial'] if 'trial' in task_data else None,
                           ps_hosts=cluster_data['ps'],
                           worker_hosts=cluster_data['worker'],
                           master=cluster_data['master'] if 'master' in cluster_data else None)

    # JOB_NAME, TASK_INDEX, PS_HOSTS, WORKER_HOSTS and MASTER_HOST are used in TensorPort
    if 'JOB_NAME' in os.environ:
        return TaskSpecDef(type=os.environ['JOB_NAME'],
                        index=os.environ['TASK_INDEX'],
                        ps_hosts=os.environ.get('PS_HOSTS', None),
                        worker_hosts=os.environ.get('WORKER_HOSTS', None),
                        master=os.environ.get('MASTER_HOST', None))
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
    """Creates a distributed session. It calls `MonitoredTrainingSession` to create a
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
    task_spec : TaskSpecDef. The task spec definition from TaskSpec()
    checkpoint_dir : A string.  Optional path to a directory where to restore
      variables.
    scaffold : A `Scaffold` used for gathering or building supportive ops. If
      not specified, a default one is created. It's used to finalize the graph.
    hooks : Optional list of `SessionRunHook` objects.
    chief_only_hooks : list of `SessionRunHook` objects. Activate these hooks if
      `is_chief==True`, ignore otherwise.
    save_checkpoint_secs : The frequency, in seconds, that a checkpoint is saved
      using a default checkpoint saver. If `save_checkpoint_secs` is set to
      `None`, then the default checkpoint saver isn't used.
    save_summaries_steps : The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If both
      `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
      the default summary saver isn't used. Default 100.
    save_summaries_secs : The frequency, in secs, that the summaries are written
      to disk using a default summary saver.  If both `save_summaries_steps` and
      `save_summaries_secs` are set to `None`, then the default summary saver
      isn't used. Default not enabled.
    config : an instance of `tf.ConfigProto` proto used to configure the session.
      It's the `config` argument of constructor of `tf.Session`.
    stop_grace_period_secs : Number of seconds given to threads to stop after
      `close()` has been called.
    log_step_count_steps : The frequency, in number of global steps, that the
      global step/sec is logged.

    References
    ----------
    - `MonitoredTrainingSession <https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession>`_
    """
    target = task_spec.target() if task_spec is not None else None
    is_chief = task_spec.is_master() if task_spec is not None else True
    return tf.train.MonitoredTrainingSession(master=target,
                                             is_chief=is_chief,
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



class StopAtTimeHook(session_run_hook.SessionRunHook):
    """Hook that requests stop after a specified time.

    Parameters
    ----------
    time_running: Maximum time running in seconds
    """

    def __init__(self, time_running):
        self._time_running = time_running

    def begin(self):
        self._end_time = time.time() + self._time_running

    def after_run(self, run_context, run_values):
        if time.time() > self._end_time:
            run_context.request_stop()


class LoadCheckpoint(session_run_hook.SessionRunHook):
    """Hook that loads a checkpoint after the session is created.

    >>> from tensorflow.python.ops import variables as tf_variables
    >>> from tensorflow.python.training.monitored_session import SingularMonitoredSession
    >>>
    >>> tensors = create_graph()
    >>> saver = tf.train.Saver(var_list=tf_variables.trainable_variables())
    >>> checkpoint_hook = LoadCheckpoint(saver, my_checkpoint_file)
    >>> with tf.SingularMonitoredSession(hooks=[checkpoint_hook]) as session:
    >>>      while not session.should_stop():
    >>>           session.run(tensors)
    """

    def __init__(self, saver, checkpoint):
        self._saver = saver
        self._checkpoint = checkpoint
        self._loaded = False

    def after_create_session(self, session, coord):
        if not self._loaded:
            self._loaded = True
            self._saver.restore(self._checkpoint)
