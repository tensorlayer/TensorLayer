#! /usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import time

import tensorflow as tf
from tensorflow.python.training import session_run_hook

from tensorlayer import tl_logging as logging
from tensorlayer.decorators import deprecated
from tensorlayer.lazy_imports import LazyImport

hvd = LazyImport('horovod.tensorflow')

__all__ = ['TaskSpecDef', 'TaskSpec', 'DistributedSession', 'StopAtTimeHook', 'LoadCheckpoint', 'Trainer']


class Trainer(object):

    def __init__(
            self, training_network_and_cost_func, training_dataset, optimizer=tf.train.AdamOptimizer,
            optimizer_args=None, batch_size=32, num_epochs=100, checkpoint_dir='./checkpoints',
            num_steps=20000, log_step_size=20,
            validation_network_cost_func=None, validation_dataset=None
    ):
        # Initialize Horovod.
        hvd.init()
        self.is_master = hvd.rank() == 0

        # Define the loss for validation dataset
        if (validation_network_cost_func is None) or (validation_dataset is None):
            self._validation_iterator = None
            self._validation_loss = None
        else:
            shard = validation_dataset.shard(num_shards=hvd.size(), index=hvd.rank()).batch(batch_size)
            self._validation_iterator = shard.make_initializable_iterator()
            next_example, next_label = self._validation_iterator.get_next()
            _, self._validation_loss = validation_network_cost_func(next_example, next_label)

        # Get the shard of the dataset based on my local rank
        training_dataset = training_dataset.shuffle(buffer_size=10000, seed=0)
        shard = training_dataset.shard(num_shards=hvd.size(), index=hvd.rank()).batch(batch_size).repeat(num_epochs)
        training_iterator = shard.make_one_shot_iterator()
        next_example, next_label = training_iterator.get_next()
        self.training_network, loss = training_network_and_cost_func(next_example, next_label)

        if not optimizer_args:
            optimizer_args = dict(learning_rate=0.001)
        # Adjust learning rate based on number of GPUs.
        optimizer_args['learning_rate'] = optimizer_args['learning_rate'] * hvd.size()
        opt = optimizer(**optimizer_args)

        # Add Horovod Distributed Optimizer.
        opt = hvd.DistributedOptimizer(opt)

        global_step = tf.train.get_or_create_global_step()
        self._train_op = opt.minimize(loss, global_step=global_step)  # TODO: support a list of losses

        hooks = [
            # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
            # from rank 0 to all other processes. This is necessary to ensure consistent
            # initialization of all workers when training is started with random weights
            # or restored from a checkpoint.
            hvd.BroadcastGlobalVariablesHook(0),

            # Horovod: adjust number of steps based on number of GPUs.
            tf.train.StopAtStepHook(last_step=num_steps // hvd.size()),
            tf.train.LoggingTensorHook(tensors={
                'training loss': loss
            }, every_n_iter=log_step_size),
        ]

        # Pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())

        # Save checkpoints only on worker 0 to prevent other workers from
        # corrupting them.
        checkpoint_dir = checkpoint_dir if self.is_master else None

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        self.sess = tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, hooks=hooks, config=config)

    def train_on_batch(self):
        self.sess.run(self._train_op)

    def train_on_all(self):
        while not self.sess.should_stop():
            # Run a training step synchronously.
            self.train_on_batch()

    def get_validation_loss(self):
        if (self._validation_iterator is None) or (self._validation_loss is None):
            raise AttributeError('Validation is not setup.')

        n = 0.0
        loss_sum = 0.0
        self.sess.run(self._validation_iterator.initializer)
        while True:
            try:
                loss = self.sess.run(self._validation_loss)
                tf.train.get_global_step()
                loss_sum += loss
                n += 1.0
            except tf.errors.OutOfRangeError:
                break
        average_loss = loss_sum / n
        return average_loss

    def train_and_validate_on_all(self, validate_step_size=50):
        step = 0
        while not self.sess.should_stop():
            self.train_on_batch() # Run a training step synchronously.
            if step % validate_step_size == 0:
                logging.info("Average loss for validation dataset: %s" % self.get_validation_loss())
            step += 1


@deprecated(date="2018-10-30", instructions="Using the TensorLayer distributed trainer.")
class TaskSpecDef(object):
    """Specification for a distributed task.

    It contains the job name, index of the task,
    the parameter servers and the worker servers. If you want to use the last worker
    for continuous evaluation you can call the method `use_last_worker_as_evaluator`
    which returns a new :class:`TaskSpecDef` object without the last worker in the
    cluster specification.

    Parameters
    ----------
    task_type : str
        Task type. One of `master`, `worker` or `ps`.
    index : int
        The zero-based index of the task. Distributed training jobs will have a single
        master task, one or more parameter servers, and one or more workers.
    trial : int
        The identifier of the trial being run.
    ps_hosts : str OR list of str
        A string with a coma separate list of hosts for the parameter servers
        or a list of hosts.
    worker_hosts : str OR list of str
        A string with a coma separate list of hosts for the worker servers
        or a list of hosts.
    master : str
        A string with the master hosts

    Notes
    ----------
    master might not be included in TF_CONFIG and can be None. The shard_index is adjusted
    in any case to assign 0 to master and >= 1 to workers.
    This implementation doesn't support sparse arrays in the `TF_CONFIG` variable as the
    official TensorFlow documentation shows, as it is not a supported by the json
    definition.

    References
    ----------
    - `ML-engine trainer considerations <https://cloud.google.com/ml-engine/docs/trainer-considerations#use_tf_config>`__

    """

    def __init__(self, task_type='master', index=0, trial=None, ps_hosts=None, worker_hosts=None, master=None):
        self.type = task_type
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
        self._server = None

        if ps_hosts and worker_hosts:
            self.ps_hosts = ps_hosts if isinstance(ps_hosts, list) else ps_hosts.split(',')
            self.num_ps = len(self.ps_hosts)
            self.worker_hosts = worker_hosts if isinstance(worker_hosts, list) else worker_hosts.split(',')
            if master is not None and len(master) > 0:
                self._cluster_spec = tf.train.ClusterSpec(
                    {
                        'ps': self.ps_hosts,
                        'worker': self.worker_hosts,
                        'master': master
                    }
                )
                # master is a worker too
                self.num_workers = len(self.worker_hosts) + 1
                if self.type == 'worker':
                    self.shard_index = self._index + 1
                self._master = self.type == 'master'
            else:
                self._cluster_spec = tf.train.ClusterSpec({'ps': self.ps_hosts, 'worker': self.worker_hosts})
                self.num_workers = len(self.worker_hosts)
                if self.type == 'worker':
                    self.shard_index = self._index
                self._master = self.type == 'worker' and self._index == 0

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
        return self.type == 'worker' and self.num_workers == self._index

    def device_fn(self):
        """Returns the function with the specification to create the graph in this server"""
        current_device = '/job:{}/task:{}'.format(self.type, self._index)
        ps_devices = '/job:ps'
        return tf.train.replica_device_setter(
            ps_device=ps_devices, worker_device=current_device, cluster=self._cluster_spec
        )

    def create_server(self):
        if self._server is None and self.ps_hosts and self.worker_hosts and not self.is_evaluator():
            # create server and join if it is a parameter server
            self._server = tf.train.Server(self._cluster_spec, job_name=self.type, task_index=self._index)
            if self.is_ps():
                self._server.join()

    def target(self):
        if self._server is None:
            self.create_server()
        if self._server is not None:
            return self._server.target
        else:
            return None

    def use_last_worker_as_evaluator(self):
        """Returns a new :class:`TaskSpecDef` where the last worker has been removed from
        the list of worker_hosts, so it is not used for training anymore. You can call
        is_evaluator to know whether this server is the evaluator one or not.
        In case there is only one server for training this method raises an exception, as
        you cannot use any server for evaluation.

        """
        if self.num_workers <= 1:
            raise Exception('You need more than one worker instance to use one as evaluator')

        return TaskSpecDef(
            task_type=self.type, index=self._index, trial=self.trial, ps_hosts=self.ps_hosts,
            worker_hosts=self.worker_hosts[:-1], master=self.master
        )


@deprecated(date="2018-10-30", instructions="Using the TensorLayer distributed trainer.")
def create_task_spec_def():
    """Returns the a :class:`TaskSpecDef` based on the environment variables for distributed training.

    References
    ----------
    - `ML-engine trainer considerations <https://cloud.google.com/ml-engine/docs/trainer-considerations#use_tf_config>`__
    - `TensorPort Distributed Computing <https://www.tensorport.com/documentation/code-details/>`__

    """
    if 'TF_CONFIG' in os.environ:
        # TF_CONFIG is used in ML-engine
        env = json.loads(os.environ.get('TF_CONFIG', '{}'))
        task_data = env.get('task', None) or {'type': 'master', 'index': 0}
        cluster_data = env.get('cluster', None) or {'ps': None, 'worker': None, 'master': None}
        return TaskSpecDef(
            task_type=task_data['type'], index=task_data['index'], trial=task_data['trial'] if 'trial' in task_data else
            None, ps_hosts=cluster_data['ps'], worker_hosts=cluster_data['worker'], master=cluster_data['master']
            if 'master' in cluster_data else None
        )
    elif 'JOB_NAME' in os.environ:
        # JOB_NAME, TASK_INDEX, PS_HOSTS, WORKER_HOSTS and MASTER_HOST are used in TensorPort
        return TaskSpecDef(
            task_type=os.environ['JOB_NAME'], index=os.environ['TASK_INDEX'], ps_hosts=os.environ.get('PS_HOSTS', None),
            worker_hosts=os.environ.get('WORKER_HOSTS', None), master=os.environ.get('MASTER_HOST', None)
        )
    else:
        raise Exception('You need to setup TF_CONFIG or JOB_NAME to define the task.')


@deprecated(date="2018-10-30", instructions="Using the TensorLayer distributed trainer.")
def create_distributed_session(
        task_spec=None, checkpoint_dir=None, scaffold=None, hooks=None, chief_only_hooks=None, save_checkpoint_secs=600,
        save_summaries_steps=object(), save_summaries_secs=object(), config=None, stop_grace_period_secs=120,
        log_step_count_steps=100
):
    """Creates a distributed session.

    It calls `MonitoredTrainingSession` to create a :class:`MonitoredSession` for distributed training.

    Parameters
    ----------
    task_spec : :class:`TaskSpecDef`.
        The task spec definition from create_task_spec_def()
    checkpoint_dir : str.
        Optional path to a directory where to restore variables.
    scaffold : ``Scaffold``
        A `Scaffold` used for gathering or building supportive ops.
        If not specified, a default one is created. It's used to finalize the graph.
    hooks : list of ``SessionRunHook`` objects.
        Optional
    chief_only_hooks : list of ``SessionRunHook`` objects.
        Activate these hooks if `is_chief==True`, ignore otherwise.
    save_checkpoint_secs : int
        The frequency, in seconds, that a checkpoint is saved
        using a default checkpoint saver. If `save_checkpoint_secs` is set to
        `None`, then the default checkpoint saver isn't used.
    save_summaries_steps : int
        The frequency, in number of global steps, that the
        summaries are written to disk using a default summary saver. If both
        `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
        the default summary saver isn't used. Default 100.
    save_summaries_secs : int
        The frequency, in secs, that the summaries are written
        to disk using a default summary saver.  If both `save_summaries_steps` and
        `save_summaries_secs` are set to `None`, then the default summary saver
        isn't used. Default not enabled.
    config : ``tf.ConfigProto``
        an instance of `tf.ConfigProto` proto used to configure the session.
        It's the `config` argument of constructor of `tf.Session`.
    stop_grace_period_secs : int
        Number of seconds given to threads to stop after
        `close()` has been called.
    log_step_count_steps : int
        The frequency, in number of global steps, that the
        global step/sec is logged.

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

    References
    ----------
    - `MonitoredTrainingSession <https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession>`__

    """
    target = task_spec.target() if task_spec is not None else None
    is_chief = task_spec.is_master() if task_spec is not None else True
    return tf.train.MonitoredTrainingSession(
        master=target, is_chief=is_chief, checkpoint_dir=checkpoint_dir, scaffold=scaffold,
        save_checkpoint_secs=save_checkpoint_secs, save_summaries_steps=save_summaries_steps,
        save_summaries_secs=save_summaries_secs, log_step_count_steps=log_step_count_steps,
        stop_grace_period_secs=stop_grace_period_secs, config=config, hooks=hooks, chief_only_hooks=chief_only_hooks
    )


@deprecated(date="2018-10-30", instructions="Using the TensorLayer distributed trainer.")
class StopAtTimeHook(session_run_hook.SessionRunHook):
    """Hook that requests stop after a specified time.

    Parameters
    ----------
    time_running: int
        Maximum time running in seconds

    """

    def __init__(self, time_running):
        self._time_running = time_running
        self._end_time = 0

    def begin(self):
        self._end_time = time.time() + self._time_running

    def after_run(self, run_context, run_values):
        if time.time() > self._end_time:
            run_context.request_stop()


@deprecated(date="2018-10-30", instructions="Using the TensorLayer distributed trainer.")
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


# Alias
TaskSpec = create_task_spec_def
DistributedSession = create_distributed_session
