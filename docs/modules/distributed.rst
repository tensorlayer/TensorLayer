API - Distribution (alpha)
=============================

Helper sessions and methods to run a distributed training.
Check this `minst example <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist_distributed.py>`_.

.. automodule:: tensorlayer.distributed

.. autosummary::

   TaskSpecDef
   TaskSpec
   DistributedSession


Distributed training
----------------------

TaskSpecDef
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: TaskSpecDef

Create TaskSpecDef from environment variables
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: TaskSpec

Distributed session object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: DistributedSession

Data sharding
^^^^^^^^^^^^^^^^^^^^^^

In some cases we want to shard the data among all the training servers and
not use all the data in all servers. TensorFlow >=1.4 provides some helper classes
to work with data that support data sharding: `Datasets <https://www.tensorflow.org/programmers_guide/datasets>`_

It is important in sharding that the shuffle or any non deterministic operation
is done after creating the shards:

.. code-block:: python

  from tensorflow.contrib.data import TextLineDataset
  from tensorflow.contrib.data import Dataset

  task_spec = TaskSpec()
  task_spec.create_server()
  files_dataset = Dataset.list_files(files_pattern)
  dataset = TextLineDataset(files_dataset)
  dataset = dataset.map(your_python_map_function, num_threads=4)
  if task_spec is not None:
        dataset = dataset.shard(task_spec.num_workers, task_spec.shard_index)
  dataset = dataset.shuffle(buffer_size)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(num_epochs)
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()
  with tf.device(task_spec.device_fn()):
        tensors = create_graph(next_element)
  with tl.DistributedSession(task_spec=task_spec,
                             checkpoint_dir='/tmp/ckpt') as session:
        while not session.should_stop():
            session.run(tensors)


Logging
^^^^^^^^^^^^^^^^^^^^^^

We can use task_spec to log only in the master server:

.. code-block:: python

  while not session.should_stop():
        should_log = task_spec.is_master() and your_conditions
        if should_log:
            results = session.run(tensors_with_log_info)
            logging.info(...)
        else:
            results = session.run(tensors)

Continuous evaluation
^^^^^^^^^^^^^^^^^^^^^^

You can use one of the workers to run an evaluation for the saved checkpoints:

.. code-block:: python

  import tensorflow as tf
  from tensorflow.python.training import session_run_hook
  from tensorflow.python.training.monitored_session import SingularMonitoredSession

  class Evaluator(session_run_hook.SessionRunHook):
        def __init__(self, checkpoints_path, output_path):
            self.checkpoints_path = checkpoints_path
            self.summary_writer = tf.summary.FileWriter(output_path)
            self.lastest_checkpoint = ''

        def after_create_session(self, session, coord):
            checkpoint = tf.train.latest_checkpoint(self.checkpoints_path)
            # wait until a new check point is available
            while self.lastest_checkpoint == checkpoint:
                time.sleep(30)
                checkpoint = tf.train.latest_checkpoint(self.checkpoints_path)
            self.saver.restore(session, checkpoint)
            self.lastest_checkpoint = checkpoint

        def end(self, session):
            super(Evaluator, self).end(session)
            # save summaries
            step = int(self.lastest_checkpoint.split('-')[-1])
            self.summary_writer.add_summary(self.summary, step)

        def _create_graph():
            # your code to create the graph with the dataset

        def run_evaluation():
            with tf.Graph().as_default():
                summary_tensors = create_graph()
                self.saver = tf.train.Saver(var_list=tf_variables.trainable_variables())
                hooks = self.create_hooks()
                hooks.append(self)
                if self.max_time_secs and self.max_time_secs > 0:
                    hooks.append(StopAtTimeHook(self.max_time_secs))
                # this evaluation runs indefinitely, until the process is killed
                while True:
                    with SingularMonitoredSession(hooks=[self]) as session:
                        try:
                            while not sess.should_stop():
                                self.summary = session.run(summary_tensors)
                        except OutOfRangeError:
                            pass
                        # end of evaluation

  task_spec = TaskSpec().user_last_worker_as_evaluator()
  if task_spec.is_evaluator():
        Evaluator().run_evaluation()
  else:
        task_spec.create_server()
        # run normal training



Session hooks
----------------------

TensorFlow provides some `Session Hooks <https://www.tensorflow.org/api_guides/python/train#Training_Hooks>`_
to do some operations in the sessions. We added more to help with common operations.


Stop after maximum time
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: StopAtTimeHook

Initialize network with checkpoint
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: LoadCheckpoint
