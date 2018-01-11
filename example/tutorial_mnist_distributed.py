#! /usr/bin/python
# -*- coding: utf-8 -*-

# Alpha Version for Distributed Training

# you can test this example in your local machine using 2 workers and 1 ps like below,
# where CUDA_VISIBLE_DEVICES can be used to set the GPUs the process can use.
#
# CUDA_VISIBLE_DEVICES= TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"type": "worker", "index": 0}}' python example/tutorial_mnist_distributed.py > output-master 2>&1 &
# CUDA_VISIBLE_DEVICES= TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"type": "worker", "index": 1}}' python example/tutorial_mnist_distributed.py > output-worker 2>&1 &
# CUDA_VISIBLE_DEVICES= TF_CONFIG='{"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"type": "ps", "index": 0}}' python example/tutorial_mnist_distributed.py > output-ps 2>&1 &
# Note: for GPU, please set CUDA_VISIBLE_DEVICES=GPU_ID

import tensorflow as tf
import tensorlayer as tl

# set buffer mode to _IOLBF for stdout
tl.ops.setlinebuf()

# load environment for distributed training
task_spec = tl.distributed.TaskSpec()
task_spec.create_server()
device_fn = task_spec.device_fn() if task_spec is not None else None

# prepare data
X_train, y_train, X_val, y_val, X_test, y_test = \
    tl.files.load_mnist_dataset(shape=(-1,784))

# create graph
with tf.device(device_fn):
    # define placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    # define the network
    network = tl.layers.InputLayer(x, name='input')
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu2')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    # the softmax is implemented internally in tl.cost.cross_entropy(y, y_) to
    # speed up computation, so we use identity here.
    # see tf.nn.sparse_softmax_cross_entropy_with_logits()
    network = tl.layers.DenseLayer(network, n_units=10,
                                    act=tf.identity, name='output')

    # define cost function and metric.
    y = network.outputs
    cost = tl.cost.cross_entropy(y, y_, name='cost')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_op = tf.argmax(tf.nn.softmax(y), 1)

    # define the optimizer
    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001
                        ).minimize(cost, var_list=train_params)

    with tl.distributed.DistributedSession(task_spec=task_spec) as sess:
        # print network information
        if task_spec.is_master():
            network.print_params(session=sess)
            network.print_layers()
            print_freq = 5
            eval_train = False
        else:
            print_freq = 1000
            eval_train = False

        # We do not need to initialize the variables as the session does it
        #tl.layers.initialize_global_variables(sess)

        # train the network
        tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
                    acc=acc, batch_size=500, n_epoch=500, print_freq=print_freq,
                    X_val=X_val, y_val=y_val, eval_train=eval_train)

        if task_spec.is_master():
            # evaluation
            tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

            # save the network to .npz file
            tl.files.save_npz(network.all_params , name='model.npz')
