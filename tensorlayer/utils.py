#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import subprocess
import sys
import time
from collections import Counter
from sys import exit as _exit
from sys import platform as _platform

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import tensorlayer as tl

__all__ = [
    'fit', 'test', 'predict', 'evaluation', 'dict_to_one', 'flatten_list', 'class_balancing_oversample',
    'get_random_int', 'list_string_to_dict', 'exit_tensorflow', 'open_tensorboard', 'clear_all_placeholder_variables',
    'set_gpu_fraction', 'train_epoch', 'run_epoch'
]


def fit(
        network, train_op, cost, X_train, y_train, acc=None, batch_size=100, n_epoch=100, print_freq=5, X_val=None,
        y_val=None, eval_train=True, tensorboard_dir=None, tensorboard_epoch_freq=5, tensorboard_weight_histograms=True,
        tensorboard_graph_vis=True
):
    """Training a given non time-series network by the given cost function, training data, batch_size, n_epoch etc.

    - MNIST example click `here <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_simple.py>`_.
    - In order to control the training details, the authors HIGHLY recommend ``tl.iterate`` see two MNIST examples `1 <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mlp_dropout1.py>`_, `2 <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mlp_dropout1.py>`_.

    Parameters
    ----------
    network : TensorLayer Model
        the network to be trained.
    train_op : TensorFlow optimizer
        The optimizer for training e.g. tf.optimizers.Adam().
    cost : TensorLayer or TensorFlow loss function
        Metric for loss function, e.g tl.cost.cross_entropy.
    X_train : numpy.array
        The input of training data
    y_train : numpy.array
        The target of training data
    acc : TensorFlow/numpy expression or None
        Metric for accuracy or others. If None, would not print the information.
    batch_size : int
        The batch size for training and evaluating.
    n_epoch : int
        The number of training epochs.
    print_freq : int
        Print the training information every ``print_freq`` epochs.
    X_val : numpy.array or None
        The input of validation data. If None, would not perform validation.
    y_val : numpy.array or None
        The target of validation data. If None, would not perform validation.
    eval_train : boolean
        Whether to evaluate the model during training.
        If X_val and y_val are not None, it reflects whether to evaluate the model on training data.
    tensorboard_dir : string
        path to log dir, if set, summary data will be stored to the tensorboard_dir/ directory for visualization with tensorboard. (default None)
    tensorboard_epoch_freq : int
        How many epochs between storing tensorboard checkpoint for visualization to log/ directory (default 5).
    tensorboard_weight_histograms : boolean
        If True updates tensorboard data in the logs/ directory for visualization
        of the weight histograms every tensorboard_epoch_freq epoch (default True).
    tensorboard_graph_vis : boolean
        If True stores the graph in the tensorboard summaries saved to log/ (default True).

    Examples
    --------
    See `tutorial_mnist_simple.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_simple.py>`_

    >>> tl.utils.fit(network, train_op=tf.optimizers.Adam(learning_rate=0.0001),
    ...              cost=tl.cost.cross_entropy, X_train=X_train, y_train=y_train, acc=acc,
    ...              batch_size=64, n_epoch=20, _val=X_val, y_val=y_val, eval_train=True)
    >>> tl.utils.fit(network, train_op, cost, X_train, y_train,
    ...            acc=acc, batch_size=500, n_epoch=200, print_freq=5,
    ...            X_val=X_val, y_val=y_val, eval_train=False, tensorboard=True)

    Notes
    --------
    'tensorboard_weight_histograms' and 'tensorboard_weight_histograms' are not supported now.

    """
    if X_train.shape[0] < batch_size:
        raise AssertionError("Number of training examples should be bigger than the batch size")

    if tensorboard_dir is not None:
        tl.logging.info("Setting up tensorboard ...")
        #Set up tensorboard summaries and saver
        tl.files.exists_or_mkdir(tensorboard_dir)

        #Only write summaries for more recent TensorFlow versions
        if hasattr(tf, 'summary') and hasattr(tf.summary, 'create_file_writer'):
            train_writer = tf.summary.create_file_writer(tensorboard_dir + '/train')
            val_writer = tf.summary.create_file_writer(tensorboard_dir + '/validation')
            if tensorboard_graph_vis:
                # FIXME : not sure how to add tl network graph
                pass
    else:
        train_writer = None
        val_writer = None

        tl.logging.info("Finished! use `tensorboard --logdir=%s/` to start tensorboard" % tensorboard_dir)

    tl.logging.info("Start training the network ...")
    start_time_begin = time.time()
    for epoch in range(n_epoch):
        start_time = time.time()
        loss_ep, _, __ = train_epoch(network, X_train, y_train, cost=cost, train_op=train_op, batch_size=batch_size)

        train_loss, train_acc = None, None
        val_loss, val_acc = None, None
        if tensorboard_dir is not None and hasattr(tf, 'summary'):
            if epoch + 1 == 1 or (epoch + 1) % tensorboard_epoch_freq == 0:
                if eval_train is True:
                    train_loss, train_acc, _ = run_epoch(
                        network, X_train, y_train, cost=cost, acc=acc, batch_size=batch_size
                    )
                    with train_writer.as_default():
                        tf.compat.v2.summary.scalar('loss', train_loss, step=epoch)
                        if acc is not None:
                            tf.summary.scalar('acc', train_acc, step=epoch)
                    # FIXME : there seems to be an internal error in Tensorboard (misuse of tf.name_scope)
                    # if tensorboard_weight_histograms is not None:
                    #     for param in network.all_weights:
                    #         tf.summary.histogram(param.name, param, step=epoch)

                if (X_val is not None) and (y_val is not None):
                    val_loss, val_acc, _ = run_epoch(network, X_val, y_val, cost=cost, acc=acc, batch_size=batch_size)
                    with val_writer.as_default():
                        tf.summary.scalar('loss', val_loss, step=epoch)
                        if acc is not None:
                            tf.summary.scalar('acc', val_acc, step=epoch)
                        # FIXME : there seems to be an internal error in Tensorboard (misuse of tf.name_scope)
                        # if tensorboard_weight_histograms is not None:
                        #     for param in network.all_weights:
                        #         tf.summary.histogram(param.name, param, step=epoch)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            if (X_val is not None) and (y_val is not None):
                tl.logging.info("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                if eval_train is True:
                    if train_loss is None:
                        train_loss, train_acc, _ = run_epoch(
                            network, X_train, y_train, cost=cost, acc=acc, batch_size=batch_size
                        )
                    tl.logging.info("   train loss: %f" % train_loss)
                    if acc is not None:
                        tl.logging.info("   train acc: %f" % train_acc)
                if val_loss is None:
                    val_loss, val_acc, _ = run_epoch(network, X_val, y_val, cost=cost, acc=acc, batch_size=batch_size)

                # tl.logging.info("   val loss: %f" % val_loss)

                if acc is not None:
                    pass
                    # tl.logging.info("   val acc: %f" % val_acc)
            else:
                tl.logging.info(
                    "Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep)
                )
    tl.logging.info("Total training time: %fs" % (time.time() - start_time_begin))


def test(network, acc, X_test, y_test, batch_size, cost=None):
    """
    Test a given non time-series network by the given test data and metric.

    Parameters
    ----------
    network : TensorLayer Model
        The network.
    acc : TensorFlow/numpy expression or None
        Metric for accuracy or others.
            - If None, would not print the information.
    X_test : numpy.array
        The input of testing data.
    y_test : numpy array
        The target of testing data
    batch_size : int or None
        The batch size for testing, when dataset is large, we should use minibatche for testing;
        if dataset is small, we can set it to None.
    cost : TensorLayer or TensorFlow loss function
        Metric for loss function, e.g tl.cost.cross_entropy. If None, would not print the information.

    Examples
    --------
    See `tutorial_mnist_simple.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_simple.py>`_

    >>> def acc(_logits, y_batch):
    ...     return np.mean(np.equal(np.argmax(_logits, 1), y_batch))
    >>> tl.utils.test(network, acc, X_test, y_test, batch_size=None, cost=tl.cost.cross_entropy)

    """
    tl.logging.info('Start testing the network ...')
    network.eval()
    if batch_size is None:
        y_pred = network(X_test)
        if cost is not None:
            test_loss = cost(y_pred, y_test)
            # tl.logging.info("   test loss: %f" % test_loss)
        test_acc = acc(y_pred, y_test)
        # tl.logging.info("   test acc: %f" % (test_acc / test_acc))
        return test_acc
    else:
        test_loss, test_acc, n_batch = run_epoch(
            network, X_test, y_test, cost=cost, acc=acc, batch_size=batch_size, shuffle=False
        )
        if cost is not None:
            tl.logging.info("   test loss: %f" % test_loss)
        tl.logging.info("   test acc: %f" % test_acc)
        return test_acc


def predict(network, X, batch_size=None):
    """
    Return the predict results of given non time-series network.

    Parameters
    ----------
    network : TensorLayer Model
        The network.
    X : numpy.array
        The inputs.
    batch_size : int or None
        The batch size for prediction, when dataset is large, we should use minibatche for prediction;
        if dataset is small, we can set it to None.

    Examples
    --------
    See `tutorial_mnist_simple.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_simple.py>`_

    >>> _logits = tl.utils.predict(network, X_test)
    >>> y_pred = np.argmax(_logits, 1)

    """
    network.eval()
    if batch_size is None:
        y_pred = network(X)
        return y_pred
    else:
        result = None
        for X_a, _ in tl.iterate.minibatches(X, X, batch_size, shuffle=False):
            result_a = network(X_a)
            if result is None:
                result = result_a
            else:
                result = np.concatenate((result, result_a))
        if result is None:
            if len(X) % batch_size == 0:
                result_a = network(X[-(len(X) % batch_size):, :])
                result = result_a
        else:
            if len(X) != len(result) and len(X) % batch_size != 0:
                result_a = network(X[-(len(X) % batch_size):, :])
                result = np.concatenate((result, result_a))
        return result


## Evaluation
def evaluation(y_test=None, y_predict=None, n_classes=None):
    """
    Input the predicted results, targets results and
    the number of class, return the confusion matrix, F1-score of each class,
    accuracy and macro F1-score.

    Parameters
    ----------
    y_test : list
        The target results
    y_predict : list
        The predicted results
    n_classes : int
        The number of classes

    Examples
    --------
    >>> c_mat, f1, acc, f1_macro = tl.utils.evaluation(y_test, y_predict, n_classes)

    """
    c_mat = confusion_matrix(y_test, y_predict, labels=[x for x in range(n_classes)])
    f1 = f1_score(y_test, y_predict, average=None, labels=[x for x in range(n_classes)])
    f1_macro = f1_score(y_test, y_predict, average='macro')
    acc = accuracy_score(y_test, y_predict)
    tl.logging.info('confusion matrix: \n%s' % c_mat)
    tl.logging.info('f1-score        : %s' % f1)
    tl.logging.info('f1-score(macro) : %f' % f1_macro)  # same output with > f1_score(y_true, y_pred, average='macro')
    tl.logging.info('accuracy-score  : %f' % acc)
    return c_mat, f1, acc, f1_macro


def dict_to_one(dp_dict):
    """Input a dictionary, return a dictionary that all items are set to one.

    Used for disable dropout, dropconnect layer and so on.

    Parameters
    ----------
    dp_dict : dictionary
        The dictionary contains key and number, e.g. keeping probabilities.

    """
    return {x: 1 for x in dp_dict}


def flatten_list(list_of_list):
    """Input a list of list, return a list that all items are in a list.

    Parameters
    ----------
    list_of_list : a list of list

    Examples
    --------
    >>> tl.utils.flatten_list([[1, 2, 3],[4, 5],[6]])
    [1, 2, 3, 4, 5, 6]

    """
    return sum(list_of_list, [])


def class_balancing_oversample(X_train=None, y_train=None, printable=True):
    """Input the features and labels, return the features and labels after oversampling.

    Parameters
    ----------
    X_train : numpy.array
        The inputs.
    y_train : numpy.array
        The targets.

    Examples
    --------
    One X

    >>> X_train, y_train = class_balancing_oversample(X_train, y_train, printable=True)

    Two X

    >>> X, y = tl.utils.class_balancing_oversample(X_train=np.hstack((X1, X2)), y_train=y, printable=False)
    >>> X1 = X[:, 0:5]
    >>> X2 = X[:, 5:]

    """
    # ======== Classes balancing
    if printable:
        tl.logging.info("Classes balancing for training examples...")

    c = Counter(y_train)

    if printable:
        tl.logging.info('the occurrence number of each stage: %s' % c.most_common())
        tl.logging.info('the least stage is Label %s have %s instances' % c.most_common()[-1])
        tl.logging.info('the most stage is  Label %s have %s instances' % c.most_common(1)[0])

    most_num = c.most_common(1)[0][1]

    if printable:
        tl.logging.info('most num is %d, all classes tend to be this num' % most_num)

    locations = {}
    number = {}

    for lab, num in c.most_common():  # find the index from y_train
        number[lab] = num
        locations[lab] = np.where(np.array(y_train) == lab)[0]
    if printable:
        tl.logging.info('convert list(np.array) to dict format')
    X = {}  # convert list to dict
    for lab, num in number.items():
        X[lab] = X_train[locations[lab]]

    # oversampling
    if printable:
        tl.logging.info('start oversampling')
    for key in X:
        temp = X[key]
        while True:
            if len(X[key]) >= most_num:
                break
            X[key] = np.vstack((X[key], temp))
    if printable:
        tl.logging.info('first features of label 0 > %d' % len(X[0][0]))
        tl.logging.info('the occurrence num of each stage after oversampling')
    for key in X:
        tl.logging.info("%s %d" % (key, len(X[key])))
    if printable:
        tl.logging.info('make each stage have same num of instances')
    for key in X:
        X[key] = X[key][0:most_num, :]
        tl.logging.info("%s %d" % (key, len(X[key])))

    # convert dict to list
    if printable:
        tl.logging.info('convert from dict to list format')
    y_train = []
    X_train = np.empty(shape=(0, len(X[0][0])))
    for key in X:
        X_train = np.vstack((X_train, X[key]))
        y_train.extend([key for i in range(len(X[key]))])
    # tl.logging.info(len(X_train), len(y_train))
    c = Counter(y_train)
    if printable:
        tl.logging.info('the occurrence number of each stage after oversampling: %s' % c.most_common())
    # ================ End of Classes balancing
    return X_train, y_train


## Random
def get_random_int(min_v=0, max_v=10, number=5, seed=None):
    """Return a list of random integer by the given range and quantity.

    Parameters
    -----------
    min_v : number
        The minimum value.
    max_v : number
        The maximum value.
    number : int
        Number of value.
    seed : int or None
        The seed for random.

    Examples
    ---------
    >>> r = get_random_int(min_v=0, max_v=10, number=5)
    [10, 2, 3, 3, 7]

    """
    rnd = random.Random()
    if seed:
        rnd = random.Random(seed)
    # return [random.randint(min,max) for p in range(0, number)]
    return [rnd.randint(min_v, max_v) for p in range(0, number)]


def list_string_to_dict(string):
    """Inputs ``['a', 'b', 'c']``, returns ``{'a': 0, 'b': 1, 'c': 2}``."""
    dictionary = {}
    for idx, c in enumerate(string):
        dictionary.update({c: idx})
    return dictionary


def exit_tensorflow(port=6006):
    """Close TensorBoard and Nvidia-process if available.

    Parameters
    ----------
    port : int
        TensorBoard port you want to close, `6006` as default.

    """
    text = "[TL] Close tensorboard and nvidia-process if available"
    text2 = "[TL] Close tensorboard and nvidia-process not yet supported by this function (tl.ops.exit_tf) on "

    if _platform == "linux" or _platform == "linux2":
        tl.logging.info('linux: %s' % text)
        os.system('nvidia-smi')
        os.system('fuser ' + str(port) + '/tcp -k')  # kill tensorboard 6006
        os.system("nvidia-smi | grep python |awk '{print $3}'|xargs kill")  # kill all nvidia-smi python process
        _exit()

    elif _platform == "darwin":
        tl.logging.info('OS X: %s' % text)
        subprocess.Popen(
            "lsof -i tcp:" + str(port) + "  | grep -v PID | awk '{print $2}' | xargs kill", shell=True
        )  # kill tensorboard
    elif _platform == "win32":
        raise NotImplementedError("this function is not supported on the Windows platform")

    else:
        tl.logging.info(text2 + _platform)


def open_tensorboard(log_dir='/tmp/tensorflow', port=6006):
    """Open Tensorboard.

    Parameters
    ----------
    log_dir : str
        Directory where your tensorboard logs are saved
    port : int
        TensorBoard port you want to open, 6006 is tensorboard default

    """
    text = "[TL] Open tensorboard, go to localhost:" + str(port) + " to access"
    text2 = " not yet supported by this function (tl.ops.open_tb)"

    if not tl.files.exists_or_mkdir(log_dir, verbose=False):
        tl.logging.info("[TL] Log reportory was created at %s" % log_dir)

    if _platform == "linux" or _platform == "linux2":
        tl.logging.info('linux: %s' % text)
        subprocess.Popen(
            sys.prefix + " | python -m tensorflow.tensorboard --logdir=" + log_dir + " --port=" + str(port), shell=True
        )  # open tensorboard in localhost:6006/ or whatever port you chose
    elif _platform == "darwin":
        tl.logging.info('OS X: %s' % text)
        subprocess.Popen(
            sys.prefix + " | python -m tensorflow.tensorboard --logdir=" + log_dir + " --port=" + str(port), shell=True
        )  # open tensorboard in localhost:6006/ or whatever port you chose
    elif _platform == "win32":
        raise NotImplementedError("this function is not supported on the Windows platform")
    else:
        tl.logging.info(_platform + text2)


def clear_all_placeholder_variables(printable=True):
    """Clears all the placeholder variables of keep prob,
    including keeping probabilities of all dropout, denoising, dropconnect etc.

    Parameters
    ----------
    printable : boolean
        If True, print all deleted variables.

    """
    tl.logging.info('clear all .....................................')
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
        if 'class' in str(globals()[var]): continue

        if printable:
            tl.logging.info(" clear_all ------- %s" % str(globals()[var]))

        del globals()[var]


def set_gpu_fraction(gpu_fraction=0.3):
    """Set the GPU memory fraction for the application.

    Parameters
    ----------
    gpu_fraction : None or float
        Fraction of GPU memory, (0 ~ 1]. If None, allow gpu memory growth.

    References
    ----------
    - `TensorFlow using GPU <https://www.tensorflow.org/alpha/guide/using_gpu#allowing_gpu_memory_growth>`__

    """
    if gpu_fraction is None:
        tl.logging.info("[TL]: ALLOW GPU MEM GROWTH")
        tf.config.gpu.set_per_process_memory_growth(True)
    else:
        tl.logging.info("[TL]: GPU MEM Fraction %f" % gpu_fraction)
        tf.config.gpu.set_per_process_memory_fraction(0.4)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # return sess


def train_epoch(
        network, X, y, cost, train_op=tf.optimizers.Adam(learning_rate=0.0001), acc=None, batch_size=100, shuffle=True
):
    """Training a given non time-series network by the given cost function, training data, batch_size etc.
    for one epoch.

    Parameters
    ----------
    network : TensorLayer Model
        the network to be trained.
    X : numpy.array
        The input of training data
    y : numpy.array
        The target of training data
    cost : TensorLayer or TensorFlow loss function
        Metric for loss function, e.g tl.cost.cross_entropy.
    train_op : TensorFlow optimizer
        The optimizer for training e.g. tf.optimizers.Adam().
    acc : TensorFlow/numpy expression or None
        Metric for accuracy or others. If None, would not print the information.
    batch_size : int
        The batch size for training and evaluating.
    shuffle : boolean
        Indicating whether to shuffle the dataset in training.

    Returns
    -------
    loss_ep : Tensor. Average loss of this epoch.
    acc_ep : Tensor or None. Average accuracy(metric) of this epoch. None if acc is not given.
    n_step : int. Number of iterations taken in this epoch.

    """
    network.train()
    loss_ep = 0
    acc_ep = 0
    n_step = 0
    for X_batch, y_batch in tl.iterate.minibatches(X, y, batch_size, shuffle=shuffle):
        _loss, _acc = _train_step(network, X_batch, y_batch, cost=cost, train_op=train_op, acc=acc)

        loss_ep += _loss
        if acc is not None:
            acc_ep += _acc
        n_step += 1

    loss_ep = loss_ep / n_step
    acc_ep = acc_ep / n_step if acc is not None else None

    return loss_ep, acc_ep, n_step


def run_epoch(network, X, y, cost=None, acc=None, batch_size=100, shuffle=False):
    """Run a given non time-series network by the given cost function, test data, batch_size etc.
    for one epoch.

    Parameters
    ----------
    network : TensorLayer Model
        the network to be trained.
    X : numpy.array
        The input of training data
    y : numpy.array
        The target of training data
    cost : TensorLayer or TensorFlow loss function
        Metric for loss function, e.g tl.cost.cross_entropy.
    acc : TensorFlow/numpy expression or None
        Metric for accuracy or others. If None, would not print the information.
    batch_size : int
        The batch size for training and evaluating.
    shuffle : boolean
        Indicating whether to shuffle the dataset in training.

    Returns
    -------
    loss_ep : Tensor. Average loss of this epoch. None if 'cost' is not given.
    acc_ep : Tensor. Average accuracy(metric) of this epoch. None if 'acc' is not given.
    n_step : int. Number of iterations taken in this epoch.
    """
    network.eval()
    loss_ep = 0
    acc_ep = 0
    n_step = 0
    for X_batch, y_batch in tl.iterate.minibatches(X, y, batch_size, shuffle=shuffle):
        _loss, _acc = _run_step(network, X_batch, y_batch, cost=cost, acc=acc)
        if cost is not None:
            loss_ep += _loss
        if acc is not None:
            acc_ep += _acc
        n_step += 1

    loss_ep = loss_ep / n_step if cost is not None else None
    acc_ep = acc_ep / n_step if acc is not None else None

    return loss_ep, acc_ep, n_step


@tf.function
def _train_step(network, X_batch, y_batch, cost, train_op=tf.optimizers.Adam(learning_rate=0.0001), acc=None):
    """Train for one step"""
    with tf.GradientTape() as tape:
        y_pred = network(X_batch)
        _loss = cost(y_pred, y_batch)

    grad = tape.gradient(_loss, network.trainable_weights)
    train_op.apply_gradients(zip(grad, network.trainable_weights))

    if acc is not None:
        _acc = acc(y_pred, y_batch)
        return _loss, _acc
    else:
        return _loss, None


# @tf.function # FIXME : enable tf.function will cause some bugs in numpy, need fixing
def _run_step(network, X_batch, y_batch, cost=None, acc=None):
    """Run for one step"""
    y_pred = network(X_batch)
    _loss, _acc = None, None
    if cost is not None:
        _loss = cost(y_pred, y_batch)
    if acc is not None:
        _acc = acc(y_pred, y_batch)
    return _loss, _acc
