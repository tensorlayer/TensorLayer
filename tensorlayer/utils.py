#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import sys
from sys import exit as _exit
from sys import platform as _platform

import random
import subprocess
import time

from collections import Counter

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import tensorflow as tf
import tensorlayer as tl

__all__ = [
    'fit',
    'test',
    'predict',
    'evaluation',
    'dict_to_one',
    'flatten_list',
    'class_balancing_oversample',
    'get_random_int',
    'list_string_to_dict',
    'exit_tensorflow',
    'open_tensorboard',
    'clear_all_placeholder_variables',
    'set_gpu_fraction',
]


def fit(
        sess, network, train_op, cost, X_train, y_train, x, y_, acc=None, batch_size=100, n_epoch=100, print_freq=5,
        X_val=None, y_val=None, eval_train=True, tensorboard_dir=None, tensorboard_epoch_freq=5,
        tensorboard_weight_histograms=True, tensorboard_graph_vis=True
):
    """Training a given non time-series network by the given cost function, training data, batch_size, n_epoch etc.

    - MNIST example click `here <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_simple.py>`_.
    - In order to control the training details, the authors HIGHLY recommend ``tl.iterate`` see two MNIST examples `1 <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mlp_dropout1.py>`_, `2 <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mlp_dropout1.py>`_.

    Parameters
    ----------
    sess : Session
        TensorFlow Session.
    network : TensorLayer layer
        the network to be trained.
    train_op : TensorFlow optimizer
        The optimizer for training e.g. tf.train.AdamOptimizer.
    X_train : numpy.array
        The input of training data
    y_train : numpy.array
        The target of training data
    x : placeholder
        For inputs.
    y_ : placeholder
        For targets.
    acc : TensorFlow expression or None
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
        Also runs `tl.layers.initialize_global_variables(sess)` internally in fit() to setup the summary nodes.
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

    >>> tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
    ...            acc=acc, batch_size=500, n_epoch=200, print_freq=5,
    ...            X_val=X_val, y_val=y_val, eval_train=False)
    >>> tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
    ...            acc=acc, batch_size=500, n_epoch=200, print_freq=5,
    ...            X_val=X_val, y_val=y_val, eval_train=False,
    ...            tensorboard=True, tensorboard_weight_histograms=True, tensorboard_graph_vis=True)

    Notes
    --------
    If tensorboard_dir not None, the `global_variables_initializer` will be run inside the fit function
    in order to initialize the automatically generated summary nodes used for tensorboard visualization,
    thus `tf.global_variables_initializer().run()` before the `fit()` call will be undefined.

    """
    if X_train.shape[0] < batch_size:
        raise AssertionError("Number of training examples should be bigger than the batch size")

    if tensorboard_dir is not None:
        tl.logging.info("Setting up tensorboard ...")
        #Set up tensorboard summaries and saver
        tl.files.exists_or_mkdir(tensorboard_dir)

        #Only write summaries for more recent TensorFlow versions
        if hasattr(tf, 'summary') and hasattr(tf.summary, 'FileWriter'):
            if tensorboard_graph_vis:
                train_writer = tf.summary.FileWriter(tensorboard_dir + '/train', sess.graph)
                val_writer = tf.summary.FileWriter(tensorboard_dir + '/validation', sess.graph)
            else:
                train_writer = tf.summary.FileWriter(tensorboard_dir + '/train')
                val_writer = tf.summary.FileWriter(tensorboard_dir + '/validation')

        #Set up summary nodes
        if (tensorboard_weight_histograms):
            for param in network.all_params:
                if hasattr(tf, 'summary') and hasattr(tf.summary, 'histogram'):
                    tl.logging.info('Param name %s' % param.name)
                    tf.summary.histogram(param.name, param)

        if hasattr(tf, 'summary') and hasattr(tf.summary, 'histogram'):
            tf.summary.scalar('cost', cost)

        merged = tf.summary.merge_all()

        #Initalize all variables and summaries
        tl.layers.initialize_global_variables(sess)
        tl.logging.info("Finished! use `tensorboard --logdir=%s/` to start tensorboard" % tensorboard_dir)

    tl.logging.info("Start training the network ...")
    start_time_begin = time.time()
    tensorboard_train_index, tensorboard_val_index = 0, 0
    for epoch in range(n_epoch):
        start_time = time.time()
        loss_ep = 0
        n_step = 0
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(network.all_drop)  # enable noise layers
            loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
            loss_ep += loss
            n_step += 1
        loss_ep = loss_ep / n_step

        if tensorboard_dir is not None and hasattr(tf, 'summary'):
            if epoch + 1 == 1 or (epoch + 1) % tensorboard_epoch_freq == 0:
                for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                    dp_dict = dict_to_one(network.all_drop)  # disable noise layers
                    feed_dict = {x: X_train_a, y_: y_train_a}
                    feed_dict.update(dp_dict)
                    result = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(result, tensorboard_train_index)
                    tensorboard_train_index += 1
                if (X_val is not None) and (y_val is not None):
                    for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
                        dp_dict = dict_to_one(network.all_drop)  # disable noise layers
                        feed_dict = {x: X_val_a, y_: y_val_a}
                        feed_dict.update(dp_dict)
                        result = sess.run(merged, feed_dict=feed_dict)
                        val_writer.add_summary(result, tensorboard_val_index)
                        tensorboard_val_index += 1

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            if (X_val is not None) and (y_val is not None):
                tl.logging.info("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                if eval_train is True:
                    train_loss, train_acc, n_batch = 0, 0, 0
                    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                        dp_dict = dict_to_one(network.all_drop)  # disable noise layers
                        feed_dict = {x: X_train_a, y_: y_train_a}
                        feed_dict.update(dp_dict)
                        if acc is not None:
                            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                            train_acc += ac
                        else:
                            err = sess.run(cost, feed_dict=feed_dict)
                        train_loss += err
                        n_batch += 1
                    tl.logging.info("   train loss: %f" % (train_loss / n_batch))
                    if acc is not None:
                        tl.logging.info("   train acc: %f" % (train_acc / n_batch))
                val_loss, val_acc, n_batch = 0, 0, 0
                for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
                    dp_dict = dict_to_one(network.all_drop)  # disable noise layers
                    feed_dict = {x: X_val_a, y_: y_val_a}
                    feed_dict.update(dp_dict)
                    if acc is not None:
                        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                        val_acc += ac
                    else:
                        err = sess.run(cost, feed_dict=feed_dict)
                    val_loss += err
                    n_batch += 1

                tl.logging.info("   val loss: %f" % (val_loss / n_batch))

                if acc is not None:
                    tl.logging.info("   val acc: %f" % (val_acc / n_batch))
            else:
                tl.logging.info(
                    "Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep)
                )
    tl.logging.info("Total training time: %fs" % (time.time() - start_time_begin))


def test(sess, network, acc, X_test, y_test, x, y_, batch_size, cost=None):
    """
    Test a given non time-series network by the given test data and metric.

    Parameters
    ----------
    sess : Session
        TensorFlow session.
    network : TensorLayer layer
        The network.
    acc : TensorFlow expression or None
        Metric for accuracy or others.
            - If None, would not print the information.
    X_test : numpy.array
        The input of testing data.
    y_test : numpy array
        The target of testing data
    x : placeholder
        For inputs.
    y_ : placeholder
        For targets.
    batch_size : int or None
        The batch size for testing, when dataset is large, we should use minibatche for testing;
        if dataset is small, we can set it to None.
    cost : TensorFlow expression or None
        Metric for cost or others. If None, would not print the information.

    Examples
    --------
    See `tutorial_mnist_simple.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_simple.py>`_

    >>> tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

    """
    tl.logging.info('Start testing the network ...')
    if batch_size is None:
        dp_dict = dict_to_one(network.all_drop)
        feed_dict = {x: X_test, y_: y_test}
        feed_dict.update(dp_dict)

        if cost is not None:
            tl.logging.info("   test loss: %f" % sess.run(cost, feed_dict=feed_dict))

        test_acc = sess.run(acc, feed_dict=feed_dict)
        tl.logging.info("   test acc: %f" % test_acc)
        # tl.logging.info("   test acc: %f" % np.mean(y_test == sess.run(y_op,
        #                                           feed_dict=feed_dict)))
        return test_acc
    else:
        test_loss, test_acc, n_batch = 0, 0, 0
        for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
            dp_dict = dict_to_one(network.all_drop)  # disable noise layers
            feed_dict = {x: X_test_a, y_: y_test_a}
            feed_dict.update(dp_dict)
            if cost is not None:
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                test_loss += err
            else:
                ac = sess.run(acc, feed_dict=feed_dict)
            test_acc += ac
            n_batch += 1
        if cost is not None:
            tl.logging.info("   test loss: %f" % (test_loss / n_batch))
        tl.logging.info("   test acc: %f" % (test_acc / n_batch))
        return test_acc / n_batch


def predict(sess, network, X, x, y_op, batch_size=None):
    """
    Return the predict results of given non time-series network.

    Parameters
    ----------
    sess : Session
        TensorFlow Session.
    network : TensorLayer layer
        The network.
    X : numpy.array
        The inputs.
    x : placeholder
        For inputs.
    y_op : placeholder
        The argmax expression of softmax outputs.
    batch_size : int or None
        The batch size for prediction, when dataset is large, we should use minibatche for prediction;
        if dataset is small, we can set it to None.

    Examples
    --------
    See `tutorial_mnist_simple.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_simple.py>`_

    >>> y = network.outputs
    >>> y_op = tf.argmax(tf.nn.softmax(y), 1)
    >>> print(tl.utils.predict(sess, network, X_test, x, y_op))

    """
    if batch_size is None:
        dp_dict = dict_to_one(network.all_drop)  # disable noise layers
        feed_dict = {
            x: X,
        }
        feed_dict.update(dp_dict)
        return sess.run(y_op, feed_dict=feed_dict)
    else:
        result = None
        for X_a, _ in tl.iterate.minibatches(X, X, batch_size, shuffle=False):
            dp_dict = dict_to_one(network.all_drop)
            feed_dict = {
                x: X_a,
            }
            feed_dict.update(dp_dict)
            result_a = sess.run(y_op, feed_dict=feed_dict)
            if result is None:
                result = result_a
            else:
                result = np.concatenate((result, result_a))
        if result is None:
            if len(X) % batch_size != 0:
                dp_dict = dict_to_one(network.all_drop)
                feed_dict = {
                    x: X[-(len(X) % batch_size):, :],
                }
                feed_dict.update(dp_dict)
                result_a = sess.run(y_op, feed_dict=feed_dict)
                result = result_a
        else:
            if len(X) != len(result) and len(X) % batch_size != 0:
                dp_dict = dict_to_one(network.all_drop)
                feed_dict = {
                    x: X[-(len(X) % batch_size):, :],
                }
                feed_dict.update(dp_dict)
                result_a = sess.run(y_op, feed_dict=feed_dict)
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

    Examples
    --------
    >>> dp_dict = dict_to_one( network.all_drop )
    >>> dp_dict = dict_to_one( network.all_drop )
    >>> feed_dict.update(dp_dict)

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


def exit_tensorflow(sess=None, port=6006):
    """Close TensorFlow session, TensorBoard and Nvidia-process if available.

    Parameters
    ----------
    sess : Session
        TensorFlow Session.
    tb_port : int
        TensorBoard port you want to close, `6006` as default.

    """
    text = "[TL] Close tensorboard and nvidia-process if available"
    text2 = "[TL] Close tensorboard and nvidia-process not yet supported by this function (tl.ops.exit_tf) on "

    if sess is not None:
        sess.close()

    if _platform == "linux" or _platform == "linux2":
        tl.logging.info('linux: %s' % text)
        os.system('nvidia-smi')
        os.system('fuser ' + port + '/tcp -k')  # kill tensorboard 6006
        os.system("nvidia-smi | grep python |awk '{print $3}'|xargs kill")  # kill all nvidia-smi python process
        _exit()

    elif _platform == "darwin":
        tl.logging.info('OS X: %s' % text)
        subprocess.Popen("lsof -i tcp:" + str(port) + "  | grep -v PID | awk '{print $2}' | xargs kill",
                         shell=True)  # kill tensorboard
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
        raise NotImplementedError()
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
    gpu_fraction : float
        Fraction of GPU memory, (0 ~ 1]

    References
    ----------
    - `TensorFlow using GPU <https://www.tensorflow.org/versions/r0.9/how_tos/using_gpu/index.html>`__

    """
    tl.logging.info("[TL]: GPU MEM Fraction %f" % gpu_fraction)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    return sess
