#! /usr/bin/python
# -*- coding: utf8 -*-
import tensorflow as tf
import tensorlayer as tl
from . import iterate
import numpy as np
import time
import math


def fit(sess, network, train_op, cost, X_train, y_train, x, y_, acc=None, batch_size=100,
        n_epoch=100, print_freq=5, X_val=None, y_val=None, eval_train=True,
        tensorboard=False, tensorboard_epoch_freq=5, tensorboard_weight_histograms=True, tensorboard_graph_vis=True):
    """Traing a given non time-series network by the given cost function, training data, batch_size, n_epoch etc.

    Parameters
    ----------
    sess : TensorFlow session
        sess = tf.InteractiveSession()
    network : a TensorLayer layer
        the network will be trained
    train_op : a TensorFlow optimizer
        like tf.train.AdamOptimizer
    X_train : numpy array
        the input of training data
    y_train : numpy array
        the target of training data
    x : placeholder
        for inputs
    y_ : placeholder
        for targets
    acc : the TensorFlow expression of accuracy (or other metric) or None
        if None, would not display the metric
    batch_size : int
        batch size for training and evaluating
    n_epoch : int
        the number of training epochs
    print_freq : int
        display the training information every ``print_freq`` epochs
    X_val : numpy array or None
        the input of validation data
    y_val : numpy array or None
        the target of validation data
    eval_train : boolean
        if X_val and y_val are not None, it refects whether to evaluate the training data
    tensorboard : boolean
        if True summary data will be stored to the log/ direcory for visualization with tensorboard.
        See also detailed tensorboard_X settings for specific configurations of features. (default False)
        Also runs tl.layers.initialize_global_variables(sess) internally in fit() to setup the summary nodes, see Note:
    tensorboard_epoch_freq : int
        how many epochs between storing tensorboard checkpoint for visualization to log/ directory (default 5)
    tensorboard_weight_histograms : boolean
        if True updates tensorboard data in the logs/ directory for visulaization
        of the weight histograms every tensorboard_epoch_freq epoch (default True)
    tensorboard_graph_vis : boolean
        if True stores the graph in the tensorboard summaries saved to log/ (default True)
    Examples
    --------
    >>> see tutorial_mnist_simple.py
    >>> tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
    ...            acc=acc, batch_size=500, n_epoch=200, print_freq=5,
    ...            X_val=X_val, y_val=y_val, eval_train=False)
    >>> tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
    ...            acc=acc, batch_size=500, n_epoch=200, print_freq=5,
    ...            X_val=X_val, y_val=y_val, eval_train=False,
    ...            tensorboard=True, tensorboard_weight_histograms=True, tensorboard_graph_vis=True)

    Note
    --------
        If tensorboard=True, the global_variables_initializer will be run inside the fit function
        in order to initalize the automatically generated summary nodes used for tensorboard visualization,
        thus tf.global_variables_initializer().run() before the fit() call will be undefined.
    """
    assert X_train.shape[0] >= batch_size, "Number of training examples should be bigger than the batch size"

    if(tensorboard):
        print("Setting up tensorboard ...")
        #Set up tensorboard summaries and saver
        tl.files.exists_or_mkdir('logs/')

        #Only write summaries for more recent TensorFlow versions
        if hasattr(tf, 'summary') and hasattr(tf.summary, 'FileWriter'):
            if tensorboard_graph_vis:
                train_writer = tf.summary.FileWriter('logs/train',sess.graph)
                val_writer = tf.summary.FileWriter('logs/validation',sess.graph)
            else:
                train_writer = tf.summary.FileWriter('logs/train')
                val_writer = tf.summary.FileWriter('logs/validation')

        #Set up summary nodes
        if(tensorboard_weight_histograms):
            for param in network.all_params:
                if hasattr(tf, 'summary') and hasattr(tf.summary, 'histogram'):
                    print('Param name ', param.name)
                    tf.summary.histogram(param.name, param)

        if hasattr(tf, 'summary') and hasattr(tf.summary, 'histogram'):
            tf.summary.scalar('cost', cost)

        merged = tf.summary.merge_all()

        #Initalize all variables and summaries
        tl.layers.initialize_global_variables(sess)
        print("Finished! use $tensorboard --logdir=logs/ to start server")

    print("Start training the network ...")
    start_time_begin = time.time()
    tensorboard_train_index, tensorboard_val_index = 0, 0
    for epoch in range(n_epoch):
        start_time = time.time()
        loss_ep = 0; n_step = 0
        for X_train_a, y_train_a in iterate.minibatches(X_train, y_train,
                                                    batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )    # enable noise layers
            loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
            loss_ep += loss
            n_step += 1
        loss_ep = loss_ep/ n_step

        if tensorboard and hasattr(tf, 'summary'):
            if epoch+1 == 1 or (epoch+1) % tensorboard_epoch_freq == 0:
                for X_train_a, y_train_a in iterate.minibatches(
                                        X_train, y_train, batch_size, shuffle=True):
                    dp_dict = dict_to_one( network.all_drop )    # disable noise layers
                    feed_dict = {x: X_train_a, y_: y_train_a}
                    feed_dict.update(dp_dict)
                    result = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(result, tensorboard_train_index)
                    tensorboard_train_index += 1

                for X_val_a, y_val_a in iterate.minibatches(
                                        X_val, y_val, batch_size, shuffle=True):
                    dp_dict = dict_to_one( network.all_drop )    # disable noise layers
                    feed_dict = {x: X_val_a, y_: y_val_a}
                    feed_dict.update(dp_dict)
                    result = sess.run(merged, feed_dict=feed_dict)
                    val_writer.add_summary(result, tensorboard_val_index)
                    tensorboard_val_index += 1

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            if (X_val is not None) and (y_val is not None):
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                if eval_train is True:
                    train_loss, train_acc, n_batch = 0, 0, 0
                    for X_train_a, y_train_a in iterate.minibatches(
                                            X_train, y_train, batch_size, shuffle=True):
                        dp_dict = dict_to_one( network.all_drop )    # disable noise layers
                        feed_dict = {x: X_train_a, y_: y_train_a}
                        feed_dict.update(dp_dict)
                        if acc is not None:
                            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                            train_acc += ac
                        else:
                            err = sess.run(cost, feed_dict=feed_dict)
                        train_loss += err;  n_batch += 1
                    print("   train loss: %f" % (train_loss/ n_batch))
                    if acc is not None:
                        print("   train acc: %f" % (train_acc/ n_batch))
                val_loss, val_acc, n_batch = 0, 0, 0
                for X_val_a, y_val_a in iterate.minibatches(
                                            X_val, y_val, batch_size, shuffle=True):
                    dp_dict = dict_to_one( network.all_drop )    # disable noise layers
                    feed_dict = {x: X_val_a, y_: y_val_a}
                    feed_dict.update(dp_dict)
                    if acc is not None:
                        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                        val_acc += ac
                    else:
                        err = sess.run(cost, feed_dict=feed_dict)
                    val_loss += err; n_batch += 1
                print("   val loss: %f" % (val_loss/ n_batch))
                if acc is not None:
                    print("   val acc: %f" % (val_acc/ n_batch))
            else:
                print("Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep))
    print("Total training time: %fs" % (time.time() - start_time_begin))


def test(sess, network, acc, X_test, y_test, x, y_, batch_size, cost=None):
    """
    Test a given non time-series network by the given test data and metric.

    Parameters
    ----------
    sess : TensorFlow session
        sess = tf.InteractiveSession()
    network : a TensorLayer layer
        the network will be trained
    acc : the TensorFlow expression of accuracy (or other metric) or None
        if None, would not display the metric
    X_test : numpy array
        the input of test data
    y_test : numpy array
        the target of test data
    x : placeholder
        for inputs
    y_ : placeholder
        for targets
    batch_size : int or None
        batch size for testing, when dataset is large, we should use minibatche for testing.
        when dataset is small, we can set it to None.
    cost : the TensorFlow expression of cost or None
        if None, would not display the cost

    Examples
    --------
    >>> see tutorial_mnist_simple.py
    >>> tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)
    """
    print('Start testing the network ...')
    if batch_size is None:
        dp_dict = dict_to_one( network.all_drop )
        feed_dict = {x: X_test, y_: y_test}
        feed_dict.update(dp_dict)
        if cost is not None:
            print("   test loss: %f" % sess.run(cost, feed_dict=feed_dict))
        print("   test acc: %f" % sess.run(acc, feed_dict=feed_dict))
            # print("   test acc: %f" % np.mean(y_test == sess.run(y_op,
            #                                           feed_dict=feed_dict)))
    else:
        test_loss, test_acc, n_batch = 0, 0, 0
        for X_test_a, y_test_a in iterate.minibatches(
                                    X_test, y_test, batch_size, shuffle=True):
            dp_dict = dict_to_one( network.all_drop )    # disable noise layers
            feed_dict = {x: X_test_a, y_: y_test_a}
            feed_dict.update(dp_dict)
            if cost is not None:
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                test_loss += err
            else:
                ac = sess.run(acc, feed_dict=feed_dict)
            test_acc += ac; n_batch += 1
        if cost is not None:
            print("   test loss: %f" % (test_loss/ n_batch))
        print("   test acc: %f" % (test_acc/ n_batch))


def predict(sess, network, X, x, y_op):
    """
    Return the predict results of given non time-series network.

    Parameters
    ----------
    sess : TensorFlow session
        sess = tf.InteractiveSession()
    network : a TensorLayer layer
        the network will be trained
    X : numpy array
        the input
    x : placeholder
        for inputs
    y_op : placeholder
        the argmax expression of softmax outputs

    Examples
    --------
    >>> see tutorial_mnist_simple.py
    >>> y = network.outputs
    >>> y_op = tf.argmax(tf.nn.softmax(y), 1)
    >>> print(tl.utils.predict(sess, network, X_test, x, y_op))
    """
    dp_dict = dict_to_one( network.all_drop )    # disable noise layers
    feed_dict = {x: X,}
    feed_dict.update(dp_dict)
    return sess.run(y_op, feed_dict=feed_dict)

## Evaluation
def evaluation(y_test=None, y_predict=None, n_classes=None):
    """
    Input the predicted results, targets results and
    the number of class, return the confusion matrix, F1-score of each class,
    accuracy and macro F1-score.

    Parameters
    ----------
    y_test : numpy.array or list
        target results
    y_predict : numpy.array or list
        predicted results
    n_classes : int
        number of classes

    Examples
    --------
    >>> c_mat, f1, acc, f1_macro = evaluation(y_test, y_predict, n_classes)
    """
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    c_mat = confusion_matrix(y_test, y_predict, labels = [x for x in range(n_classes)])
    f1    = f1_score(y_test, y_predict, average = None, labels = [x for x in range(n_classes)])
    f1_macro = f1_score(y_test, y_predict, average='macro')
    acc   = accuracy_score(y_test, y_predict)
    print('confusion matrix: \n',c_mat)
    print('f1-score:',f1)
    print('f1-score(macro):',f1_macro)   # same output with > f1_score(y_true, y_pred, average='macro')
    print('accuracy-score:', acc)
    return c_mat, f1, acc, f1_macro

def dict_to_one(dp_dict={}):
    """
    Input a dictionary, return a dictionary that all items are set to one,
    use for disable dropout, dropconnect layer and so on.

    Parameters
    ----------
    dp_dict : dictionary
        keeping probabilities

    Examples
    --------
    >>> dp_dict = dict_to_one( network.all_drop )
    >>> dp_dict = dict_to_one( network.all_drop )
    >>> feed_dict.update(dp_dict)
    """
    return {x: 1 for x in dp_dict}

def flatten_list(list_of_list=[[],[]]):
    """
    Input a list of list, return a list that all items are in a list.

    Parameters
    ----------
    list_of_list : a list of list

    Examples
    --------
    >>> tl.utils.flatten_list([[1, 2, 3],[4, 5],[6]])
    ... [1, 2, 3, 4, 5, 6]
    """
    return sum(list_of_list, [])


def class_balancing_oversample(X_train=None, y_train=None, printable=True):
    """Input the features and labels, return the features and labels after oversampling.

    Parameters
    ----------
    X_train : numpy.array
        Features, each row is an example
    y_train : numpy.array
        Labels

    Examples
    --------
    >>> X_train, y_train = class_balancing_oversample(X_train, y_train, printable=True)
    """
    # ======== Classes balancing
    if printable:
        print("Classes balancing for training examples...")
    from collections import Counter
    c = Counter(y_train)
    if printable:
        print('the occurrence number of each stage: %s' % c.most_common())
        print('the least stage is Label %s have %s instances' % c.most_common()[-1])
        print('the most stage is  Label %s have %s instances' % c.most_common(1)[0])
    most_num = c.most_common(1)[0][1]
    if printable:
        print('most num is %d, all classes tend to be this num' % most_num)

    locations = {}
    number = {}

    for lab, num in c.most_common():    # find the index from y_train
        number[lab] = num
        locations[lab] = np.where(np.array(y_train)==lab)[0]
    if printable:
        print('convert list(np.array) to dict format')
    X = {}  # convert list to dict
    for lab, num in number.items():
        X[lab] = X_train[locations[lab]]

    # oversampling
    if printable:
        print('start oversampling')
    for key in X:
        temp = X[key]
        while True:
            if len(X[key]) >= most_num:
                break
            X[key] = np.vstack((X[key], temp))
    if printable:
        print('first features of label 0 >', len(X[0][0]))
        print('the occurrence num of each stage after oversampling')
    for key in X:
        print(key, len(X[key]))
    if printable:
        print('make each stage have same num of instances')
    for key in X:
        X[key] = X[key][0:most_num,:]
        print(key, len(X[key]))

    # convert dict to list
    if printable:
        print('convert from dict to list format')
    y_train = []
    X_train = np.empty(shape=(0,len(X[0][0])))
    for key in X:
        X_train = np.vstack( (X_train, X[key] ) )
        y_train.extend([key for i in range(len(X[key]))])
    # print(len(X_train), len(y_train))
    c = Counter(y_train)
    if printable:
        print('the occurrence number of each stage after oversampling: %s' % c.most_common())
    # ================ End of Classes balancing
    return X_train, y_train




#
# def class_balancing_sequence_4D(X_train, y_train, sequence_length, model='downsampling' ,printable=True):
#     ''' 输入、输出都是sequence format
#         oversampling or downsampling
#     '''
#     n_features = X_train.shape[2]
#     # ======== Classes balancing for sequence
#     if printable:
#         print("Classes balancing for 4D sequence training examples...")
#     from collections import Counter
#     c = Counter(y_train)    # Counter({2: 454, 4: 267, 3: 124, 1: 57, 0: 48})
#     if printable:
#         print('the occurrence number of each stage: %s' % c.most_common())
#         print('the least Label %s have %s instances' % c.most_common()[-1])
#         print('the most  Label %s have %s instances' % c.most_common(1)[0])
#     # print(c.most_common()) # [(2, 454), (4, 267), (3, 124), (1, 57), (0, 48)]
#     most_num = c.most_common(1)[0][1]
#     less_num = c.most_common()[-1][1]
#
#     locations = {}
#     number = {}
#     for lab, num in c.most_common():
#         number[lab] = num
#         locations[lab] = np.where(np.array(y_train)==lab)[0]
#     # print(locations)
#     # print(number)
#     if printable:
#         print('  convert list to dict')
#     X = {}  # convert list to dict
#     ### a sequence
#     for lab, _ in number.items():
#         X[lab] = np.empty(shape=(0,1,n_features,1)) # 4D
#     for lab, _ in number.items():
#         #X[lab] = X_train[locations[lab]
#         for l in locations[lab]:
#             X[lab] = np.vstack((X[lab], X_train[l*sequence_length : (l+1)*(sequence_length)]))
#         # X[lab] = X_train[locations[lab]*sequence_length : locations[lab]*(sequence_length+1)]    # a sequence
#     # print(X)
#
#     if model=='oversampling':
#         if printable:
#             print('  oversampling -- most num is %d, all classes tend to be this num\nshuffle applied' % most_num)
#         for key in X:
#             temp = X[key]
#             while True:
#                 if len(X[key]) >= most_num * sequence_length:   # sequence
#                     break
#                 X[key] = np.vstack((X[key], temp))
#             # print(key, len(X[key]))
#         if printable:
#             print('  make each stage have same num of instances')
#         for key in X:
#             X[key] = X[key][0:most_num*sequence_length,:]   # sequence
#             if printable:
#                 print(key, len(X[key]))
#     elif model=='downsampling':
#         import random
#         if printable:
#             print('  downsampling -- less num is %d, all classes tend to be this num by randomly choice without replacement\nshuffle applied' % less_num)
#         for key in X:
#             # print(key, len(X[key]))#, len(X[key])/sequence_length)
#             s_idx = [ i for i in range(int(len(X[key])/sequence_length))]
#             s_idx = np.asarray(s_idx)*sequence_length   # start index of sequnce in X[key]
#             # print('s_idx',s_idx)
#             r_idx = np.random.choice(s_idx, less_num, replace=False)    # random choice less_num of s_idx
#             # print('r_idx',r_idx)
#             temp = X[key]
#             X[key] = np.empty(shape=(0,1,n_features,1)) # 4D
#             for idx in r_idx:
#                 X[key] = np.vstack((X[key], temp[idx:idx+sequence_length]))
#             # print(key, X[key])
#             # np.random.choice(l, len(l), replace=False)
#     else:
#         raise Exception('  model should be oversampling or downsampling')
#
#     # convert dict to list
#     if printable:
#         print('  convert dict to list')
#     y_train = []
#     # X_train = np.empty(shape=(0,len(X[0][0])))
#     # X_train = np.empty(shape=(0,len(X[1][0])))    # 2D
#     X_train = np.empty(shape=(0,1,n_features,1))    # 4D
#     l_key = list(X.keys())  # shuffle
#     random.shuffle(l_key)   # shuffle
#     # for key in X:     # no shuffle
#     for key in l_key:   # shuffle
#         X_train = np.vstack( (X_train, X[key] ) )
#         # print(len(X[key]))
#         y_train.extend([key for i in range(int(len(X[key])/sequence_length))])
#     # print(X_train,y_train, type(X_train), type(y_train))
#     # ================ End of Classes balancing for sequence
#     # print(X_train.shape, len(y_train))
#     return X_train, np.asarray(y_train)
