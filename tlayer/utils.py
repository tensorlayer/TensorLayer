#! /usr/bin/python
# -*- coding: utf8 -*-



import numpy as np

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
