## Evaluation
def evaluation(y_test, y_predict, n_classes):
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

def dict_to_one(dp_dict):
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

def class_balancing_oversample(X_train, y_train, printable=True):
    """
    Input the features and labels, return the features and labels
    after oversampling.

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
