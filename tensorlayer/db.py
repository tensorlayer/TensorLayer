#! /usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import time
import os
import sys
from datetime import datetime

import gridfs
import pymongo

# from tensorlayer.files import load_graph_and_params
from tensorlayer.files import exists_or_mkdir
from tensorlayer.files import del_folder

from tensorlayer import logging

import tensorflow as tf
import numpy as np


class TensorHub(object):
    """It is a MongoDB based manager that help you to manage data, network architecture, parameters and logging.

    Parameters
    -------------
    ip : str
        Localhost or IP address.
    port : int
        Port number.
    dbname : str
        Database name.
    username : str or None
        User name, set to None if you do not need authentication.
    password : str
        Password.
    project_name : str or None
        Experiment key for this entire project, similar with the repository name of Github.

    Attributes
    ------------
    ip, port, dbname and other input parameters : see above
        See above.
    project_name : str
        The given project name, if no given, set to the script name.
    db : mongodb client
        See ``pymongo.MongoClient``.
    """

    # @deprecated_alias(db_name='dbname', user_name='username', end_support_version=2.1)
    def __init__(
            self, ip='localhost', port=27017, dbname='dbname', username='None', password='password', project_name=None
    ):
        self.ip = ip
        self.port = port
        self.dbname = dbname
        self.username = username

        print("[Database] Initializing ...")
        # connect mongodb
        client = pymongo.MongoClient(ip, port)
        self.db = client[dbname]
        if username is None:
            print(username, password)
            self.db.authenticate(username, password)
        else:
            print("[Database] No username given, it works if authentication is not required")
        if project_name is None:
            self.project_name = sys.argv[0].split('.')[0]
            print("[Database] No project_name given, use {}".format(self.project_name))
        else:
            self.project_name = project_name

        # define file system (Buckets)
        self.dataset_fs = gridfs.GridFS(self.db, collection="datasetFilesystem")
        self.model_fs = gridfs.GridFS(self.db, collection="modelfs")
        # self.params_fs = gridfs.GridFS(self.db, collection="parametersFilesystem")
        # self.architecture_fs = gridfs.GridFS(self.db, collection="architectureFilesystem")

        print("[Database] Connected ")
        _s = "[Database] Info:\n"
        _s += "  ip             : {}\n".format(self.ip)
        _s += "  port           : {}\n".format(self.port)
        _s += "  dbname         : {}\n".format(self.dbname)
        _s += "  username       : {}\n".format(self.username)
        _s += "  password       : {}\n".format("*******")
        _s += "  project_name : {}\n".format(self.project_name)
        self._s = _s
        print(self._s)

    def __str__(self):
        """Print information of databset."""
        return self._s

    def _fill_project_info(self, args):
        """Fill in project_name for all studies, architectures and parameters."""
        return args.update({'project_name': self.project_name})

    @staticmethod
    def _serialization(ps):
        """Serialize data."""
        return pickle.dumps(ps, protocol=pickle.HIGHEST_PROTOCOL)  # protocol=2)
        # with open('_temp.pkl', 'wb') as file:
        #     return pickle.dump(ps, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _deserialization(ps):
        """Deseralize data."""
        return pickle.loads(ps)

    # =========================== MODELS ================================
    def save_model(self, network=None, model_name='model', **kwargs):
        """Save model architecture and parameters into database, timestamp will be added automatically.

        Parameters
        ----------
        network : TensorLayer layer
            TensorLayer layer instance.
        model_name : str
            The name/key of model.
        kwargs : other events
            Other events, such as name, accuracy, loss, step number and etc (optinal).

        Examples
        ---------
        Save model architecture and parameters into database.
        >>> db.save_model(net, accuracy=0.8, loss=2.3, name='second_model')

        Load one model with parameters from database (run this in other script)
        >>> net = db.find_top_model(sess=sess, accuracy=0.8, loss=2.3)

        Find and load the latest model.
        >>> net = db.find_top_model(sess=sess, sort=[("time", pymongo.DESCENDING)])
        >>> net = db.find_top_model(sess=sess, sort=[("time", -1)])

        Find and load the oldest model.
        >>> net = db.find_top_model(sess=sess, sort=[("time", pymongo.ASCENDING)])
        >>> net = db.find_top_model(sess=sess, sort=[("time", 1)])

        Get model information
        >>> net._accuracy
        ... 0.8

        Returns
        ---------
        boolean : True for success, False for fail.
        """
        kwargs.update({'model_name': model_name})
        self._fill_project_info(kwargs)  # put project_name into kwargs

        params = network.get_all_params()

        s = time.time()

        kwargs.update({'architecture': network.all_graphs, 'time': datetime.utcnow()})

        try:
            params_id = self.model_fs.put(self._serialization(params))
            kwargs.update({'params_id': params_id, 'time': datetime.utcnow()})
            self.db.Model.insert_one(kwargs)
            print("[Database] Save model: SUCCESS, took: {}s".format(round(time.time() - s, 2)))
            return True
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.info("{}  {}  {}  {}  {}".format(exc_type, exc_obj, fname, exc_tb.tb_lineno, e))
            print("[Database] Save model: FAIL")
            return False

    def find_top_model(self, sess, sort=None, model_name='model', **kwargs):
        """Finds and returns a model architecture and its parameters from the database which matches the requirement.

        Parameters
        ----------
        sess : Session
            TensorFlow session.
        sort : List of tuple
            PyMongo sort comment, search "PyMongo find one sorting" and `collection level operations <http://api.mongodb.com/python/current/api/pymongo/collection.html>`__ for more details.
        model_name : str or None
            The name/key of model.
        kwargs : other events
            Other events, such as name, accuracy, loss, step number and etc (optinal).

        Examples
        ---------
        - see ``save_model``.

        Returns
        ---------
        network : TensorLayer layer
            Note that, the returned network contains all information of the document (record), e.g. if you saved accuracy in the document, you can get the accuracy by using ``net._accuracy``.
        """
        # print(kwargs)   # {}
        kwargs.update({'model_name': model_name})
        self._fill_project_info(kwargs)

        s = time.time()

        d = self.db.Model.find_one(filter=kwargs, sort=sort)

        _temp_file_name = '_find_one_model_ztemp_file'
        if d is not None:
            params_id = d['params_id']
            graphs = d['architecture']
            _datetime = d['time']
            exists_or_mkdir(_temp_file_name, False)
            with open(os.path.join(_temp_file_name, 'graph.pkl'), 'wb') as file:
                pickle.dump(graphs, file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("[Database] FAIL! Cannot find model: {}".format(kwargs))
            return False
        try:
            params = self._deserialization(self.model_fs.get(params_id).read())
            np.savez(os.path.join(_temp_file_name, 'params.npz'), params=params)

            network = load_graph_and_params(name=_temp_file_name, sess=sess)
            del_folder(_temp_file_name)

            pc = self.db.Model.find(kwargs)
            print(
                "[Database] Find one model SUCCESS. kwargs:{} sort:{} save time:{} took: {}s".format(
                    kwargs, sort, _datetime, round(time.time() - s, 2)
                )
            )

            # put all informations of model into the TL layer
            for key in d:
                network.__dict__.update({"_%s" % key: d[key]})

            # check whether more parameters match the requirement
            params_id_list = pc.distinct('params_id')
            n_params = len(params_id_list)
            if n_params != 1:
                print("     Note that there are {} models match the kwargs".format(n_params))
            return network
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.info("{}  {}  {}  {}  {}".format(exc_type, exc_obj, fname, exc_tb.tb_lineno, e))
            return False

    def delete_model(self, **kwargs):
        """Delete model.

        Parameters
        -----------
        kwargs : logging information
            Find items to delete, leave it empty to delete all log.
        """
        self._fill_project_info(kwargs)
        self.db.Model.delete_many(kwargs)
        logging.info("[Database] Delete Model SUCCESS")

    # =========================== DATASET ===============================
    def save_dataset(self, dataset=None, dataset_name=None, **kwargs):
        """Saves one dataset into database, timestamp will be added automatically.

        Parameters
        ----------
        dataset : any type
            The dataset you want to store.
        dataset_name : str
            The name of dataset.
        kwargs : other events
            Other events, such as description, author and etc (optinal).

        Examples
        ----------
        Save dataset
        >>> db.save_dataset([X_train, y_train, X_test, y_test], 'mnist', description='this is a tutorial')

        Get dataset
        >>> dataset = db.find_top_dataset('mnist')

        Returns
        ---------
        boolean : Return True if save success, otherwise, return False.
        """
        self._fill_project_info(kwargs)
        if dataset_name is None:
            raise Exception("dataset_name is None, please give a dataset name")
        kwargs.update({'dataset_name': dataset_name})

        s = time.time()
        try:
            dataset_id = self.dataset_fs.put(self._serialization(dataset))
            kwargs.update({'dataset_id': dataset_id, 'time': datetime.utcnow()})
            self.db.Dataset.insert_one(kwargs)
            # print("[Database] Save params: {} SUCCESS, took: {}s".format(file_name, round(time.time()-s, 2)))
            print("[Database] Save dataset: SUCCESS, took: {}s".format(round(time.time() - s, 2)))
            return True
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.info("{}  {}  {}  {}  {}".format(exc_type, exc_obj, fname, exc_tb.tb_lineno, e))
            print("[Database] Save dataset: FAIL")
            return False

    def find_top_dataset(self, dataset_name=None, sort=None, **kwargs):
        """Finds and returns a dataset from the database which matches the requirement.

        Parameters
        ----------
        dataset_name : str
            The name of dataset.
        sort : List of tuple
            PyMongo sort comment, search "PyMongo find one sorting" and `collection level operations <http://api.mongodb.com/python/current/api/pymongo/collection.html>`__ for more details.
        kwargs : other events
            Other events, such as description, author and etc (optinal).

        Examples
        ---------
        Save dataset
        >>> db.save_dataset([X_train, y_train, X_test, y_test], 'mnist', description='this is a tutorial')

        Get dataset
        >>> dataset = db.find_top_dataset('mnist')
        >>> datasets = db.find_datasets('mnist')

        Returns
        --------
        dataset : the dataset or False
            Return False if nothing found.

        """

        self._fill_project_info(kwargs)
        if dataset_name is None:
            raise Exception("dataset_name is None, please give a dataset name")
        kwargs.update({'dataset_name': dataset_name})

        s = time.time()

        d = self.db.Dataset.find_one(filter=kwargs, sort=sort)

        if d is not None:
            dataset_id = d['dataset_id']
        else:
            print("[Database] FAIL! Cannot find dataset: {}".format(kwargs))
            return False
        try:
            dataset = self._deserialization(self.dataset_fs.get(dataset_id).read())
            pc = self.db.Dataset.find(kwargs)
            print("[Database] Find one dataset SUCCESS, {} took: {}s".format(kwargs, round(time.time() - s, 2)))

            # check whether more datasets match the requirement
            dataset_id_list = pc.distinct('dataset_id')
            n_dataset = len(dataset_id_list)
            if n_dataset != 1:
                print("     Note that there are {} datasets match the requirement".format(n_dataset))
            return dataset
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.info("{}  {}  {}  {}  {}".format(exc_type, exc_obj, fname, exc_tb.tb_lineno, e))
            return False

    def find_datasets(self, dataset_name=None, **kwargs):
        """Finds and returns all datasets from the database which matches the requirement.
        In some case, the data in a dataset can be stored separately for better management.

        Parameters
        ----------
        dataset_name : str
            The name/key of dataset.
        kwargs : other events
            Other events, such as description, author and etc (optional).

        Returns
        --------
        params : the parameters, return False if nothing found.

        """

        self._fill_project_info(kwargs)
        if dataset_name is None:
            raise Exception("dataset_name is None, please give a dataset name")
        kwargs.update({'dataset_name': dataset_name})

        s = time.time()
        pc = self.db.Dataset.find(kwargs)

        if pc is not None:
            dataset_id_list = pc.distinct('dataset_id')
            dataset_list = []
            for dataset_id in dataset_id_list:  # you may have multiple Buckets files
                tmp = self.dataset_fs.get(dataset_id).read()
                dataset_list.append(self._deserialization(tmp))
        else:
            print("[Database] FAIL! Cannot find any dataset: {}".format(kwargs))
            return False

        print("[Database] Find {} datasets SUCCESS, took: {}s".format(len(dataset_list), round(time.time() - s, 2)))
        return dataset_list

    def delete_datasets(self, **kwargs):
        """Delete datasets.

        Parameters
        -----------
        kwargs : logging information
            Find items to delete, leave it empty to delete all log.

        """

        self._fill_project_info(kwargs)
        self.db.Dataset.delete_many(kwargs)
        logging.info("[Database] Delete Dataset SUCCESS")

    # =========================== LOGGING ===============================
    def save_training_log(self, **kwargs):
        """Saves the training log, timestamp will be added automatically.

        Parameters
        -----------
        kwargs : logging information
            Events, such as accuracy, loss, step number and etc.

        Examples
        ---------
        >>> db.save_training_log(accuracy=0.33, loss=0.98)

        """

        self._fill_project_info(kwargs)
        kwargs.update({'time': datetime.utcnow()})
        _result = self.db.TrainLog.insert_one(kwargs)
        _log = self._print_dict(kwargs)
        logging.info("[Database] train log: " + _log)

    def save_validation_log(self, **kwargs):
        """Saves the validation log, timestamp will be added automatically.

        Parameters
        -----------
        kwargs : logging information
            Events, such as accuracy, loss, step number and etc.

        Examples
        ---------
        >>> db.save_validation_log(accuracy=0.33, loss=0.98)

        """

        self._fill_project_info(kwargs)
        kwargs.update({'time': datetime.utcnow()})
        _result = self.db.ValidLog.insert_one(kwargs)
        _log = self._print_dict(kwargs)
        logging.info("[Database] valid log: " + _log)

    def save_testing_log(self, **kwargs):
        """Saves the testing log, timestamp will be added automatically.

        Parameters
        -----------
        kwargs : logging information
            Events, such as accuracy, loss, step number and etc.

        Examples
        ---------
        >>> db.save_testing_log(accuracy=0.33, loss=0.98)

        """

        self._fill_project_info(kwargs)
        kwargs.update({'time': datetime.utcnow()})
        _result = self.db.TestLog.insert_one(kwargs)
        _log = self._print_dict(kwargs)
        logging.info("[Database] test log: " + _log)

    def delete_training_log(self, **kwargs):
        """Deletes training log.

        Parameters
        -----------
        kwargs : logging information
            Find items to delete, leave it empty to delete all log.

        Examples
        ---------
        Save training log
        >>> db.save_training_log(accuracy=0.33)
        >>> db.save_training_log(accuracy=0.44)

        Delete logs that match the requirement
        >>> db.delete_training_log(accuracy=0.33)

        Delete all logs
        >>> db.delete_training_log()
        """
        self._fill_project_info(kwargs)
        self.db.TrainLog.delete_many(kwargs)
        logging.info("[Database] Delete TrainLog SUCCESS")

    def delete_validation_log(self, **kwargs):
        """Deletes validation log.

        Parameters
        -----------
        kwargs : logging information
            Find items to delete, leave it empty to delete all log.

        Examples
        ---------
        - see ``save_training_log``.
        """
        self._fill_project_info(kwargs)
        self.db.ValidLog.delete_many(kwargs)
        logging.info("[Database] Delete ValidLog SUCCESS")

    def delete_testing_log(self, **kwargs):
        """Deletes testing log.

        Parameters
        -----------
        kwargs : logging information
            Find items to delete, leave it empty to delete all log.

        Examples
        ---------
        - see ``save_training_log``.
        """
        self._fill_project_info(kwargs)
        self.db.TestLog.delete_many(kwargs)
        logging.info("[Database] Delete TestLog SUCCESS")

    # def find_training_logs(self, **kwargs):
    #     pass
    #
    # def find_validation_logs(self, **kwargs):
    #     pass
    #
    # def find_testing_logs(self, **kwargs):
    #     pass

    # =========================== Task ===================================
    def create_task(self, task_name=None, script=None, hyper_parameters=None, saved_result_keys=None, **kwargs):
        """Uploads a task to the database, timestamp will be added automatically.

        Parameters
        -----------
        task_name : str
            The task name.
        script : str
            File name of the python script.
        hyper_parameters : dictionary
            The hyper parameters pass into the script.
        saved_result_keys : list of str
            The keys of the task results to keep in the database when the task finishes.
        kwargs : other parameters
            Users customized parameters such as description, version number.

        Examples
        -----------
        Uploads a task
        >>> db.create_task(task_name='mnist', script='example/tutorial_mnist_simple.py', description='simple tutorial')

        Finds and runs the latest task
        >>> db.run_top_task(sess=sess, sort=[("time", pymongo.DESCENDING)])
        >>> db.run_top_task(sess=sess, sort=[("time", -1)])

        Finds and runs the oldest task
        >>> db.run_top_task(sess=sess, sort=[("time", pymongo.ASCENDING)])
        >>> db.run_top_task(sess=sess, sort=[("time", 1)])

        """
        if not isinstance(task_name, str):  # is None:
            raise Exception("task_name should be string")
        if not isinstance(script, str):  # is None:
            raise Exception("script should be string")
        if hyper_parameters is None:
            hyper_parameters = {}
        if saved_result_keys is None:
            saved_result_keys = []

        self._fill_project_info(kwargs)
        kwargs.update({'time': datetime.utcnow()})
        kwargs.update({'hyper_parameters': hyper_parameters})
        kwargs.update({'saved_result_keys': saved_result_keys})

        _script = open(script, 'rb').read()

        kwargs.update({'status': 'pending', 'script': _script, 'result': {}})
        self.db.Task.insert_one(kwargs)
        logging.info("[Database] Saved Task - task_name: {} script: {}".format(task_name, script))

    def run_top_task(self, task_name=None, sort=None, **kwargs):
        """Finds and runs a pending task that in the first of the sorting list.

        Parameters
        -----------
        task_name : str
            The task name.
        sort : List of tuple
            PyMongo sort comment, search "PyMongo find one sorting" and `collection level operations <http://api.mongodb.com/python/current/api/pymongo/collection.html>`__ for more details.
        kwargs : other parameters
            Users customized parameters such as description, version number.

        Examples
        ---------
        Monitors the database and pull tasks to run
        >>> while True:
        >>>     print("waiting task from distributor")
        >>>     db.run_top_task(task_name='mnist', sort=[("time", -1)])
        >>>     time.sleep(1)

        Returns
        --------
        boolean : True for success, False for fail.
        """
        if not isinstance(task_name, str):  # is None:
            raise Exception("task_name should be string")
        self._fill_project_info(kwargs)
        kwargs.update({'status': 'pending'})

        # find task and set status to running
        task = self.db.Task.find_one_and_update(kwargs, {'$set': {'status': 'running'}}, sort=sort)

        try:
            # get task info e.g. hyper parameters, python script
            if task is None:
                logging.info("[Database] Find Task FAIL: key: {} sort: {}".format(task_name, sort))
                return False
            else:
                logging.info("[Database] Find Task SUCCESS: key: {} sort: {}".format(task_name, sort))
            _datetime = task['time']
            _script = task['script']
            _id = task['_id']
            _hyper_parameters = task['hyper_parameters']
            _saved_result_keys = task['saved_result_keys']
            logging.info("  hyper parameters:")
            for key in _hyper_parameters:
                globals()[key] = _hyper_parameters[key]
                logging.info("    {}: {}".format(key, _hyper_parameters[key]))
            # run task
            s = time.time()
            logging.info("[Database] Start Task: key: {} sort: {} push time: {}".format(task_name, sort, _datetime))
            _script = _script.decode('utf-8')
            with tf.Graph().as_default():  # as graph: # clear all TF graphs
                exec(_script, globals())

            # set status to finished
            _ = self.db.Task.find_one_and_update({'_id': _id}, {'$set': {'status': 'finished'}})

            # return results
            __result = {}
            for _key in _saved_result_keys:
                logging.info("  result: {}={} {}".format(_key, globals()[_key], type(globals()[_key])))
                __result.update({"%s" % _key: globals()[_key]})
            _ = self.db.Task.find_one_and_update(
                {
                    '_id': _id
                }, {'$set': {
                    'result': __result
                }}, return_document=pymongo.ReturnDocument.AFTER
            )
            logging.info(
                "[Database] Finished Task: task_name - {} sort: {} push time: {} took: {}s".format(
                    task_name, sort, _datetime,
                    time.time() - s
                )
            )
            return True
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.info("{}  {}  {}  {}  {}".format(exc_type, exc_obj, fname, exc_tb.tb_lineno, e))
            logging.info("[Database] Fail to run task")
            # if fail, set status back to pending
            _ = self.db.Task.find_one_and_update({'_id': _id}, {'$set': {'status': 'pending'}})
            return False

    def delete_tasks(self, **kwargs):
        """Delete tasks.

        Parameters
        -----------
        kwargs : logging information
            Find items to delete, leave it empty to delete all log.

        Examples
        ---------
        >>> db.delete_tasks()

        """

        self._fill_project_info(kwargs)
        self.db.Task.delete_many(kwargs)
        logging.info("[Database] Delete Task SUCCESS")

    def check_unfinished_task(self, task_name=None, **kwargs):
        """Finds and runs a pending task.

        Parameters
        -----------
        task_name : str
            The task name.
        kwargs : other parameters
            Users customized parameters such as description, version number.

        Examples
        ---------
        Wait until all tasks finish in user's local console

        >>> while not db.check_unfinished_task():
        >>>     time.sleep(1)
        >>> print("all tasks finished")
        >>> sess = tf.InteractiveSession()
        >>> net = db.find_top_model(sess=sess, sort=[("test_accuracy", -1)])
        >>> print("the best accuracy {} is from model {}".format(net._test_accuracy, net._name))

        Returns
        --------
        boolean : True for success, False for fail.

        """

        if not isinstance(task_name, str):  # is None:
            raise Exception("task_name should be string")
        self._fill_project_info(kwargs)

        kwargs.update({'$or': [{'status': 'pending'}, {'status': 'running'}]})

        # ## find task
        # task = self.db.Task.find_one(kwargs)
        task = self.db.Task.find(kwargs)

        task_id_list = task.distinct('_id')
        n_task = len(task_id_list)

        if n_task == 0:
            logging.info("[Database] No unfinished task - task_name: {}".format(task_name))
            return False
        else:

            logging.info("[Database] Find {} unfinished task - task_name: {}".format(n_task, task_name))
            return True

    @staticmethod
    def _print_dict(args):
        string = ''
        for key, value in args.items():
            if key is not '_id':
                string += str(key) + ": " + str(value) + " / "
        return string
