#! /usr/bin/python
# -*- coding: utf-8 -*-

import inspect
import pickle
import time
import uuid
import os, sys
from datetime import datetime

import gridfs
import pymongo
from tensorlayer.files import load_graph_and_params, exists_or_mkdir, del_folder
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
    project_key : str or None
        Experiment key for this entire project, similar with the repository name of Github.

    Attributes
    ------------
    ip, port, dbname and other input parameters : see above
        See above.
    project_key : str
        The given project name, if no given, set to the script name.
    db : mongodb client
        See ``pymongo.MongoClient``.
    """

    # @deprecated_alias(db_name='dbname', user_name='username', end_support_version=2.1)
    def __init__(
            self, ip='localhost', port=27017, dbname='dbname', username=None, password='password', project_key=None
    ):
        self.ip = ip
        self.port = port
        self.dbname = dbname
        self.username = username

        print("[Database] Initializing ...")
        ## connect mongodb
        client = pymongo.MongoClient(ip, port)
        self.db = client[dbname]
        if username != None:
            self.db.authenticate(username, password)
        else:
            print("[Database] No username given, it works if authentication is not required")
        if project_key is None:
            self.project_key = sys.argv[0].split('.')[0]
            print("[Database] No project_key given, use {}".format(self.project_key))
        else:
            self.project_key = project_key

        ## define file system (Buckets)
        self.dataset_fs = gridfs.GridFS(self.db, collection="datasetFilesystem")
        self.model_fs = gridfs.GridFS(self.db, collection="modelfs")
        # self.params_fs = gridfs.GridFS(self.db, collection="parametersFilesystem")
        # self.architecture_fs = gridfs.GridFS(self.db, collection="architectureFilesystem")

        ## print info
        print("[Database] Connected ")
        _s = "[Database] Info:\n"
        _s += "  ip             : {}\n".format(self.ip)
        _s += "  port           : {}\n".format(self.port)
        _s += "  dbname         : {}\n".format(self.dbname)
        _s += "  username       : {}\n".format(self.username)
        _s += "  password       : {}\n".format("*******")
        _s += "  project_key : {}\n".format(self.project_key)
        self._s = _s
        print(self._s)

    def __str__(self):
        """ Print information of databset. """
        return self._s

    def _fill_project_info(self, args):
        """ Fill in project_key for all studies, architectures and parameters. """
        return args.update({'project_key': self.project_key})

    @staticmethod
    def _serialization(ps):
        """ Seralize data. """
        return pickle.dumps(ps, protocol=pickle.HIGHEST_PROTOCOL)#protocol=2)
        # with open('_temp.pkl', 'wb') as file:
        #     return pickle.dump(ps, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _deserialization(ps):
        """ Deseralize data. """
        return pickle.loads(ps)

    ## =========================== MODELS ================================ ##
    def save_model(self, network=None, **kwargs):  #args=None):
        """Save model archietcture and parameters into database, timestamp will be added automatically.

        Parameters
        ----------
        network : TensorLayer layer
            TensorLayer layer instance.
        kwargs : other events
            Other events, such as name, accuracy, loss, step number and etc (optinal).

        Examples
        ---------
        - Save model architecture and parameters into database.
        >>> db.save_model(net, accuray=0.8, loss=2.3, name='second_model')

        - Load one model with parameters from database (run this in other script)
        >>> net = db.find_one_model(sess=sess, accuray=0.8, loss=2.3)

        - Find and load the latest model.
        >>> net = db.find_one_model(sess=sess, sort=[("time", pymongo.DESCENDING)])
        >>> net = db.find_one_model(sess=sess, sort=[("time", -1)])

        - Find and load the oldest model.
        >>> net = db.find_one_model(sess=sess, sort=[("time", pymongo.ASCENDING)])
        >>> net = db.find_one_model(sess=sess, sort=[("time", 1)])

        Returns
        ---------
        boolean : True for success, False for fail.
        """
        self._fill_project_info(kwargs)# put project_key into kwargs

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
            print(e)
            print("[Database] Save model: FAIL")
            return False

    def find_one_model(self, sess, sort=None, **kwargs):
        """Finds and returns a model archietcture and its parameters from the database which matches the requirement.

        Parameters
        ----------
        sess : Session
            TensorFlow session.
        sort : List of tuple
            PyMongo sort comment, search "PyMongo find one sorting" and `collection level operations <http://api.mongodb.com/python/current/api/pymongo/collection.html>`__ for more details.
        kwargs : other events
            Other events, such as name, accuracy, loss, step number and etc (optinal).

        Examples
        ---------
        - see ``save_model``.

        Returns
        ---------
        network : TensorLayer layer
        """
        self._fill_project_info(kwargs)
        # if dataset_key is None:
            # raise Exception("dataset_key is None, please give a dataset name")
        # kwargs.update({'dataset_key': dataset_key})

        s = time.time()

        d = self.db.Model.find_one(filter=kwargs, sort=sort)

        if d is not None:
            params_id = d['params_id']
            graphs = d['architecture']
            _datetime = d['time']
            exists_or_mkdir('__ztemp', False)
            with open(os.path.join('__ztemp', 'graph.pkl'), 'wb') as file:
                pickle.dump(graphs, file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("[Database] FAIL! Cannot find model: {}".format(kwargs))
            return False
        try:
            params = self._deserialization(self.model_fs.get(params_id).read())
            np.savez(os.path.join('__ztemp', 'params.npz'), params=params)

            network = load_graph_and_params(name='__ztemp', sess=sess)
            del_folder('__ztemp')

            pc = self.db.Model.find(kwargs)
            print("[Database] Find one model SUCCESS. kwargs:{} sort:{} save time:{} took: {}s".format(kwargs, sort, _datetime, round(time.time() - s, 2)))

            # put all informations of model into the TL layer
            for key in d:
                network.__dict__.update({"_%s"%key : d[key]})

            # check whether more parameters match the requirement
            params_id_list = pc.distinct('params_id')
            n_params = len(params_id_list)
            if n_params != 1:
                print("     Note that there are {} models match the requirement".format(n_params))
            return network
        except Exception as e:
            print(e)
            return False

    def del_model(self, **kwargs):
        """Delete model.

        Parameters
        -----------
        kwargs : logging information
            Find items to delete, leave it empty to delete all log.
        """
        self._fill_project_info(kwargs)
        self.db.Model.delete_many(kwargs)
        logging.info("[Database] Delete Model SUCCESS")

    ## =========================== DATASET =============================== ##
    def save_dataset(self, dataset=None, dataset_key=None, **kwargs):
        """Saves one dataset into database, timestamp will be added automatically.

        Parameters
        ----------
        dataset : any type
            The dataset you want to store.
        dataset_key : str
            The name/key of dataset.
        kwargs : other events
            Other events, such as description, author and etc (optinal).

        Examples
        ----------
        - Save dataset
        >>> db.save_dataset([X_train, y_train, X_test, y_test], 'mnist', description='this is a tutorial')
        - Get dataset
        >>> dataset = db.find_one_dataset('mnist')

        Returns
        ---------
        boolean : Return True if save success, otherwise, return False.
        """
        self._fill_project_info(kwargs)
        if dataset_key is None:
            raise Exception("dataset_key is None, please give a dataset name")
        kwargs.update({'dataset_key': dataset_key})

        s = time.time()
        try:
            dataset_id = self.dataset_fs.put(self._serialization(dataset))
            kwargs.update({'dataset_id': dataset_id, 'time': datetime.utcnow()})
            self.db.Dataset.insert_one(kwargs)
            # print("[Database] Save params: {} SUCCESS, took: {}s".format(file_name, round(time.time()-s, 2)))
            print("[Database] Save dataset: SUCCESS, took: {}s".format(round(time.time() - s, 2)))
            return True
        except Exception as e:
            print(e)
            print("[Database] Save dataset: FAIL")
            return False

    def find_one_dataset(self, dataset_key=None, sort=None, **kwargs):
        """Finds and returns a dataset from the database which matches the requirement.

        Parameters
        ----------
        dataset_key : str
            The name/key of dataset.
        sort : List of tuple
            PyMongo sort comment, search "PyMongo find one sorting" and `collection level operations <http://api.mongodb.com/python/current/api/pymongo/collection.html>`__ for more details.
        kwargs : other events
            Other events, such as description, author and etc (optinal).

        Examples
        ---------
        - Save dataset
        >>> db.save_dataset([X_train, y_train, X_test, y_test], 'mnist', description='this is a tutorial')
        - Get dataset
        >>> dataset = db.find_one_dataset('mnist')
        >>> datasets = db.find_all_datasets('mnist')

        Returns
        --------
        dataset : the dataset or False
            Return False if nothing found.
        """
        self._fill_project_info(kwargs)
        if dataset_key is None:
            raise Exception("dataset_key is None, please give a dataset name")
        kwargs.update({'dataset_key': dataset_key})

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
        except Exception:
            return False

    def find_all_datasets(self, dataset_key=None, **kwargs):
        """Finds and returns all datasets from the database which matches the requirement.

        Parameters
        ----------
        dataset_key : str
            The name/key of dataset.
        kwargs : other events
            Other events, such as description, author and etc (optinal).

        Returns
        --------
        params : the parameters, return False if nothing found.
        """
        self._fill_project_info(kwargs)
        if dataset_key is None:
            raise Exception("dataset_key is None, please give a dataset name")
        kwargs.update({'dataset_key': dataset_key})

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

    def del_dataset(self, **kwargs):
        """Delete datasets.

        Parameters
        -----------
        kwargs : logging information
            Find items to delete, leave it empty to delete all log.
        """
        self._fill_project_info(kwargs)
        self.db.Dataset.delete_many(kwargs)
        logging.info("[Database] Delete Dataset SUCCESS")

    ## =========================== LOGGING =============================== ##
    def train_log(self, **kwargs):
        """Saves the training log, timestamp will be added automatically.

        Parameters
        -----------
        kwargs : logging information
            Events, such as accuracy, loss, step number and etc.

        Examples
        ---------
        >>> db.train_log(accuray=0.33, loss=0.98)
        """
        self._fill_project_info(kwargs)
        kwargs.update({'time': datetime.utcnow()})
        _result = self.db.TrainLog.insert_one(kwargs)
        _log = self._print_dict(kwargs)
        logging.info("[Database] train log: " + _log)

    def valid_log(self, **kwargs):
        """Saves the validation log, timestamp will be added automatically.

        Parameters
        -----------
        kwargs : logging information
            Events, such as accuracy, loss, step number and etc.

        Examples
        ---------
        >>> db.valid_log(accuray=0.33, loss=0.98)
        """
        self._fill_project_info(kwargs)
        kwargs.update({'time': datetime.utcnow()})
        _result = self.db.ValidLog.insert_one(kwargs)
        _log = self._print_dict(kwargs)
        logging.info("[Database] valid log: " + _log)

    def test_log(self, **kwargs):
        """Saves the testing log, timestamp will be added automatically.

        Parameters
        -----------
        kwargs : logging information
            Events, such as accuracy, loss, step number and etc.

        Examples
        ---------
        >>> db.test_log(accuray=0.33, loss=0.98)
        """
        self._fill_project_info(kwargs)
        kwargs.update({'time': datetime.utcnow()})
        _result = self.db.TestLog.insert_one(kwargs)
        _log = self._print_dict(kwargs)
        logging.info("[Database] test log: " + _log)

    def del_train_log(self, **kwargs):
        """Deletes training log.

        Parameters
        -----------
        kwargs : logging information
            Find items to delete, leave it empty to delete all log.

        Examples
        ---------
        - Save training log
        >>> db.train_log(accuray=0.33)
        >>> db.train_log(accuray=0.44)

        - Delete logs that match the requirement
        >>> db.del_train_log(accuray=0.33)

        - Delete all logs
        >>> db.del_train_log()
        """
        self._fill_project_info(kwargs)
        self.db.TrainLog.delete_many(kwargs)
        logging.info("[Database] Delete TrainLog SUCCESS")

    def del_valid_log(self, **kwargs):
        """Deletes validation log.

        Parameters
        -----------
        kwargs : logging information
            Find items to delete, leave it empty to delete all log.

        Examples
        ---------
        - see ``train_log``.
        """
        self._fill_project_info(kwargs)
        self.db.ValidLog.delete_many(kwargs)
        logging.info("[Database] Delete ValidLog SUCCESS")

    def del_test_log(self, **kwargs):
        """Deletes testing log.

        Parameters
        -----------
        kwargs : logging information
            Find items to delete, leave it empty to delete all log.

        Examples
        ---------
        - see ``train_log``.
        """
        self._fill_project_info(kwargs)
        self.db.TestLog.delete_many(kwargs)
        logging.info("[Database] Delete TestLog SUCCESS")

    ## =========================== JOB =================================== ##
    def push_task(self, task_key=None, script=None, hyper_parameters=None, result_key=None, **kwargs):
        """Uploads a task to the database, timestamp will be added automatically.

        Parameters
        -----------
        task_key : str
            The task name.
        script : str
            File name of the python script.
        hyper_parameters : dictionary
            The hyper parameters pass into the script.
        kwargs : other parameters
            Users customized parameters such as description, version number.

        Examples
        -----------
        - Uploads a task
        >>> db.push_task(task_key='mnist', script='example/tutorial_mnist_simple.py', description='simple tutorial')

        - Finds and runs the latest task
        >>> db.run_one_task(sess=sess, sort=[("time", pymongo.DESCENDING)])
        >>> db.run_one_task(sess=sess, sort=[("time", -1)])

        - Finds and runs the oldest task
        >>> db.run_one_task(sess=sess, sort=[("time", pymongo.ASCENDING)])
        >>> db.run_one_task(sess=sess, sort=[("time", 1)])

        """
        if not isinstance(task_key, str):# is None:
            raise Exception("task_key should be string")
        if not isinstance(script, str):# is None:
            raise Exception("script should be string")
        if hyper_parameters is None:
            hyper_parameters = {}
        if result_key is None:
            result_key = []

        self._fill_project_info(kwargs)
        kwargs.update({'time': datetime.utcnow()})
        kwargs.update({'hyper_parameters': hyper_parameters})
        kwargs.update({'result_key': result_key})

        _script = open(script, 'rb').read()

        kwargs.update({'status': 'pending', 'script': _script, 'result': {}})
        self.db.Task.insert_one(kwargs)
        logging.info("[Database] Saved Task: {} / {}".format(task_key, script))


    def run_one_task(self, task_key=None, sort=None, **kwargs):
        """Finds and runs a pending task.

        Parameters
        -----------
        task_key : str
            The task name.
        sort : List of tuple
            PyMongo sort comment, search "PyMongo find one sorting" and `collection level operations <http://api.mongodb.com/python/current/api/pymongo/collection.html>`__ for more details.
        kwargs : other parameters
            Users customized parameters such as description, version number.

        Examples
        ---------
        - see ``push_task``

        - Servers wait task
        >>> while True:
        >>>     db.run_one_task(task_key='mnist')
        >>>     time.sleep(1)

        Returns
        --------
        boolean : True for success, False for fail.
        """
        if not isinstance(task_key, str):# is None:
            raise Exception("task_key should be string")
        self._fill_project_info(kwargs)
        kwargs.update({'status': 'pending'})

        ## find task and set status to running
        # task = self.db.Task.find_one(kwargs)
        task = self.db.Task.find_one_and_update(kwargs, {'$set': {'status': 'running'}}, sort=sort)#, return_document=ReturnDocument.AFTER)

        try:
            ## get task info e.g. hyper parameters, python script
            if task is None:
                logging.info("[Database] Find Task FAIL: key: {} sort: {}".format(task_key, sort))
                return False
            else:
                logging.info("[Database] Find Task SUCCESS: key: {} sort: {}".format(task_key, sort))
            _datetime = task['time']
            _script = task['script']
            _id = task['_id']
            _hyper_parameters = task['hyper_parameters']
            _result_key = task['result_key']
            logging.info("  hyper parameters:")
            for key in _hyper_parameters:
                globals()[key] = _hyper_parameters[key]
                logging.info("    {}: {}".format(key, _hyper_parameters[key]))
            ## run task
            # f = open('_ztemp.py', 'wb')
            # f.write(task['script'])
            # f.close()
            s = time.time()
            logging.info("[Database] Start Task: key: {} sort: {} push time: {}".format(task_key, sort, _datetime))
            # os.system("python __ztemp.py")
            # import _ztemp
            _script = _script.decode('utf-8')
            with tf.Graph().as_default() as graph: # clear all TF graphs
                exec(_script, globals())
            # os.remove("_ztemp.py")

            ## set status to finished
            _ = self.db.Task.find_one_and_update({'_id': _id}, {'$set': {'status': 'finished'}})

            ## return results
            __result = {}
            for _key in _result_key:
                logging.info("  result: {}={} {}".format(_key, globals()[_key], type(globals()[_key])))
                # print(type(_key), type(globals()[_key]))
                # __result__.update({_key, globals()[str(_key)]})
                __result.update({"%s" % _key :globals()[_key]})
            # print(__result, type(globals()[_key]))
            _ = self.db.Task.find_one_and_update({'_id': _id}, {'$set': {'result': __result}}, return_document=pymongo.ReturnDocument.AFTER)
            # print(_['result'])
            logging.info("[Database] Finished Task: key: {} sort: {} push time: {} took: {}s".format(task_key, sort, _datetime, time.time()-s))
            # exit()
            return True
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.info("{}  {}  {}  {}".format(exc_type, fname, exc_tb.tb_lineno, e))
            logging.info("[Database] Fail to run task")
            ## if fail, set status back to pending
            _ = self.db.Task.find_one_and_update({'_id': _id}, {'$set': {'status': 'pending'}})
            return False

    def del_task(self, **kwargs):
        """Delete tasks.

        Parameters
        -----------
        kwargs : logging information
            Find items to delete, leave it empty to delete all log.

        Examples
        ---------
        >>> db.del_task()
        """
        self._fill_project_info(kwargs)
        self.db.Task.delete_many(kwargs)
        logging.info("[Database] Delete Task SUCCESS")

    def check_unfinished_task(self, task_key=None, **kwargs):
        """Finds and runs a pending task.

        Parameters
        -----------
        task_key : str
            The task name.
        kwargs : other parameters
            Users customized parameters such as description, version number.

        Examples
        ---------
        - Wait until all tasks finish in user's local console
        >>> while not db.check_unfinished_task():
        >>>     time.sleep(1)
        >>> ... get results from database ...

        Returns
        --------
        boolean : True for success, False for fail.
        """
        if not isinstance(task_key, str):# is None:
            raise Exception("task_key should be string")
        self._fill_project_info(kwargs)

        kwargs.update({'$or': [ { 'status': 'pending' }, { 'status': 'running' }]})

        ## find task
        task = self.db.Task.find_one(kwargs)
        # print(task)
        if task is None:
            logging.info("[Database] No unfinished task: key: {}".format(task_key))
            return False
        else:
            logging.info("[Database] Find unfinished task: key: {}".format(task_key))
            return True

    @staticmethod
    def _print_dict(args):
        # return " / ".join(str(key) + ": "+ str(value) for key, value in args.items())
        string = ''
        for key, value in args.items():
            if key is not '_id':
                string += str(key) + ": " + str(value) + " / "
        return string


def AutoFill(func):

    def func_wrapper(self, *args, **kwargs):
        d = inspect.getcallargs(func, self, *args, **kwargs)
        d['args'].update({"studyID": self.studyID})
        return func(**d)

    return func_wrapper


class TensorDB(object):
    """TensorDB is a MongoDB based manager that help you to manage data, network topology, parameters and logging.

    Parameters
    -------------
    ip : str
        Localhost or IP address.
    port : int
        Port number.
    db_name : str
        Database name.
    user_name : str
        User name. Set to None if it donnot need authentication.
    password : str
        Password

    Attributes
    ------------
    db : ``pymongo.MongoClient[db_name]``, xxxxxx
    datafs : ``gridfs.GridFS(self.db, collection="datafs")``, xxxxxxxxxx
    modelfs : ``gridfs.GridFS(self.db, collection="modelfs")``,
    paramsfs : ``gridfs.GridFS(self.db, collection="paramsfs")``,
    db.Params : Collection for
    db.TrainLog : Collection for
    db.ValidLog : Collection for
    db.TestLog : Collection for
    studyID : string, unique ID, if None random generate one.

    Notes
    -------------
    - MongoDB, as TensorDB is based on MongoDB, you need to install it in your local machine or remote machine.
    - pip install pymongo, for MongoDB python API.
    - You may like to install MongoChef or Mongo Management Studo APP for visualizing or testing your MongoDB.
    """

    def __init__(
            self, ip='localhost', port=27017, db_name='db_name', user_name=None, password='password', studyID=None
    ):
        ## connect mongodb
        client = pymongo.MongoClient(ip, port)
        self.db = client[db_name]
        if user_name != None:
            self.db.authenticate(user_name, password)

        if studyID is None:
            self.studyID = str(uuid.uuid1())
        else:
            self.studyID = studyID

        ## define file system (Buckets)
        self.datafs = gridfs.GridFS(self.db, collection="datafs")
        self.modelfs = gridfs.GridFS(self.db, collection="modelfs")
        self.paramsfs = gridfs.GridFS(self.db, collection="paramsfs")
        self.archfs = gridfs.GridFS(self.db, collection="ModelArchitecture")
        ##
        print("[TensorDB] Connect SUCCESS {}:{} {} {} {}".format(ip, port, db_name, user_name, studyID))

        self.ip = ip
        self.port = port
        self.db_name = db_name
        self.user_name = user_name

    @classmethod
    def __autofill(self, args):
        return args.update({'studyID': self.studyID})

    @staticmethod
    def __serialization(ps):
        return pickle.dumps(ps, protocol=2)

    @staticmethod
    def __deserialization(ps):
        return pickle.loads(ps)

    def save_params(self, params=None, args=None):  #, file_name='parameters'):
        """ Save parameters into MongoDB Buckets, and save the file ID into Params Collections.

        Parameters
        ----------
        params : a list of parameters
        args : dictionary, item meta data.

        Returns
        ---------
        f_id : the Buckets ID of the parameters.
        """
        if params is None:
            params = []
        if args is None:
            args = {}
        self.__autofill(args)
        s = time.time()
        f_id = self.paramsfs.put(self.__serialization(params))  #, file_name=file_name)
        args.update({'f_id': f_id, 'time': datetime.utcnow()})
        self.db.Params.insert_one(args)
        # print("[TensorDB] Save params: {} SUCCESS, took: {}s".format(file_name, round(time.time()-s, 2)))
        print("[TensorDB] Save params: SUCCESS, took: {}s".format(round(time.time() - s, 2)))
        return f_id

    @AutoFill
    def find_one_params(self, args=None, sort=None):
        """ Find one parameter from MongoDB Buckets.

        Parameters
        ----------
        args : dictionary
            For finding items.

        Returns
        --------
        params : the parameters, return False if nothing found.
        f_id : the Buckets ID of the parameters, return False if nothing found.
        """
        if args is None:
            args = {}
        s = time.time()
        # print(args)
        d = self.db.Params.find_one(filter=args, sort=sort)

        if d is not None:
            f_id = d['f_id']
        else:
            print("[TensorDB] FAIL! Cannot find: {}".format(args))
            return False, False
        try:
            params = self.__deserialization(self.paramsfs.get(f_id).read())
            print("[TensorDB] Find one params SUCCESS, {} took: {}s".format(args, round(time.time() - s, 2)))
            return params, f_id
        except Exception:
            return False, False

    @AutoFill
    def find_all_params(self, args=None):
        """ Find all parameter from MongoDB Buckets

        Parameters
        ----------
        args : dictionary, find items

        Returns
        --------
        params : the parameters, return False if nothing found.

        """
        if args is None:
            args = {}
        s = time.time()
        pc = self.db.Params.find(args)

        if pc is not None:
            f_id_list = pc.distinct('f_id')
            params = []
            for f_id in f_id_list:  # you may have multiple Buckets files
                tmp = self.paramsfs.get(f_id).read()
                params.append(self.__deserialization(tmp))
        else:
            print("[TensorDB] FAIL! Cannot find any: {}".format(args))
            return False

        print("[TensorDB] Find all params SUCCESS, took: {}s".format(round(time.time() - s, 2)))
        return params

    @AutoFill
    def del_params(self, args=None):
        """ Delete params in MongoDB uckets.

        Parameters
        -----------
        args : dictionary, find items to delete, leave it empty to delete all parameters.
        """
        if args is None:
            args = {}
        pc = self.db.Params.find(args)
        f_id_list = pc.distinct('f_id')
        # remove from Buckets
        for f in f_id_list:
            self.paramsfs.delete(f)
        # remove from Collections
        self.db.Params.remove(args)

        print("[TensorDB] Delete params SUCCESS: {}".format(args))

    @staticmethod
    def _print_dict(args):
        # return " / ".join(str(key) + ": "+ str(value) for key, value in args.items())

        string = ''
        for key, value in args.items():
            if key is not '_id':
                string += str(key) + ": " + str(value) + " / "
        return string

    ## =========================== LOG =================================== ##
    @AutoFill
    def train_log(self, args=None):
        """Save the training log.

        Parameters
        -----------
        args : dictionary, items to save.

        Examples
        ---------
        >>> db.train_log(time=time.time(), {'loss': loss, 'acc': acc})
        """
        if args is None:
            args = {}
        _result = self.db.TrainLog.insert_one(args)
        _log = self._print_dict(args)
        #print("[TensorDB] TrainLog: " +_log)
        return _result

    @AutoFill
    def del_train_log(self, args=None):
        """ Delete train log.

        Parameters
        -----------
        args : dictionary, find items to delete, leave it empty to delete all log.
        """
        if args is None:
            args = {}
        self.db.TrainLog.delete_many(args)
        print("[TensorDB] Delete TrainLog SUCCESS")

    @AutoFill
    def valid_log(self, args=None):
        """Save the validating log.

        Parameters
        -----------
        args : dictionary, items to save.

        Examples
        ---------
        >>> db.valid_log(time=time.time(), {'loss': loss, 'acc': acc})
        """
        if args is None:
            args = {}
        _result = self.db.ValidLog.insert_one(args)
        # _log = "".join(str(key) + ": " + str(value) for key, value in args.items())
        _log = self._print_dict(args)
        print("[TensorDB] ValidLog: " + _log)
        return _result

    @AutoFill
    def del_valid_log(self, args=None):
        """ Delete validation log.

        Parameters
        -----------
        args : dictionary, find items to delete, leave it empty to delete all log.
        """
        if args is None:
            args = {}
        self.db.ValidLog.delete_many(args)
        print("[TensorDB] Delete ValidLog SUCCESS")

    @AutoFill
    def test_log(self, args=None):
        """Save the testing log.

        Parameters
        -----------
        args : dictionary, items to save.

        Examples
        ---------
        >>> db.test_log(time=time.time(), {'loss': loss, 'acc': acc})
        """
        if args is None:
            args = {}
        _result = self.db.TestLog.insert_one(args)
        # _log = "".join(str(key) + str(value) for key, value in args.items())
        _log = self._print_dict(args)
        print("[TensorDB] TestLog: " + _log)
        return _result

    @AutoFill
    def del_test_log(self, args=None):
        """ Delete test log.

        Parameters
        -----------
        args : dictionary, find items to delete, leave it empty to delete all log.
        """
        if args is None:
            args = {}

        self.db.TestLog.delete_many(args)
        print("[TensorDB] Delete TestLog SUCCESS")

    # =========================== Network Architecture ================== ##
    @AutoFill
    def save_model_architecture(self, s, args=None):
        if args is None:
            args = {}

        self.__autofill(args)
        fid = self.archfs.put(s, filename="modelarchitecture")
        args.update({"fid": fid})
        self.db.march.insert_one(args)

    @AutoFill
    def load_model_architecture(self, args=None):

        if args is None:
            args = {}

        d = self.db.march.find_one(args)
        if d is not None:
            fid = d['fid']
            print(d)
            print(fid)
            # "print find"
        else:
            print("[TensorDB] FAIL! Cannot find: {}".format(args))
            print("no idtem")
            return False, False
        try:
            archs = self.archfs.get(fid).read()
            return archs, fid
        except Exception as e:
            print("exception")
            print(e)
            return False, False

    @AutoFill
    def save_job(self, script=None, args=None):
        """Save the job.

        Parameters
        -----------
        script : a script file name or None.
        args : dictionary, items to save.

        Examples
        ---------
        >>> # Save your job
        >>> db.save_job('your_script.py', {'job_id': 1, 'learning_rate': 0.01, 'n_units': 100})
        >>> # Run your job
        >>> temp = db.find_one_job(args={'job_id': 1})
        >>> print(temp['learning_rate'])
        0.01
        >>> import _your_script
        running your script
        """

        if args is None:
            args = {}

        self.__autofill(args)
        if script is not None:
            _script = open(script, 'rb').read()
            args.update({'script': _script, 'script_name': script})
        # _result = self.db.Job.insert_one(args)
        _result = self.db.Job.replace_one(args, args, upsert=True)
        _log = self._print_dict(args)
        print("[TensorDB] Save Job: script={}, args={}".format(script, args))
        return _result

    @AutoFill
    def find_one_job(self, args=None):
        """ Find one job from MongoDB Job Collections.

        Parameters
        ----------
        args : dictionary, find items.

        Returns
        --------
        dictionary : contains all meta data and script.
        """

        if args is None:
            args = {}

        temp = self.db.Job.find_one(args)

        if temp is not None:
            if 'script_name' in temp.keys():
                f = open('_' + temp['script_name'], 'wb')
                f.write(temp['script'])
                f.close()
            print("[TensorDB] Find Job: {}".format(args))
        else:
            print("[TensorDB] FAIL! Cannot find any: {}".format(args))
            return False

        return temp

    def push_job(self, margs, wargs, dargs, epoch):

        _ms, mid = self.load_model_architecture(margs)
        _weight, wid = self.find_one_params(wargs)
        args = {
            "weight": wid,
            "model": mid,
            "dargs": dargs,
            "epoch": epoch,
            "time": datetime.utcnow(),
            "Running": False
        }
        self.__autofill(args)
        self.db.JOBS.insert_one(args)

    def peek_job(self):
        args = {'Running': False}
        self.__autofill(args)
        m = self.db.JOBS.find_one(args)
        print(m)
        if m is None:
            return False

        s = self.paramsfs.get(m['weight']).read()
        w = self.__deserialization(s)

        ach = self.archfs.get(m['model']).read()

        return m['_id'], ach, w, m["dargs"], m['epoch']

    def run_job(self, jid):
        self.db.JOBS.find_one_and_update({'_id': jid}, {'$set': {'Running': True, "Since": datetime.utcnow()}})

    def del_job(self, jid):
        self.db.JOBS.find_one_and_update({'_id': jid}, {'$set': {'Running': True, "Finished": datetime.utcnow()}})

    def __str__(self):
        _s = "[TensorDB] Info:\n"
        _t = _s + "    " + str(self.db)
        return _t
