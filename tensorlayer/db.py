#! /usr/bin/python
# -*- coding: utf-8 -*-

import inspect
import pickle
import time
import uuid
import os
from datetime import datetime

import gridfs
import pymongo
from tensorlayer.files import load_graph_and_params, exists_or_mkdir
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
    experiment_key : str or None
        Experiment key for this project, similar with the repository name of Github.

    Attributes
    ------------
    ip, port, dbname and other input parameters : see above
        See above.
    experiment_key : str
        The given study ID, if no given, set to the script name.
    db : mongodb client
        See ``pymongo.MongoClient``.
    """

    # @deprecated_alias(db_name='dbname', user_name='username', end_support_version=2.1)
    def __init__(
            self, ip='localhost', port=27017, dbname='dbname', username=None, password='password', experiment_key=None
    ):
        self.ip = ip
        self.port = port
        self.dbname = dbname
        self.username = username

        print("[TensorDB] Initializing ...")
        ## connect mongodb
        client = pymongo.MongoClient(ip, port)
        self.db = client[dbname]
        if username != None:
            self.db.authenticate(username, password)
        else:
            print("[TensorDB] No username given, it works if authentication is not required")
        if experiment_key is None:
            self.experiment_key = sys.argv[0].split('.')[0]
            print("[TensorDB] No experiment_key given, use {}".format(self.experiment_key))
        else:
            self.experiment_key = experiment_key

        ## define file system (Buckets)
        self.dataset_fs = gridfs.GridFS(self.db, collection="datasetFilesystem")
        self.model_fs = gridfs.GridFS(self.db, collection="modelfs")
        # self.params_fs = gridfs.GridFS(self.db, collection="parametersFilesystem")
        # self.architecture_fs = gridfs.GridFS(self.db, collection="architectureFilesystem")

        ## print info
        print("[TensorDB] Connected ")
        _s = "[TensorDB] Info:\n"
        _s += "  ip             : {}\n".format(self.ip)
        _s += "  port           : {}\n".format(self.port)
        _s += "  dbname         : {}\n".format(self.dbname)
        _s += "  username       : {}\n".format(self.username)
        _s += "  password       : {}\n".format("*******")
        _s += "  experiment_key : {}\n".format(self.experiment_key)
        self._s = _s
        print(self._s)

    def __str__(self):
        """ Print information of databset. """
        return self._s

    def _fill_experiment_info(self, args):
        """ Fill in experiment_key for all studies, architectures and parameters. """
        return args.update({'experimentKey': self.experiment_key})

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

    def save_model(self, network=None, **kwargs):  #args=None):
        """Save model archietcture and parameters into database.

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

        Returns
        ---------
        boolean : True for success, False for fail.
        """
        params = network.get_all_params()

        self._fill_experiment_info(kwargs)  # put experimentKey into kwargs
        s = time.time()

        kwargs.update({'architecture': network.all_graphs})

        try:
            params_id = self.model_fs.put(self._serialization(params))
            kwargs.update({'params_id': params_id, 'time': datetime.utcnow()})
            self.db.Model.insert_one(kwargs)
            print("[TensorDB] Save model: SUCCESS, took: {}s".format(round(time.time() - s, 2)))
            return True
        except Exception as e:
            print(e)
            print("[TensorDB] Save model: FAIL")
            return False

    def find_one_model(self, sess, sort=None, **kwargs):
        """Returns one model archietcture and parameters from database that match with the requirement.

        Parameters
        ----------
        sess : Session
            TensorFlow session.
        sort : XX
            XXX
        kwargs : other events
            Other events, such as name, accuracy, loss, step number and etc (optinal).

        Examples
        ---------
        - see ``save_model``.

        Returns
        ---------
        network : TensorLayer layer
        """
        # if dataset_key is None:
            # raise Exception("dataset_key is None, please give a dataset name")
        # kwargs.update({'datasetKey': dataset_key})

        s = time.time()

        d = self.db.Model.find_one(filter=kwargs, sort=sort)

        if d is not None:
            params_id = d['params_id']
            graphs = d['architecture']
            # print(graphs)
            exists_or_mkdir('__ztemp', False)
            with open(os.path.join('__ztemp', 'graph.pkl'), 'wb') as file:
                pickle.dump(graphs, file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("[TensorDB] FAIL! Cannot find model: {}".format(kwargs))
            return False
        try:
            params = self._deserialization(self.model_fs.get(params_id).read())
            # print(params)
            np.savez(os.path.join('__ztemp', 'params.npz'), params=params)

            network = load_graph_and_params(name='__ztemp', sess=sess)

            pc = self.db.Model.find(kwargs)
            print("[TensorDB] Find one model SUCCESS, {} took: {}s".format(kwargs, round(time.time() - s, 2)))

            # check whether more parameters match the requirement
            params_id_list = pc.distinct('params_id')
            n_params = len(params_id_list)
            if n_params != 1:
                print("     Note that there are {} models match the requirement".format(n_params))
            return network
        except Exception as e:
            print(e)
            return False

    def save_dataset(self, dataset=None, dataset_key=None, **kwargs):
        """Saves one dataset into database.

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
        if dataset_key is None:
            raise Exception("dataset_key is None, please give a dataset name")
        kwargs.update({'datasetKey': dataset_key})

        # self._fill_experiment_info(kwargs)

        s = time.time()
        try:
            dataset_id = self.dataset_fs.put(self._serialization(dataset))
            kwargs.update({'dataset_id': dataset_id, 'time': datetime.utcnow()})
            self.db.Dataset.insert_one(kwargs)
            # print("[TensorDB] Save params: {} SUCCESS, took: {}s".format(file_name, round(time.time()-s, 2)))
            print("[TensorDB] Save dataset: SUCCESS, took: {}s".format(round(time.time() - s, 2)))
            return True
        except Exception as e:
            print(e)
            print("[TensorDB] Save dataset: FAIL")
            return False

    def find_one_dataset(self, dataset_key=None, sort=None, **kwargs):
        """Returns one dataset from database that match with the requirement.

        Parameters
        ----------
        dataset_key : str
            The name/key of dataset.
        sort : XX
            see mongodb
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
        if dataset_key is None:
            raise Exception("dataset_key is None, please give a dataset name")
        kwargs.update({'datasetKey': dataset_key})

        s = time.time()

        d = self.db.Dataset.find_one(filter=kwargs, sort=sort)

        if d is not None:
            dataset_id = d['dataset_id']
        else:
            print("[TensorDB] FAIL! Cannot find dataset: {}".format(kwargs))
            return False
        try:
            dataset = self._deserialization(self.dataset_fs.get(dataset_id).read())
            pc = self.db.Dataset.find(kwargs)
            print("[TensorDB] Find one dataset SUCCESS, {} took: {}s".format(kwargs, round(time.time() - s, 2)))

            # check whether more datasets match the requirement
            dataset_id_list = pc.distinct('dataset_id')
            n_dataset = len(dataset_id_list)
            if n_dataset != 1:
                print("     Note that there are {} datasets match the requirement".format(n_dataset))
            return dataset
        except Exception:
            return False

    def find_all_datasets(self, dataset_key=None, **kwargs):
        """Returns all datasets from database that match with the requirement.

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
        if dataset_key is None:
            raise Exception("dataset_key is None, please give a dataset name")
        kwargs.update({'datasetKey': dataset_key})

        s = time.time()
        pc = self.db.Dataset.find(kwargs)

        if pc is not None:
            dataset_id_list = pc.distinct('dataset_id')
            dataset_list = []
            for dataset_id in dataset_id_list:  # you may have multiple Buckets files
                tmp = self.dataset_fs.get(dataset_id).read()
                dataset_list.append(self._deserialization(tmp))
        else:
            print("[TensorDB] FAIL! Cannot find any dataset: {}".format(kwargs))
            return False

        print("[TensorDB] Find {} datasets SUCCESS, took: {}s".format(len(dataset_list), round(time.time() - s, 2)))
        return dataset_list




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
