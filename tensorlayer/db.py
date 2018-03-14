#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Experimental Database Management System.

Latest Version
"""

import inspect
import pickle
import time
import uuid
from datetime import datetime

import gridfs
from pymongo import MongoClient


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

    def __init__(self, ip='localhost', port=27017, db_name='db_name', user_name=None, password='password', studyID=None):
        ## connect mongodb
        client = MongoClient(ip, port)
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
        ... 0.01
        >>> import _your_script
        ... running your script
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
        args = {"weight": wid, "model": mid, "dargs": dargs, "epoch": epoch, "time": datetime.utcnow(), "Running": False}
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
