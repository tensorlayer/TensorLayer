#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import tensorlayer as tl

import hyperdash as hd

__all__ = ["HyperDashHandler", "monitor", "Experiment", "IPythonMagicsWrapper"]


class HyperDashHandler(object):
    apikey = None

    @classmethod
    def reset_apikey(cls):
        cls.apikey = None

    @classmethod
    def set_apikey(cls, apikey):
        cls.apikey = apikey

    @classmethod
    def get_apikey(cls):

        if cls.apikey is None:
            raise ValueError(
                "Hyperdash API is not set.\n"
                "You can obtain your API Key using: `hyperdash login --email` or `hyperdash login --github`\n"
                "You should first call `HyperDashHandler.set_apikey('my_api_key')` in order to use `hyperdash`"
            )

        tl.logging.debug("Hyperdash API Key: %s" % cls.apikey)

        return cls.apikey

    @classmethod
    def monitor(cls, model_name, api_key=None, capture_io=True):

        if api_key is not None:
            cls.set_apikey(api_key)

        return hd.monitor(model_name, api_key_getter=cls.get_apikey, capture_io=capture_io)


class Experiment(hd.Experiment):

    def __init__(
            self,
            model_name,
            api_key=None,
            capture_io=True,
    ):

        if api_key is not None:
            HyperDashHandler.set_apikey(api_key)

        super(Experiment,
              self).__init__(model_name=model_name, api_key_getter=HyperDashHandler.get_apikey, capture_io=capture_io)


monitor = HyperDashHandler.monitor
IPythonMagicsWrapper = hd.IPythonMagicsWrapper
