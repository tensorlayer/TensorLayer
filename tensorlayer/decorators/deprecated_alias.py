#! /usr/bin/python
# -*- coding: utf-8 -*-

import functools
import warnings

from tensorlayer import logging


def deprecated_alias(end_support_version, **aliases):

    def deco(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):

            try:
                func_name = "{}.{}".format(args[0].__class__.__name__, f.__name__)
            except (NameError, IndexError):
                func_name = f.__name__

            rename_kwargs(kwargs, aliases, end_support_version, func_name)

            return f(*args, **kwargs)

        return wrapper

    return deco


def rename_kwargs(kwargs, aliases, end_support_version, func_name):

    for alias, new in aliases.items():

        if alias in kwargs:

            if new in kwargs:
                raise TypeError('{}() received both {} and {}'.format(func_name, alias, new))

            warnings.warn('{}() - {} is deprecated; use {}'.format(func_name, alias, new), DeprecationWarning)
            logging.warning(
                "DeprecationWarning: {}(): "
                "`{}` argument is deprecated and will be removed in version {}, "
                "please change for `{}.`".format(func_name, alias, end_support_version, new)
            )
            kwargs[new] = kwargs.pop(alias)
