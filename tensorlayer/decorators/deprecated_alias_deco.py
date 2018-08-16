#! /usr/bin/python
# -*- coding: utf-8 -*-

import functools

from tensorlayer.decorators.utils import rename_kwargs

__all__ = ['deprecated_alias']


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
