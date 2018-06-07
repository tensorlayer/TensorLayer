#! /usr/bin/python
# -*- coding: utf-8 -*-

import inspect


def private_method(func):
    """decorator for making an instance method private"""

    def func_wrapper(*args, **kwargs):
        """decorator wrapper function"""
        outer_frame = inspect.stack()[1][0]
        if 'self' not in outer_frame.f_locals or outer_frame.f_locals['self'] is not args[0]:
            raise RuntimeError('%s.%s is a private method' % (args[0].__class__.__name__, func.__name__))

        return func(*args, **kwargs)

    return func_wrapper


def protected_method(func):
    """decorator for making an instance method private"""

    def func_wrapper(*args, **kwargs):
        """decorator wrapper function"""
        outer_frame = inspect.stack()[1][0]

        caller = inspect.getmro(outer_frame.f_locals['self'].__class__)[:-1]
        target = inspect.getmro(args[0].__class__)[:-1]

        share_subsclass = False

        for cls_ in target:
            if issubclass(caller[0], cls_) or caller[0] is cls_:
                share_subsclass = True
                break

        if ('self' not in outer_frame.f_locals or
                outer_frame.f_locals['self'] is not args[0]) and (not share_subsclass):
            raise RuntimeError('%s.%s is a protected method' % (args[0].__class__.__name__, func.__name__))

        return func(*args, **kwargs)

    return func_wrapper
