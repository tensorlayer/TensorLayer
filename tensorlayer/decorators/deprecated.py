#! /usr/bin/python
# -*- coding: utf-8 -*-

import inspect
import sys
import functools

from tensorlayer.decorators.utils import add_deprecation_notice_to_docstring
from tensorlayer.decorators.utils import get_qualified_name
from tensorlayer.decorators.utils import validate_deprecation_args

import wrapt

__all__ = ['deprecated']

# Allow deprecation warnings to be silenced temporarily with a context manager.
_PRINT_DEPRECATION_WARNINGS = True

# Remember which deprecation warnings have been printed already.
_PRINTED_WARNING = {}


def deprecated(wrapped=None, date='', instructions='', warn_once=True):

    if wrapped is None:
        return functools.partial(deprecated, date=date, instructions=instructions, warn_once=warn_once)

    @wrapt.decorator
    def wrapper(wrapped, instance=None, args=None, kwargs=None):

        validate_deprecation_args(date, instructions)

        if _PRINT_DEPRECATION_WARNINGS:

            class_or_func_name = get_qualified_name(wrapped)

            if class_or_func_name not in _PRINTED_WARNING:
                if warn_once:
                    _PRINTED_WARNING[class_or_func_name] = True

                from tensorlayer import logging

                logging.warning(
                    '%s: `%s.%s` (in file: %s) is deprecated and will be removed %s.\n'
                    'Instructions for updating: %s\n' % (
                        "Class" if inspect.isclass(wrapped) else "Function", wrapped.__module__, class_or_func_name,
                        wrapped.__code__.co_filename, 'in a future version' if date is None else
                        ('after %s' % date), instructions
                    )
                )

        return wrapped(*args, **kwargs)

    decorated = wrapper(wrapped)

    if sys.version_info > (3, 0):  # docstring can only be edited with Python 3
        wrapt.FunctionWrapper.__setattr__(
            decorated, "__doc__", add_deprecation_notice_to_docstring(wrapped.__doc__, date, instructions)
        )

    return decorated
