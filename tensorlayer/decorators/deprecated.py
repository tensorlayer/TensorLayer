#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import functools

from tensorflow.python.util import decorator_utils

from tensorflow.python.util.deprecation import _call_location
from tensorflow.python.util.deprecation import _validate_deprecation_args

from tensorlayer import tl_logging as logging

import wrapt

__all__ = ['deprecated']

# Allow deprecation warnings to be silenced temporarily with a context manager.
_PRINT_DEPRECATION_WARNINGS = True

# Remember which deprecation warnings have been printed already.
_PRINTED_WARNING = {}


def add_notice_to_docstring(doc, no_doc_str, notice):
    """Adds a deprecation notice to a docstring."""
    if not doc:
        lines = [no_doc_str]

    else:
        lines = decorator_utils._normalize_docstring(doc).splitlines()

    notice = [''] + notice

    if len(lines) > 1:
        # Make sure that we keep our distance from the main body
        if lines[1].strip():
            notice.append('')

        lines[1:1] = notice
    else:
        lines += notice

    return '\n'.join(lines)


def _add_deprecated_function_notice_to_docstring(doc, date, instructions):
    """Adds a deprecation notice to a docstring for deprecated functions."""

    if instructions:
        deprecation_message = """
            .. warning::
                **THIS FUNCTION IS DEPRECATED:** It will be removed after %s.
                *Instructions for updating:* %s.
        """ % (('in a future version' if date is None else ('after %s' % date)), instructions)

    else:
        deprecation_message = """
            .. warning::
                **THIS FUNCTION IS DEPRECATED:** It will be removed after %s.
        """ % (('in a future version' if date is None else ('after %s' % date)))

    main_text = [deprecation_message]

    return add_notice_to_docstring(doc, 'DEPRECATED FUNCTION', main_text)


def deprecated(wrapped=None, date='', instructions='', warn_once=True):

    if wrapped is None:
        return functools.partial(deprecated, date=date, instructions=instructions, warn_once=warn_once)

    @wrapt.decorator
    def deprecated_wrapper(wrapped, instance, args, kwargs):

        _validate_deprecation_args(date, instructions)

        if _PRINT_DEPRECATION_WARNINGS:

            class_or_func_name = decorator_utils.get_qualified_name(wrapped)

            if class_or_func_name not in _PRINTED_WARNING:
                if warn_once:
                    _PRINTED_WARNING[class_or_func_name] = True

                logging.warning(
                    'From %s: %s (from %s) is deprecated and will be removed %s.\n'
                    'Instructions for updating: %s\n' % (
                        _call_location(), class_or_func_name, wrapped.__module__, 'in a future version'
                        if date is None else ('after %s' % date), instructions
                    )
                )

        return wrapped(*args, **kwargs)

    decorated = deprecated_wrapper(wrapped)

    if (sys.version_info > (3, 0)):  # docstring can only be edited with Python 3
        wrapt.FunctionWrapper.__setattr__(
            decorated, "__doc__", _add_deprecated_function_notice_to_docstring(wrapped.__doc__, date, instructions)
        )

    return decorated
