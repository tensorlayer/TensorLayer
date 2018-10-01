#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
NOTE: DO NOT REMOVE THESE FILES. They are copied from Tensorflow repository and are necessary to build the library without installing TF.

Source: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/util

They replace the following imports:
>>> from tensorflow.python.util import decorator_utils
>>> from tensorflow.python.util.deprecation import _validate_deprecation_args
"""

import collections
import inspect
import re
import sys
import warnings

import tensorlayer as tl

from tensorflow.python.util import tf_inspect

__all__ = [
    "DeprecatedArgSpec", "add_deprecated_arg_notice_to_docstring", "add_deprecation_notice_to_docstring",
    "call_location", "get_qualified_name", "get_network_obj", "rename_kwargs", "validate_deprecation_args"
]

DeprecatedArgSpec = collections.namedtuple('DeprecatedArgSpec', ['position', 'has_ok_value', 'ok_value'])


def add_deprecated_arg_notice_to_docstring(doc, end_support_version, instructions):
    """Adds a deprecation notice to a docstring for deprecated arguments."""
    return _add_notice_to_docstring(
        doc=doc,
        instructions=instructions,
        no_doc_str='DEPRECATED FUNCTION ARGUMENTS',
        suffix_str='(deprecated arguments)',
        notice=[
            'SOME ARGUMENTS ARE DEPRECATED. They will be removed in TensorLayer version: %s.' % end_support_version,
            'Instructions for updating:'
        ]
    )

    # if instructions:
    #     deprecation_message = """
    #         .. warning::
    #             **THIS FUNCTION IS DEPRECATED:** It will be removed after %s.
    #             *Instructions for updating:* %s.
    #     """ % (('in a future version' if date is None else ('after %s' % date)), instructions)
    #
    # else:
    #     deprecation_message = """
    #         .. warning::
    #             **THIS FUNCTION IS DEPRECATED:** It will be removed after %s.
    #     """ % (('in a future version' if date is None else ('after %s' % date)))
    #
    # main_text = [deprecation_message]
    #
    # return _add_notice_to_docstring(doc=doc, instructions='DEPRECATED FUNCTION', notice=main_text)


def add_deprecation_notice_to_docstring(doc, end_support_version, instructions):
    """Adds a deprecation notice to a docstring for deprecated functions."""

    if instructions:
        deprecation_message = """
            .. warning::
                **THIS FUNCTION IS DEPRECATED:** It will be removed in TensorLayer version: %s.
                *Instructions for updating:* %s.
        """ % (end_support_version, instructions)

    else:
        deprecation_message = """
            .. warning::
                **THIS FUNCTION IS DEPRECATED:** It will be removed in TensorLayer version: %s.
        """ % end_support_version

    main_text = [deprecation_message]

    # _add_notice_to_docstring(doc, instructions, no_doc_str, suffix_str, notice)
    return _add_notice_to_docstring(
        doc=doc, instructions='DEPRECATED CLASS OR FUNCTION', no_doc_str="", suffix_str="", notice=main_text
    )


def call_location():
    """Returns call location given level up from current call."""
    frame = tf_inspect.currentframe()

    if frame:
        # CPython internals are available, use them for performance.
        # walk back two frames to get to deprecated function caller.
        first_frame = frame.f_back
        second_frame = first_frame.f_back
        frame = second_frame if second_frame else first_frame
        return '%s:%d' % (frame.f_code.co_filename, frame.f_lineno)

    else:
        # Slow fallback path
        stack = tf_inspect.stack(0)  # 0 avoids generating unused context
        entry = stack[2]
        return '%s:%d' % (entry[1], entry[2])


def get_qualified_name(function):
    # Python 3
    if hasattr(function, '__qualname__'):
        return function.__qualname__

    # Python 2
    if hasattr(function, 'im_class'):
        return function.im_class.__name__ + '.' + function.__name__
    return function.__name__


def get_network_obj(skip=2):
    stack = inspect.stack()

    if len(stack) < skip + 1:
        raise ValueError("The length of the inspection stack is shorter than the requested start position.")

    for current_stack in stack[skip:]:

        try:
            args, _, _, values = inspect.getargvalues(current_stack[0])

            if 'self' in values.keys() and isinstance(values['self'], tl.networks.CustomModel):
                return values['self']

            if 'cls' in values.keys() and isinstance(values['cls'], tl.networks.CustomModel):
                return values['cls']

        except Exception as e:
            print("Except Type 1:", type(e))
            continue

    return None


def rename_kwargs(kwargs, aliases, end_support_version, func_name):

    for alias, new in aliases.items():

        if alias in kwargs:

            if new in kwargs:
                raise TypeError('{}() received both {} and {}'.format(func_name, alias, new))

            warnings.warn('{}() - {} is deprecated; use {}'.format(func_name, alias, new), DeprecationWarning)
            tl.logging.warning(
                "DeprecationWarning: {}(): "
                "`{}` argument is deprecated and will be removed in version {}, "
                "please change for `{}.`".format(func_name, alias, end_support_version, new)
            )
            kwargs[new] = kwargs.pop(alias)


# def validate_deprecation_args(end_support_version, instructions):
#     if end_support_version is not None and not re.match(r'\d+\.\d+(\.\d+)?(\S*)?$', end_support_version):
#         raise ValueError('end_support_version does not comply with the semantic version format.')
def validate_deprecation_args(instructions):
    if not instructions:
        raise ValueError('Don\'t deprecate things without conversion instructions!')


def _add_notice_to_docstring(doc, instructions, no_doc_str, suffix_str, notice):
    """Adds a deprecation notice to a docstring."""
    if not doc:
        lines = [no_doc_str]

    else:
        lines = _normalize_docstring(doc).splitlines()
        # lines[0] += ' ' + suffix_str

    # notice = [''] + notice
    notice = [''] + notice + [instructions]

    if len(lines) > 1:
        # Make sure that we keep our distance from the main body
        if lines[1].strip():
            notice.append('')

        lines[1:1] = notice
    else:
        lines += notice

    return '\n'.join(lines)


def _normalize_docstring(docstring):
    """Normalizes the docstring.

    Replaces tabs with spaces, removes leading and trailing blanks lines, and
    removes any indentation.

    Copied from PEP-257:
    https://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation

    Args:
        docstring: the docstring to normalize

    Returns:
        The normalized docstring
    """
    if not docstring:
        return ''
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    # (we use sys.maxsize because sys.maxint doesn't exist in Python 3)
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return '\n'.join(trimmed)
