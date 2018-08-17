#! /usr/bin/python
# -*- coding: utf-8 -*-

import inspect
import sys
import functools

import tensorlayer as tl

from tensorlayer.decorators.utils import DeprecatedArgSpec
from tensorlayer.decorators.utils import add_deprecated_arg_notice_to_docstring
from tensorlayer.decorators.utils import add_deprecation_notice_to_docstring
from tensorlayer.decorators.utils import call_location
from tensorlayer.decorators.utils import get_qualified_name
from tensorlayer.decorators.utils import rename_kwargs
from tensorlayer.decorators.utils import validate_deprecation_args

from tensorflow.python.util import decorator_utils
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

import wrapt

__all__ = ['deprecated', 'deprecated_alias', 'deprecated_args']

# Allow deprecation warnings to be silenced temporarily with a context manager.
_PRINT_DEPRECATION_WARNINGS = True

# Remember which deprecation warnings have been printed already.
_PRINTED_WARNING = {}


def deprecated(wrapped=None, end_support_version='', instructions='', warn_once=True):

    if wrapped is None:
        return functools.partial(
            deprecated, end_support_version=end_support_version, instructions=instructions, warn_once=warn_once
        )

    @wrapt.decorator
    def wrapper(wrapped, instance=None, args=None, kwargs=None):

        validate_deprecation_args(instructions)

        if _PRINT_DEPRECATION_WARNINGS:

            class_or_func_name = get_qualified_name(wrapped)

            if class_or_func_name not in _PRINTED_WARNING:
                if warn_once:
                    _PRINTED_WARNING[class_or_func_name] = True

                if not inspect.isclass(wrapped):
                    filename = wrapped.__code__.co_filename
                    wrapped_type = "Function"
                else:
                    filename = wrapped.__module__
                    wrapped_type = "Class"

                tl.logging.warning(
                    '%s: `%s.%s` (in file: `%s`) is deprecated and will be removed in version %s.\n'
                    'Instructions for updating: %s\n' %
                    (wrapped_type, wrapped.__module__, class_or_func_name, filename, end_support_version, instructions)
                )

        return wrapped(*args, **kwargs)

    decorated = wrapper(wrapped)

    if sys.version_info > (3, 0):  # docstring can only be edited with Python 3
        wrapt.FunctionWrapper.__setattr__(
            decorated, "__doc__",
            add_deprecation_notice_to_docstring(wrapped.__doc__, end_support_version, instructions)
        )

    return decorated


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


def deprecated_args(end_support_version, instructions, deprecated_args, warn_once=True):
    """Decorator for marking specific function arguments as deprecated.

    This decorator logs a deprecation warning whenever the decorated function is
    called with the deprecated argument. It has the following format:
    Calling <function> (from <module>) with <arg> is deprecated and will be
    removed after <date>. Instructions for updating:
    <instructions>

    If `date` is None, 'after <date>' is replaced with 'in a future version'.
    <function> includes the class name if it is a method.
    It also edits the docstring of the function: ' (deprecated arguments)' is
    appended to the first line of the docstring and a deprecation notice is
    prepended to the rest of the docstring.

    Args:
        date: String or None. The date the function is scheduled to be removed.
            Must be ISO 8601 (YYYY-MM-DD), or None.
        instructions: String. Instructions on how to update code using the
            deprecated function.
        deprecated_args: A Tuple of strings or 2-Tuple(String,
            [ok_vals]).    The string is the deprecated argument name.
            Optionally, an ok-value may be provided.    If the user provided
            argument equals this value, the warning is suppressed.
        warn_once: If `warn_once=False` is passed, every call with a deprecated
            argument will log a warning. The default behavior is to only warn the
            first time the function is called with any given deprecated argument.

    Returns:
        Decorated function or method.

    Raises:
        ValueError: If date is not None or in ISO 8601 format, instructions are
            empty, the deprecated arguments are not present in the function
            signature, the second element of a deprecated_tuple is not a
            list, or if a kwarg other than `warn_once` is passed.
    """

    validate_deprecation_args(instructions)

    if not deprecated_args:
        raise ValueError('Specify which argument is deprecated.')

    def _get_arg_names_to_ok_vals():
        """Returns a dict mapping arg_name to DeprecatedArgSpec w/o position."""
        d = {}

        for name_or_tuple in deprecated_args:

            if isinstance(name_or_tuple, tuple):
                d[name_or_tuple[0]] = DeprecatedArgSpec(-1, True, name_or_tuple[1])

            else:
                d[name_or_tuple] = DeprecatedArgSpec(-1, False, None)

        return d

    def _get_deprecated_positional_arguments(names_to_ok_vals, arg_spec):
        """Builds a dictionary from deprecated arguments to their spec.

        Returned dict is keyed by argument name.
        Each value is a DeprecatedArgSpec with the following fields:
            position: The zero-based argument position of the argument
                within the signature. None if the argument isn't found in the signature.
            ok_values: Values of this argument for which warning will be suppressed.
        Args:
            names_to_ok_vals: dict from string arg_name to a list of values,
                possibly empty, which should not elicit a warning.
            arg_spec: Output from tf_inspect.getargspec on the called function.
        Returns:
            Dictionary from arg_name to DeprecatedArgSpec.
        """
        arg_name_to_pos = dict((name, pos) for (pos, name) in enumerate(arg_spec.args))

        deprecated_positional_args = {}

        for arg_name, spec in iter(names_to_ok_vals.items()):

            if arg_name in arg_name_to_pos:
                pos = arg_name_to_pos[arg_name]
                deprecated_positional_args[arg_name] = DeprecatedArgSpec(pos, spec.has_ok_value, spec.ok_value)

        return deprecated_positional_args

    def deprecated_wrapper(func):
        """Deprecation decorator."""
        decorator_utils.validate_callable(func, 'deprecated_args')

        deprecated_arg_names = _get_arg_names_to_ok_vals()

        arg_spec = tf_inspect.getargspec(func)

        deprecated_positions = _get_deprecated_positional_arguments(deprecated_arg_names, arg_spec)

        is_varargs_deprecated = arg_spec.varargs in deprecated_arg_names
        is_kwargs_deprecated = arg_spec.keywords in deprecated_arg_names

        if (len(deprecated_positions) + is_varargs_deprecated + is_kwargs_deprecated != len(deprecated_args)):
            known_args = arg_spec.args + [arg_spec.varargs, arg_spec.keywords]

            missing_args = [arg_name for arg_name in deprecated_arg_names if arg_name not in known_args]

            raise ValueError(
                'The following deprecated arguments are not present '
                'in the function signature: %s. '
                'Found next arguments: %s.' % (missing_args, known_args)
            )

        def _same_value(a, b):
            """A comparison operation that works for multiple object types.

            Returns True for two empty lists, two numeric values with the
            same value, etc.

            Returns False for (pd.DataFrame, None), and other pairs which
            should not be considered equivalent.

            Args:
                a: value one of the comparison.
                b: value two of the comparison.

            Returns:
                A boolean indicating whether the two inputs are the same value
                for the purposes of deprecation.
            """
            if a is b:
                return True

            try:
                equality = a == b
                if isinstance(equality, bool):
                    return equality

            except TypeError:
                return False

            return False

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            """Deprecation wrapper."""
            if _PRINT_DEPRECATION_WARNINGS:
                invalid_args = []
                named_args = tf_inspect.getcallargs(func, *args, **kwargs)

                for arg_name, spec in iter(deprecated_positions.items()):
                    if (
                        spec.position < len(args) and
                        not (spec.has_ok_value and _same_value(named_args[arg_name], spec.ok_value))
                    ):

                        invalid_args.append(arg_name)

                if is_varargs_deprecated and len(args) > len(arg_spec.args):
                    invalid_args.append(arg_spec.varargs)

                if is_kwargs_deprecated and kwargs:
                    invalid_args.append(arg_spec.keywords)

                for arg_name in deprecated_arg_names:
                    if (
                        arg_name in kwargs and not (
                            deprecated_positions[arg_name].has_ok_value and
                            _same_value(named_args[arg_name], deprecated_positions[arg_name].ok_value)
                        )
                    ):
                        invalid_args.append(arg_name)

                for arg_name in invalid_args:
                    '''
                    tl.logging.warning(
                        '%s: `%s.%s` (in file: `%s`) is deprecated and will be removed %s.\n'
                        'Instructions for updating: %s\n' % (
                            wrapped_type, wrapped.__module__, class_or_func_name, filename, 'in a future version'
                            if date is None else ('after %s' % date), instructions
                        )
                    )
                    '''

                    if (func, arg_name) not in _PRINTED_WARNING:
                        if warn_once:
                            _PRINTED_WARNING[(func, arg_name)] = True

                        tl.logging.warning(
                            'From %s: calling %s (from %s) with %s is deprecated and will be removed in version %s\n'
                            'Instructions for updating:\n%s', call_location(), decorator_utils.get_qualified_name(func),
                            func.__module__, arg_name, end_support_version, instructions
                        )

            return func(*args, **kwargs)

        return tf_decorator.make_decorator(
            func, new_func, 'deprecated',
            add_deprecated_arg_notice_to_docstring(func.__doc__, end_support_version, instructions)
        )

    return deprecated_wrapper
