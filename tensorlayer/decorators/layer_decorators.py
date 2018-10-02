#! /usr/bin/python
# -*- coding: utf-8 -*-

from functools import wraps

import wrapt

import tensorflow as tf

from tensorlayer.decorators.utils import get_network_obj

__all__ = ['force_return_self', 'layer_autoregister', 'overwrite_layername_in_network']


def force_return_self(method):
    """decorator that overwrite the returned value and return instead the object."""

    @wraps(method)
    def _impl(self, *args, **kwargs):
        method(self, *args, **kwargs)
        return self

    return _impl


@wrapt.decorator
def layer_autoregister(wrapped, instance, args, kwargs):

    cls = wrapped(*args, **kwargs)

    try:
        network_obj = get_network_obj()

        if network_obj is not None:
            network_obj.register_new_layer(instance)

    except Exception as e:
        print("Except Type 2: %s - Error: %s" % (type(e), str(e)))
        pass

    return cls


@wrapt.decorator
def overwrite_layername_in_network(wrapped, instance, args, kwargs):

    cls = wrapped(*args, **kwargs)

    try:

        network_obj = get_network_obj()

        if network_obj is not None:
            current_var_scope = tf.get_default_graph().get_name_scope()

            if len(current_var_scope) > 0:

                result_scope = ""

                for i_varscope, i_modelscope in zip(current_var_scope.split("/"), network_obj.model_scope.split("/")):
                    if i_varscope != i_modelscope:
                        result_scope += i_varscope + "/"

                instance.name = result_scope + instance.name

    except Exception as e:
        print("Except Type 3: %s - Error: %s" % (type(e), str(e)))
        pass

    return cls
