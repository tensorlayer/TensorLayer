#! /usr/bin/python
# -*- coding: utf-8 -*-

import inspect
import wrapt

from tensorlayer.decorators.utils import get_network_obj

__all__ = ['layer_autoregister']


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
