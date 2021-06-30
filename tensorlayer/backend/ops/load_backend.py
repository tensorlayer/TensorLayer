#! /usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import sys

BACKEND = 'tensorflow'
# BACKEND = 'mindspore'
# BACKEND = 'paddle'

# Check for backend.json files
tl_backend_dir = os.path.expanduser('~')
if not os.access(tl_backend_dir, os.W_OK):
    tl_backend_dir = '/tmp'
tl_dir = os.path.join(tl_backend_dir, '.tl')

config = {
    'backend': BACKEND,
}
if not os.path.exists(tl_dir):
    path = os.path.join(tl_dir, 'tl_backend.json')
    os.makedirs(tl_dir)
    with open(path, "w") as f:
        json.dump(config, f)
    BACKEND = config['backend']
    sys.stderr.write("Create the backend configuration file :" + path + '\n')
else:
    path = os.path.join(tl_dir, 'tl_backend.json')
    with open(path, 'r') as load_f:
        load_dict = json.load(load_f)
    if load_dict['backend'] is not config['backend']:
        BACKEND = config['backend']
    else:
        BACKEND = load_dict['backend']

# Set backend based on TL_BACKEND.
if 'TL_BACKEND' in os.environ:
    backend = os.environ['TL_BACKEND']
    if backend:
        BACKEND = backend

# import backend functions
if BACKEND == 'tensorflow':
    from .tensorflow_backend import *
    from .tensorflow_nn import *
    import tensorflow as tf
    BACKEND_VERSION = tf.__version__
    sys.stderr.write('Using TensorFlow backend.\n')

elif BACKEND == 'mindspore':
    from .mindspore_backend import *
    from .mindspore_nn import *
    import mindspore as ms
    BACKEND_VERSION = ms.__version__
    # set context
    import mindspore.context as context
    import os
    os.environ['DEVICE_ID'] = '0'
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU'),
    # context.set_context(mode=context.GRAPH_MODE, device_target='CPU'),
    # enable_task_sink=True, enable_loop_sink=True)
    # context.set_context(mode=context.GRAPH_MODE, backend_policy='ms',
    #                     device_target='Ascend', enable_task_sink=True, enable_loop_sink=True)
    sys.stderr.write('Using MindSpore backend.\n')

elif BACKEND == 'paddle':
    from .paddle_backend import *
    from .paddle_nn import *
    import paddle as pd
    BACKEND_VERSION = pd.__version__
    sys.stderr.write('Using Paddle backend.\n')
else:
    raise NotImplementedError("This backend is not supported")
