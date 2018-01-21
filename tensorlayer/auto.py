import json
import os

import tensorflow as tf
from . import distributed
 

def Session():
    task_spec = distributed.TaskSpec()
    if not task_spec:
        return tf.InteractiveSession().as_default()
    task_spec.create_server()
    return distributed.DistributedSession(task_spec=task_spec)
