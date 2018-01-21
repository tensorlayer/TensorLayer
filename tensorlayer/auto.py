import json
import os

import tensorflow as tf


def create_session():
    tf_config = os.environ.get('TF_CONFIG')
    if not tf_config:
        return tf.InteractiveSession()
    # TODO: use tl.distributed
    tf_config = json.loads(tf_config)
    cluster = tf.train.ClusterSpec(tf_config['cluster'])
    task = tf_config['task']
    server = tf.train.Server(
        cluster, job_name=task['type'], task_index=task['index'])
    sess = tf.Session(server.target)
    return sess
