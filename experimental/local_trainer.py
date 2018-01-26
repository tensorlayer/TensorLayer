#!/usr/bin/env python
# encoding: utf-8

# example usage:
#   ./experimental/local_trainer.py -w 2 -f example/tutorial_mnist_distributed.py

import argparse
import json
import multiprocessing
import os
import signal
import subprocess
import sys

from tensorflow.python.client import device_lib

PORT_BASE = 10000


def create_tf_config_str(cluster_spec, task_type, task_index):
    tf_config = {
        'cluster': cluster_spec,
        'task': {
            'type': task_type,
            'index': task_index
        },
    }
    return json.dumps(tf_config)


def run_workers(cluster_spec, enable_gpu, file):
    for task_index in range(len(cluster_spec['worker'])):
        add_env = dict()
        add_env['CUDA_VISIBLE_DEVICES'] = str(task_index) if enable_gpu else ''
        add_env['TF_CONFIG'] = create_tf_config_str(cluster_spec, 'worker', task_index)
        new_env = os.environ.copy()
        new_env.update(add_env)
        cmd = 'python3 ' + file
        yield subprocess.Popen(cmd, env=new_env, shell=True)


def run_parameter_servers(cluster_spec, file):
    for task_index in range(len(cluster_spec['ps'])):
        add_env = dict()
        add_env['CUDA_VISIBLE_DEVICES'] = ''  # Parameter server does not need to see any GPU
        add_env['TF_CONFIG'] = create_tf_config_str(cluster_spec, 'ps', task_index)
        new_env = os.environ.copy()
        new_env.update(add_env)
        cmd = 'python3 ' + file
        yield subprocess.Popen(cmd, env=new_env, shell=True)


def validate_arguments(args):
    if args.num_pss < 1:
        print('Value error: must have ore than one parameter servers.')
        exit(1)

    if args.enable_gpu:
        num_gpus = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
        if args.num_workers > num_gpus:
            print('Value error: there are %s available GPUs but you are requiring %s.' % (num_gpus, args.num_workers))
            exit(1)
    else:
        num_cpus = multiprocessing.cpu_count()
        if args.num_workers > num_cpus:
            print('Value error: there are %s available CPUs but you are requiring %s.' % (num_cpus, args.num_workers))
            exit(1)

    if not os.path.isfile(args.file):
        print('Value error: model trainning file does not exist')
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pss', dest='num_pss', type=int, default=1, help='number of parameter servers')
    parser.add_argument('-w', '--workers', dest='num_workers', type=int, required=True, help='number of workers')
    parser.add_argument(
        '-g', '--enable_gpu', dest='enable_gpu', action='store_true', help='enable GPU (GPU and CPU are NOT enabled together to avoid stragglers)')
    parser.add_argument('-f', '--file', dest='file', help='model trainning file path')
    args = parser.parse_args()

    validate_arguments(args)

    cluster_spec = {
        'ps': ['localhost:%d' % (PORT_BASE + i) for i in range(args.num_pss)],
        'worker': ['localhost:%d' % (PORT_BASE + args.num_pss + i) for i in range(args.num_workers)]
    }

    try:
        processes = list(run_parameter_servers(cluster_spec, args.file)) + list(run_workers(cluster_spec, args.enable_gpu, args.file))
        input('Press ENTER to exit the training ...\n')
    except KeyboardInterrupt:
        print('Keyboard interrupt received, stopping ...')
    finally:
        # clean up
        for p in processes:
            p.kill()
