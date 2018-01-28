#!/usr/bin/env python
# encoding: utf-8

# example usage:
#   ./experimental/local_trainer.py -w 2 -f example/tutorial_mnist_distributed.py

import argparse
import json
import multiprocessing
import os
import re
import subprocess
import sys

PORT_BASE = 10000


def get_gpus():
    return [d for d in os.listdir('/dev') if re.match('^nvidia\d+$', d)]


def create_tf_config(cluster_spec, task_type, task_index):
    return {
        'cluster': cluster_spec,
        'task': {
            'type': task_type,
            'index': task_index
        },
    }


def create_tf_jobs(prog, cluster_spec, enable_gpu=False):
    for job_type in cluster_spec:
        for task_index in range(len(cluster_spec[job_type])):
            new_env = os.environ.copy()
            new_env.update({
                'CUDA_VISIBLE_DEVICES': str(task_index) if job_type == 'worker' and enable_gpu else '',
                'TF_CONFIG': json.dumps(create_tf_config(cluster_spec, job_type, task_index)),
            })
            yield subprocess.Popen(['python3', prog], env=new_env)


def validate_arguments(args):
    if args.num_pss < 1:
        print('Value error: must have ore than one parameter servers.')
        exit(1)

    if args.enable_gpu:
        num_gpus = len(get_gpus())
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

    processes = list(create_tf_jobs(args.file, cluster_spec, args.enable_gpu))
    try:
        print('Press ENTER to exit the training ...')
        sys.stdin.readline()
    except KeyboardInterrupt:  # https://docs.python.org/3/library/exceptions.html#KeyboardInterrupt
        print('Keyboard interrupt received')
    finally:
        print('stopping all subprocesses ...')
        for p in processes:
            p.kill()
        print('END')
