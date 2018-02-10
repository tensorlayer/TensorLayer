#!/usr/bin/env python
# encoding: utf-8

"""

(Alpha release)

The tensorlayer.cli.train module provides the `tl train` subcommand.
It helps the user to setup a `TF_CONFIG` environment to bootstrap a
TF_CONFIG aware program for distributed training using multiple GPU 
cards or CPUs on a computer.

You need to first setup the CUDA_VISIBLE_DEVICES to tell `tl train`
while GPUs to use in parallel. If the CUDA_VISIBLE_DEVICES is not given,
`tl train` would try best to discover all available GPUs.


Usage:

tl train [-h] [-p NUM_PSS] [-c CPU_TRAINERS] <file> [args [args ...]]

.. code-block:: bash

  tl train example/tutorial_mnist_distributed.py
  tl train example/tutorial_imagenet_inceptionV3_distributed.py -- --batch_size 16


Parameters:

`file`: python file path.
`pss` : The number of parameter servers.
`cpu_trainers`: The number of CPU trainers. Note that `pss + cpu_trainers <= cpu count`
`--`: Any parameter after `--` would be passed to the python program.

Notes:

A parallel training program would require multiple parameter servers
to help paralell trainers to exchange intermediate gradients.
The best number of parameter servers is often propotional to the 
size of your model as well as the number of CPUs available.
You can control the number of parameter servers using the `pss` parameter.

If you have a single computer with massive CPUs, you can use the `cpu_trainers`
to enable CPU-only parallel training.
The reason we are not supporting GPU-CPU co-training is because GPU and 
CPU are running at different speeds. Using them together in training would
incur severe straggler issues.

"""

import argparse
import json
import multiprocessing
import os
import platform
import re
import subprocess
import sys

PORT_BASE = 10000


def _get_gpu_ids():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        return [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')]
    if platform.system() in ['Darwin', 'Linux']:
        return [int(d.replace('nvidia', '')) for d in os.listdir('/dev') if re.match('^nvidia\d+$', d)]
    else:
        print('Please set CUDA_VISIBLE_DEVICES (see http://acceleware.com/blog/cudavisibledevices-masking-gpus)')
        return []


GPU_IDS = _get_gpu_ids()


def create_tf_config(cluster_spec, task_type, task_index):
    return {
        'cluster': cluster_spec,
        'task': {
            'type': task_type,
            'index': task_index
        },
    }


def create_tf_jobs(cluster_spec, prog, args):
    gpu_assignment = dict((('worker', idx), gpu_idx) for (idx, gpu_idx) in enumerate(GPU_IDS))
    for job_type in cluster_spec:
        for task_index in range(len(cluster_spec[job_type])):
            new_env = os.environ.copy()
            new_env.update({
                'CUDA_VISIBLE_DEVICES': str(gpu_assignment.get((job_type, task_index), '')),
                'TF_CONFIG': json.dumps(create_tf_config(cluster_spec, job_type, task_index)),
            })
            yield subprocess.Popen(['python3', prog] + args, env=new_env)


def validate_arguments(args):
    if args.num_pss < 1:
        print('Value error: must have ore than one parameter servers.')
        exit(1)

    if not GPU_IDS:
        num_cpus = multiprocessing.cpu_count()
        if args.cpu_trainers > num_cpus:
            print('Value error: there are %s available CPUs but you are requiring %s.' % (num_cpus, args.cpu_trainers))
            exit(1)

    if not os.path.isfile(args.file):
        print('Value error: model trainning file does not exist')
        exit(1)


def main(args):
    validate_arguments(args)
    num_workers = len(GPU_IDS) if GPU_IDS else args.cpu_trainers
    print('Using program %s with args %s' % (args.file, ' '.join(args.args)))
    print('Using %d workers, %d parameter servers, %d GPUs.' % (num_workers, args.num_pss, len(GPU_IDS)))
    cluster_spec = {
        'ps': ['localhost:%d' % (PORT_BASE + i) for i in range(args.num_pss)],
        'worker': ['localhost:%d' % (PORT_BASE + args.num_pss + i) for i in range(num_workers)]
    }
    processes = list(create_tf_jobs(cluster_spec, args.file, args.args))
    try:
        print('Press ENTER to exit the training ...')
        sys.stdin.readline()
    except KeyboardInterrupt:  # https://docs.python.org/3/library/exceptions.html#KeyboardInterrupt
        print('Keyboard interrupt received')
    finally:
        print('stopping all subprocesses ...')
        for p in processes:
            p.kill()
        for p in processes:
            p.wait()
        print('END')


def build_arg_parser(parser):
    parser.add_argument('-p', '--pss', dest='num_pss', type=int, default=1, help='number of parameter servers')
    parser.add_argument('-c', '--cpu_trainers', dest='cpu_trainers', type=int, default=1, help='number of CPU trainers')
    parser.add_argument('file', help='model trainning file path')
    parser.add_argument('args', nargs='*', type=str, help='arguments to <file>')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    build_arg_parser(parser)
    args = parser.parse_args()
    main(args)
