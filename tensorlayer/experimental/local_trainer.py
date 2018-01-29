#!/usr/bin/env python
# encoding: utf-8

# example usage:
#   ./tensorlayer/experimental/local_trainer.py ./example/tutorial_mnist_distributed.py
#   ./tensorlayer/experimental/local_trainer.py ./example/tutorial_imagenet_inceptionV3_distributed.py -- --batch_size 16

import argparse
import json
import multiprocessing
import os
import re
import subprocess
import sys

PORT_BASE = 10000


def _get_gpu_ids():
    available_gpu_ids = [int(d.replace('nvidia', '')) for d in os.listdir('/dev') if re.match('^nvidia\d+$', d)]
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        visiable_gpu_ids = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')]
        available_gpu_ids = list(set(available_gpu_ids) & set(visiable_gpu_ids))
    return available_gpu_ids


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

    if GPU_IDS:
        num_gpus = len(GPU_IDS)
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


def main(args):
    validate_arguments(args)
    print('Using program %s with args %s' % (args.file, ' '.join(args.args)))
    print('Using %d workers, %d parameter servers, %d GPUs.' % (args.num_workers, args.num_pss, len(GPU_IDS)))
    cluster_spec = {
        'ps': ['localhost:%d' % (PORT_BASE + i) for i in range(args.num_pss)],
        'worker': ['localhost:%d' % (PORT_BASE + args.num_pss + i) for i in range(args.num_workers)]
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
    parser.add_argument('-w', '--workers', dest='num_workers', type=int, default=len(GPU_IDS) if GPU_IDS else 1, help='number of workers')
    parser.add_argument('file', help='model trainning file path')
    parser.add_argument('args', nargs='*', type=str, help='arguments to <file>')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    build_arg_parser(parser)
    args = parser.parse_args()
    main(args)
