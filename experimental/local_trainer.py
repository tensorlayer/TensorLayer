#!/usr/bin/env python
# encoding: utf-8

import argparse
import json
import os
import subprocess

PORT_BASE = 10000
ENABLE_GPU = False  # Set to False only for local testing


def create_tf_config_str(cluster_spec, task_type, task_index):
    full_spec = cluster_spec.copy()
    full_spec['task'] = {
        'type': task_type,
        'index': task_index
    }
    return json.dumps(full_spec)


def run_workers(cluster_spec, file):
    processes = []
    gpu_id = 0  # Assume GPU device id starts from 0
    for task_index in range(0, len(cluster_spec['cluster']['worker'])):
        add_env = dict()

        if ENABLE_GPU:
            add_env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            gpu_id = gpu_id + 1
        else:
            add_env['CUDA_VISIBLE_DEVICES'] = ''  # Worker uses CPU if GPU is not available

        add_env['TF_CONFIG'] = create_tf_config_str(cluster_spec, 'worker', task_index)

        new_env = os.environ.copy()
        new_env.update(add_env)

        cmd = 'python3 ' + file
        process = subprocess.Popen(cmd, env=new_env, shell=True)
        processes.append(process)
    return processes


def run_parameter_servers(cluster_spec, file):
    processes = []
    for task_index in range(0, len(cluster_spec['cluster']['ps'])):
        add_env = dict()
        add_env['CUDA_VISIBLE_DEVICES'] = ''  # Parameter server does not need to see any GPU
        add_env['TF_CONFIG'] = create_tf_config_str(cluster_spec, 'ps', task_index)

        cmd = 'python3 ' + file

        new_env = os.environ.copy()
        new_env.update(add_env)
        process = subprocess.Popen(cmd, env=new_env, shell=True)
        processes.append(process)
    return processes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pss", help="number of parameter servers")
    parser.add_argument("-w", "--workers", help="number of workers")
    parser.add_argument("-f", "--file", help="file path")
    args = parser.parse_args()

    cluster_spec = {
        'cluster': {
            'ps': ['localhost:' + str(PORT_BASE + i) for i in range(0, int(args.pss))],
            'worker': ['localhost:' + str(PORT_BASE + int(args.pss) + i) for i in range(0, int(args.workers))]
        }
    }

    pss = run_parameter_servers(cluster_spec, args.file)
    workers = run_workers(cluster_spec, args.file)

    input("Press Enter to exit...\n")

    for p in pss:
        p.kill()

    for p in workers:
        p.kill()

    print("END")
