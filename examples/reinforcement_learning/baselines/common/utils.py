"""
Functions for utilization.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""
import operator
import os
import random
import re
from importlib import import_module

import gym
import matplotlib.pyplot as plt
import numpy as np

import tensorlayer as tl


def plot(episode_rewards, Algorithm_name, Env_name):
    '''
    plot the learning curve, saved as ./img/Algorithm_name.png
    :episode_rewards: array of floats
    :Algorithm_name: string
    :Env_name: string
    '''
    plt.figure(figsize=(10, 5))
    plt.title(Algorithm_name + '-' + Env_name)
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    if not os.path.exists('img'):
        os.makedirs('img')
    plt.savefig('./img/' + Algorithm_name + '.png')


def save_model(model, Model_name, Algorithm_name):
    '''
    save trained neural network model
    :model: tensorlayer.models.Model
    :Model_name: string, e.g. 'model_sac_q1'
    :Algorithm_name: string, e.g. 'SAC'
    '''
    if not os.path.exists('model/' + Algorithm_name):
        os.makedirs('model/' + Algorithm_name)
    tl.files.save_npz(model.trainable_weights, './model/' + Algorithm_name + '/' + Model_name)


def load_model(model, Model_name, Algorithm_name):
    '''
    load saved neural network model
    :model: tensorlayer.models.Model
    :Model_name: string, e.g. 'model_sac_q1'
    :Algorithm_name: string, e.g. 'SAC'
    '''
    try:
        tl.files.load_and_assign_npz('./model/' + Algorithm_name + '/' + Model_name + '.npz', model)
    except:
        print('Load Model Fails!')


def parse_all_args(parser):
    """ Parse known and unknown args """
    common_options, other_args = parser.parse_known_args()
    other_options = dict()
    index = 0
    n = len(other_args)
    float_pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    while index < n:  # only str, int and float type will be parsed
        if other_args[index].startswith('--'):
            if other_args[index].__contains__('='):
                key, value = other_args[index].split('=')
                index += 1
            else:
                key, value = other_args[index:index + 2]
                index += 2
            if re.match(float_pattern, value):
                value = float(value)
                if value.is_integer():
                    value = int(value)
            other_options[key[2:]] = value
    return common_options, other_options


def make_env(env_id):
    env = gym.make(env_id).unwrapped
    ''' add env wrappers here '''
    return env


def learn(env_id, algorithm, train_episodes, **kwargs):
    algorithm = algorithm.lower()
    module = get_algorithm_module(algorithm, algorithm)
    # env = make_env(env_id)
    getattr(module, 'learn')(env_id=env_id, train_episodes=train_episodes, **kwargs)  # call module.learn()


# def get_algorithm_parameters(env, env_type, algorithm, **kwargs):
#     """ Get algorithm hyper-parameters """
#     module = get_algorithm_module(algorithm, 'default')
#     return getattr(module, env_type)(env, **kwargs)


def get_algorithm_module(algorithm, submodule):
    """ Get algorithm module in the corresponding folder """
    return import_module('.'.join(['algorithms', algorithm, submodule]))
