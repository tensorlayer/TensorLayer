"""
Functions for utilization.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""
import random
import time

import matplotlib.pyplot as plt
import tensorlayer as tl
import numpy as np
import os



def plot(episode_rewards, Algorithm_name, Env_name):
    '''
    plot the learning curve, saved as ./img/Algorithm_name.png
    :episode_rewards: array of floats
    :Algorithm_name: string
    :Env_name: string
    '''
    plt.figure(figsize=(10,5))
    plt.title(Algorithm_name + '-' + Env_name )
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    if not os.path.exists('img'):
        os.makedirs('img')
    plt.savefig( './img/' + Algorithm_name + '.png')


def save_model(model, Model_name, Algorithm_name):
    '''
    save trained neural network model
    :model: tensorlayer.models.Model
    :Model_name: string, e.g. 'model_sac_q1'
    :Algorithm_name: string, e.g. 'SAC'
    '''
    if not os.path.exists('model/'+Algorithm_name):
        os.makedirs('model/'+Algorithm_name)
    tl.files.save_npz(model.trainable_weights, './model/' + Algorithm_name + '/'+Model_name)

def load_model(model, Model_name, Algorithm_name):
    '''
    load saved neural network model
    :model: tensorlayer.models.Model
    :Model_name: string, e.g. 'model_sac_q1'
    :Algorithm_name: string, e.g. 'SAC'
    '''
    try:
        tl.files.load_and_assign_npz('./model/' + Algorithm_name + '/'+Model_name + '.npz', model)
    except:
        print('Load Model Fails!')


class ReplayBuffer:
    '''
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    '''
    def __init__(self, capacity):
        self.capacity = capacity  # mamimum number of samples
        self.buffer = []
        self.position = 0      # pointer 
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)





