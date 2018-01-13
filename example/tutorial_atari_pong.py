#! /usr/bin/python
# -*- coding: utf-8 -*-


""" Monte-Carlo Policy Network π(a|s)  (REINFORCE)

To understand Reinforcement Learning, we let computer to learn how to play
Pong game from the original screen inputs. Before we start, we highly recommend
you to go through a famous blog called “Deep Reinforcement Learning: Pong from
Pixels” which is a minimalistic implementation of deep reinforcement learning by
using python-numpy and OpenAI gym environment.

The code here is the reimplementation of Karpathy's Blog by using TensorLayer.

Compare with Karpathy's code, we store observation for a batch, he store
observation for a episode only, they store gradients instead. (so we will use
more memory if the observation is very large.)

Link
-----
http://karpathy.github.io/2016/05/31/rl/

"""

import time

import gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

# hyperparameters
image_size = 80
D = image_size * image_size
H = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
render = False          # display the game environment
# resume = True         # load existing policy network
model_file_name = "model_pong"
np.set_printoptions(threshold=np.nan)

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]
    I = I[::2,::2,0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
running_reward = None
reward_sum = 0
episode_number = 0

xs, ys, rs = [], [], []
# observation for training and inference
t_states = tf.placeholder(tf.float32, shape=[None, D])
# policy network
network = InputLayer(t_states, name='input')
network = DenseLayer(network, n_units=H, act=tf.nn.relu, name='hidden')
network = DenseLayer(network, n_units=3, name='output')
probs = network.outputs
sampling_prob = tf.nn.softmax(probs)

t_actions = tf.placeholder(tf.int32, shape=[None])
t_discount_rewards = tf.placeholder(tf.float32, shape=[None])
loss = tl.rein.cross_entropy_reward_loss(probs, t_actions, t_discount_rewards)
train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)

with tf.Session() as sess:
    tl.layers.initialize_global_variables(sess)
    # if resume:
    #     load_params = tl.files.load_npz(name=model_file_name+'.npz')
    #     tl.files.assign_params(sess, load_params, network)
    tl.files.load_and_assign_npz(sess, model_file_name+'.npz', network)
    network.print_params()
    network.print_layers()

    start_time = time.time()
    game_number = 0
    while True:
        if render: env.render()

        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        x = x.reshape(1, D)
        prev_x = cur_x

        prob = sess.run(
            sampling_prob,
            feed_dict={t_states: x})
        
        # action. 1: STOP  2: UP  3: DOWN
        # action = np.random.choice([1,2,3], p=prob.flatten())
        action = tl.rein.choice_action_by_probs(prob.flatten(), [1,2,3])

        observation, reward, done, _ = env.step(action)
        reward_sum += reward
        xs.append(x)            # all observations in an episode
        ys.append(action - 1)   # all fake labels in an episode (action begins from 1, so minus 1)
        rs.append(reward)       # all rewards in an episode
        
        if done:
            episode_number += 1
            game_number = 0

            if episode_number % batch_size == 0:
                print('batch over...... updating parameters......')
                epx = np.vstack(xs)
                epy = np.asarray(ys)
                epr = np.asarray(rs)
                disR = tl.rein.discount_episode_rewards(epr, gamma)
                disR -= np.mean(disR)
                disR /= np.std(disR)

                xs, ys, rs = [], [], []

                sess.run(
                    train_op,
                    feed_dict={
                        t_states: epx,
                        t_actions: epy,
                        t_discount_rewards: disR})

            if episode_number % (batch_size * 100) == 0:
                tl.files.save_npz(network.all_params, name=model_file_name+'.npz')

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            observation = env.reset() # reset env
            prev_x = None

        if reward != 0:
            print(('episode %d: game %d took %.5fs, reward: %f' %
                        (episode_number, game_number,
                        time.time()-start_time, reward)),
                        ('' if reward == -1 else ' !!!!!!!!'))
            start_time = time.time()
            game_number += 1
