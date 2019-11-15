'''
Twin Delayed DDPG (TD3)
------------------------
DDPG suffers from problems like overestimate of Q-values and sensitivity to hyper-parameters.
Twin Delayed DDPG (TD3) is a variant of DDPG with several tricks:
* Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”), 
and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.

* Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently 
than the Q-function. 

* Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for 
the policy to exploit Q-function errors by smoothing out Q along changes in action.

The implementation of TD3 includes 6 networks: 2 Q-net, 2 target Q-net, 1 policy net, 1 target policy net
Actor policy in TD3 is deterministic, with Gaussian exploration noise.

Reference
---------
original paper: https://arxiv.org/pdf/1802.09477.pdf


Environment
---
Openai Gym Pendulum-v0, continuous action space
https://gym.openai.com/envs/Pendulum-v0/

Prerequisites
---
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

&&
pip install box2d box2d-kengz --user


'''

import argparse
import math
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from IPython.display import clear_output

import tensorlayer as tl
from common.buffer import *
from common.networks import *
from common.utils import *
from tensorlayer.layers import Dense
from tensorlayer.models import Model

tfd = tfp.distributions
Normal = tfd.Normal

tl.logging.set_verbosity(tl.logging.DEBUG)

###############################  TD3  ####################################


class PolicyNetwork(Model):
    ''' the network for generating non-determinstic (Gaussian distributed) action from the state input '''

    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')

        self.output_linear = Dense(n_units=num_actions, W_init=w_init, \
        b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim, name='policy_output')

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)

        output = tf.nn.tanh(self.output_linear(x))  # unit range output [-1, 1]

        return output

    def evaluate(self, state, eval_noise_scale):
        ''' 
        generate action with state for calculating gradients;
        eval_noise_scale: as the trick of target policy smoothing, for generating noisy actions.
        '''
        state = state.astype(np.float32)
        action = self.forward(state)

        action = self.action_range * action

        # add noise
        normal = Normal(0, 1)
        eval_noise_clip = 2 * eval_noise_scale
        noise = normal.sample(action.shape) * eval_noise_scale
        noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip)
        action = action + noise

        return action

    def get_action(self, state, explore_noise_scale):
        ''' generate action with state for interaction with envronment '''
        action = self.forward([state])
        action = action.numpy()[0]

        # add noise
        normal = Normal(0, 1)
        noise = normal.sample(action.shape) * explore_noise_scale
        action = self.action_range * action + noise

        return action.numpy()

    def sample_action(self, ):
        ''' generate random actions for exploration '''
        a = tf.random.uniform([self.num_actions], -1, 1)

        return self.action_range * a.numpy()


class TD3_Trainer():

    def __init__(
            self, replay_buffer, hidden_dim, state_dim, action_dim, action_range, policy_target_update_interval=1,
            q_lr=3e-4, policy_lr=3e-4
    ):
        self.replay_buffer = replay_buffer

        # initialize all networks
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = tf.optimizers.Adam(q_lr)
        self.q_optimizer2 = tf.optimizers.Adam(q_lr)
        self.policy_optimizer = tf.optimizers.Adam(policy_lr)

    def target_ini(self, net, target_net):
        ''' hard-copy update for initializing target networks '''
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        ''' soft update the target net with Polyak averaging '''
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
            )
        return target_net

    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):
        ''' update all networks in TD3 '''
        self.update_cnt += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        reward = reward[:, np.newaxis]  # expand dim
        done = done[:, np.newaxis]

        new_next_action = self.target_policy_net.evaluate(
            next_state, eval_noise_scale=eval_noise_scale
        )  # clipped normal noise
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (
            np.std(reward, axis=0) + 1e-6
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem

        # Training Q Function
        target_q_input = tf.concat([next_state, new_next_action], 1)  # the dim 0 is number of samples
        target_q_min = tf.minimum(self.target_q_net1(target_q_input), self.target_q_net2(target_q_input))

        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_input = tf.concat([state, action], 1)  # input of q_net

        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.square(predicted_q_value1 - target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.q_net1.trainable_weights)
        self.q_optimizer1.apply_gradients(zip(q1_grad, self.q_net1.trainable_weights))

        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.square(predicted_q_value2 - target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.q_net2.trainable_weights)
        self.q_optimizer2.apply_gradients(zip(q2_grad, self.q_net2.trainable_weights))

        # Training Policy Function
        if self.update_cnt % self.policy_target_update_interval == 0:
            with tf.GradientTape() as p_tape:
                new_action = self.policy_net.evaluate(
                    state, eval_noise_scale=0.0
                )  # no noise, deterministic policy gradients
                new_q_input = tf.concat([state, new_action], 1)
                # ''' implementation 1 '''
                # predicted_new_q_value = tf.minimum(self.q_net1(new_q_input),self.q_net2(new_q_input))
                ''' implementation 2 '''
                predicted_new_q_value = self.q_net1(new_q_input)
                policy_loss = -tf.reduce_mean(predicted_new_q_value)
            p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

            # Soft update the target nets
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

    def save_weights(self):  # save trained weights
        save_model(self.q_net1, 'model_q_net1', 'TD3')
        save_model(self.q_net2, 'model_q_net2', 'TD3')
        save_model(self.target_q_net1, 'model_target_q_net1', 'TD3')
        save_model(self.target_q_net2, 'model_target_q_net2', 'TD3')
        save_model(self.policy_net, 'model_policy_net', 'TD3')
        save_model(self.target_policy_net, 'model_target_policy_net', 'TD3')

    def load_weights(self):  # load trained weights
        load_model(self.q_net1, 'model_q_net1', 'TD3')
        load_model(self.q_net2, 'model_q_net2', 'TD3')
        load_model(self.target_q_net1, 'model_target_q_net1', 'TD3')
        load_model(self.target_q_net2, 'model_target_q_net2', 'TD3')
        load_model(self.policy_net, 'model_policy_net', 'TD3')
        load_model(self.target_policy_net, 'model_target_policy_net', 'TD3')


def learn(env_id, train_episodes, test_episodes=1000, max_steps=150, batch_size=64, explore_steps=500, update_itr=3, hidden_dim=32, \
    q_lr = 3e-4, policy_lr = 3e-4, policy_target_update_interval = 3, action_range = 1., \
    replay_buffer_size = 5e5, reward_scale = 1. , seed=2, save_interval=500, explore_noise_scale = 1.0, eval_noise_scale = 0.5, mode='train'):
    '''
    parameters
    ----------
    env: learning environment
    train_episodes:  total number of episodes for training
    test_episodes:  total number of episodes for testing
    max_steps:  maximum number of steps for one episode
    batch_size:  udpate batchsize
    explore_steps:  for random action sampling in the beginning of training
    update_itr: repeated updates for single step
    hidden_dim:  size of hidden layers for networks
    q_lr: q_net learning rate
    policy_lr: policy_net learning rate
    policy_target_update_interval: delayed update for the policy network and target networks
    action_range: range of action value
    replay_buffer_size: size of replay buffer
    reward_scale: value range of reward
    save_interval: timesteps for saving the weights and plotting the results
    explore_noise_scale: range of action noise for exploration
    eval_noise_scale: range of action noise for evaluation of action value
    mode: train or test

    '''
    env = make_env(env_id)  # make env with common.utils and wrappers
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)  # reproducible

    # initialization of buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)
    # initialization of trainer
    td3_trainer=TD3_Trainer(replay_buffer, hidden_dim=hidden_dim, state_dim=state_dim, action_dim=action_dim, policy_target_update_interval=policy_target_update_interval, \
    action_range=action_range, q_lr=q_lr, policy_lr=policy_lr )
    # set train mode
    td3_trainer.q_net1.train()
    td3_trainer.q_net2.train()
    td3_trainer.target_q_net1.train()
    td3_trainer.target_q_net2.train()
    td3_trainer.policy_net.train()
    td3_trainer.target_policy_net.train()

    # training loop
    if mode == 'train':
        frame_idx = 0
        rewards = []
        t0 = time.time()
        for eps in range(train_episodes):
            state = env.reset()
            state = state.astype(np.float32)
            episode_reward = 0
            if frame_idx < 1:
                _ = td3_trainer.policy_net(
                    [state]
                )  # need an extra call here to make inside functions be able to use model.forward
                _ = td3_trainer.target_policy_net([state])

            for step in range(max_steps):
                if frame_idx > explore_steps:
                    action = td3_trainer.policy_net.get_action(state, explore_noise_scale=1.0)
                else:
                    action = td3_trainer.policy_net.sample_action()

                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                env.render()
                done = 1 if done ==True else 0

                replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                frame_idx += 1

                if len(replay_buffer) > batch_size:
                    for i in range(update_itr):
                        td3_trainer.update(batch_size, eval_noise_scale=0.5, reward_scale=1.)

                if done:
                    break

            if eps % int(save_interval) == 0:
                plot(rewards, Algorithm_name='TD3', Env_name=env_id)
                td3_trainer.save_weights()

            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'\
            .format(eps, train_episodes, episode_reward, time.time()-t0 ))
            rewards.append(episode_reward)
        td3_trainer.save_weights()

    if mode == 'test':
        frame_idx = 0
        rewards = []
        t0 = time.time()

        td3_trainer.load_weights()

        for eps in range(test_episodes):
            state = env.reset()
            state = state.astype(np.float32)
            episode_reward = 0
            if frame_idx < 1:
                _ = td3_trainer.policy_net(
                    [state]
                )  # need an extra call to make inside functions be able to use forward
                _ = td3_trainer.target_policy_net([state])

            for step in range(max_steps):
                action = td3_trainer.policy_net.get_action(state, explore_noise_scale=1.0)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                env.render()
                done = 1 if done ==True else 0

                state = next_state
                episode_reward += reward
                frame_idx += 1

                # if frame_idx % 50 == 0:
                #     plot(frame_idx, rewards)

                if done:
                    break
            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'\
            .format(eps, test_episodes, episode_reward, time.time()-t0 ) )
            rewards.append(episode_reward)
