"""
Actor-Critic 
-------------
It uses TD-error as the Advantage.

Actor Critic History
----------------------
A3C > DDPG > AC

Advantage
----------
AC converge faster than Policy Gradient.

Disadvantage (IMPORTANT)
------------------------
The Policy is oscillated (difficult to converge), DDPG can solve
this problem using advantage of DQN.

Reference
----------
paper: https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf
View more on MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Environment
------------
CartPole-v0: https://gym.openai.com/envs/CartPole-v0

A pole is attached by an un-actuated joint to a cart, which moves along a
frictionless track. The system is controlled by applying a force of +1 or -1
to the cart. The pendulum starts upright, and the goal is to prevent it from
falling over.

A reward of +1 is provided for every timestep that the pole remains upright.
The episode ends when the pole is more than 15 degrees from vertical, or the
cart moves more than 2.4 units from the center.


Prerequisites
--------------
tensorflow >=2.0.0a0
tensorlayer >=2.0.0

"""
import argparse
import time

import gym
import numpy as np
import tensorflow as tf

import tensorlayer as tl
from common.buffer import *
from common.networks import *
from common.utils import *
from tensorlayer.models import Model

tl.logging.set_verbosity(tl.logging.DEBUG)

###############################  Actor-Critic  ####################################


class Actor(object):

    def __init__(self, n_features, n_actions, lr, hidden_dim, hidden_layer):

        self.model = DeterministicPolicyNetwork(n_features, n_actions, hidden_dim, hidden_layer).model()
        self.model.train()
        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, s, a, td):
        with tf.GradientTape() as tape:
            _logits = self.model(np.array([s]))
            ## cross-entropy loss weighted by td-error (advantage),
            # the cross-entropy mearsures the difference of two probability distributions: the predicted logits and sampled action distribution,
            # then weighted by the td-error: small difference of real and predict actions for large td-error (advantage); and vice versa.
            _exp_v = tl.rein.cross_entropy_reward_loss(logits=_logits, actions=[a], rewards=td[0])
        grad = tape.gradient(_exp_v, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        return _exp_v

    def choose_action(self, s):
        _logits = self.model(np.array([s]))
        _probs = tf.nn.softmax(_logits).numpy()
        return tl.rein.choice_action_by_probs(_probs.ravel())  # sample according to probability distribution

    def choose_action_greedy(self, s):
        _logits = self.model(np.array([s]))  # logits: probability distribution of actions
        _probs = tf.nn.softmax(_logits).numpy()
        return np.argmax(_probs.ravel())

    def save_ckpt(self):  # save trained weights
        save_model(self.model, 'model_actor', 'AC')

    def load_ckpt(self):  # load trained weights
        load_model(self.model, 'model_actor', 'AC')


class Critic(object):

    def __init__(self, n_features, gamma, lr, hidden_dim, hidden_layer):
        self.GAMMA = gamma

        self.model = ValueNetwork(n_features, hidden_dim, hidden_layer).model()  # from common.networks
        self.model.train()

        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, s, r, s_):
        v_ = self.model(np.array([s_]))
        with tf.GradientTape() as tape:
            v = self.model(np.array([s]))
            ## TD_error = r + lambd * V(newS) - V(S)
            td_error = r + self.GAMMA * v_ - v
            loss = tf.square(td_error)
        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

        return td_error

    def save_ckpt(self):  # save trained weights
        save_model(self.model, 'model_critic', 'AC')

    def load_ckpt(self):  # load trained weights
        load_model(self.model, 'model_critic', 'AC')


def learn(
        env_id, train_episodes, test_episodes=1000, max_steps=1000, gamma=0.9, actor_lr=1e-3, critic_lr=1e-2,
        actor_hidden_dim=30, actor_hidden_layer=1, critic_hidden_dim=30, critic_hidden_layer=1, seed=2,
        save_interval=100, mode='train', render=False
):
    '''
    parameters
    -----------
    env: learning environment
    train_episodes:  total number of episodes for training
    test_episodes:  total number of episodes for testing
    max_steps:  maximum number of steps for one episode
    number_workers: manually set number of workers
    gamma: reward discount factor
    actor_lr: learning rate for actor
    critic_lr: learning rate for critic
    save_interval: timesteps for saving the weights and plotting the results
    mode: train or test

    '''

    env = make_env(env_id)
    env.seed(seed)  # reproducible
    np.random.seed(seed)
    tf.random.set_seed(seed)  # reproducible
    # env = env.unwrapped
    N_F = env.observation_space.shape[0]
    # N_A = env.action_space.shape[0]
    N_A = env.action_space.n

    print("observation dimension: %d" % N_F)  # 4
    print("observation high: %s" % env.observation_space.high)  # [ 2.4 , inf , 0.41887902 , inf]
    print("observation low : %s" % env.observation_space.low)  # [-2.4 , -inf , -0.41887902 , -inf]
    print("num of actions: %d" % N_A)  # 2 : left or right

    actor = Actor(n_features=N_F, n_actions=N_A, lr=actor_lr, hidden_dim=actor_hidden_dim,\
    hidden_layer = actor_hidden_layer)
    # we need a good teacher, so the teacher should learn faster than the actor
    critic = Critic(n_features=N_F, gamma=gamma, lr=critic_lr, hidden_dim=critic_hidden_dim,\
    hidden_layer = critic_hidden_layer)

    if mode == 'train':
        t0 = time.time()
        rewards = []
        for i_episode in range(train_episodes):
            # episode_time = time.time()
            s = env.reset().astype(np.float32)
            t = 0  # number of step in this episode
            all_r = []  # rewards of all steps

            while True:

                if render: env.render()

                a = actor.choose_action(s)

                s_new, r, done, info = env.step(a)
                s_new = s_new.astype(np.float32)

                if done: r = -20

                all_r.append(r)

                td_error = critic.learn(
                    s, r, s_new
                )  # learn Value-function : gradient = grad[r + lambda * V(s_new) - V(s)]
                try:
                    actor.learn(s, a, td_error)  # learn Policy : true_gradient = grad[logPi(s, a) * td_error]
                except KeyboardInterrupt:  # if Ctrl+C at running actor.learn(), then save model, or exit if not at actor.learn()
                    actor.save_ckpt()
                    critic.save_ckpt()
                    # logging

                s = s_new
                t += 1

                if done or t >= max_steps:
                    ep_rs_sum = sum(all_r)

                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

                    rewards.append(running_reward)
                    # start rending if running_reward greater than a threshold
                    # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True
                    # print("Episode: %d reward: %f running_reward %f took: %.5f" % \
                    #     (i_episode, ep_rs_sum, running_reward, time.time() - episode_time))
                    print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'\
                    .format(i_episode, train_episodes, ep_rs_sum, time.time()-t0 ))

                    # Early Stopping for quick check
                    if t >= max_steps:
                        print("Early Stopping")
                        s = env.reset().astype(np.float32)
                        rall = 0
                        while True:
                            env.render()
                            # a = actor.choose_action(s)
                            a = actor.choose_action_greedy(s)  # Hao Dong: it is important for this task
                            s_new, r, done, info = env.step(a)
                            s_new = np.concatenate((s_new[0:N_F], s[N_F:]), axis=0).astype(np.float32)
                            rall += r
                            s = s_new
                            if done:
                                # print("reward", rall)
                                s = env.reset().astype(np.float32)
                                rall = 0
                    break

            if i_episode % save_interval == 0:
                actor.save_ckpt()
                critic.save_ckpt()
                plot(rewards, Algorithm_name='AC', Env_name=env_id)
        actor.save_ckpt()
        critic.save_ckpt()

    if mode == 'test':
        actor.load_ckpt()
        critic.load_ckpt()
        t0 = time.time()

        for i_episode in range(test_episodes):
            episode_time = time.time()
            s = env.reset().astype(np.float32)
            t = 0  # number of step in this episode
            all_r = []  # rewards of all steps
            while True:
                if render: env.render()
                a = actor.choose_action(s)
                s_new, r, done, info = env.step(a)
                s_new = s_new.astype(np.float32)
                if done: r = -20

                all_r.append(r)
                s = s_new
                t += 1

                if done or t >= max_steps:
                    ep_rs_sum = sum(all_r)

                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                    # start rending if running_reward greater than a threshold
                    # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True
                    # print("Episode: %d reward: %f running_reward %f took: %.5f" % \
                    #     (i_episode, ep_rs_sum, running_reward, time.time() - episode_time))
                    print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'\
                    .format(i_episode, test_episodes, ep_rs_sum, time.time()-t0 ))

                    # Early Stopping for quick check
                    if t >= max_steps:
                        print("Early Stopping")
                        s = env.reset().astype(np.float32)
                        rall = 0
                        while True:
                            env.render()
                            # a = actor.choose_action(s)
                            a = actor.choose_action_greedy(s)  # Hao Dong: it is important for this task
                            s_new, r, done, info = env.step(a)
                            s_new = np.concatenate((s_new[0:N_F], s[N_F:]), axis=0).astype(np.float32)
                            rall += r
                            s = s_new
                            if done:
                                # print("reward", rall)
                                s = env.reset().astype(np.float32)
                                rall = 0
                    break
