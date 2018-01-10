"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

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
"""

import time

import gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# hyper-parameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 100  # renders environment if running reward is greater then this threshold
MAX_EP_STEPS = 1000             # maximum time step in one episode
RENDER = False   # rendering wastes time
LAMBDA = 0.9     # reward discount in TD error
LR_A = 0.001     # learning rate for actor
LR_C = 0.01      # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(2)     # reproducible
# env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n
# env.action_space.sample() random sample

print("observation dimension: %d" % N_F)                    # 4
print("observation high: %s" % env.observation_space.high)  # [ 2.4 , inf , 0.41887902 , inf]
print("observation low : %s" % env.observation_space.low)   # [-2.4 , -inf , -0.41887902 , -inf]
print("num of actions: %d" % N_A)                           # 2 : left or right

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, [None], "act")
        self.td_error = tf.placeholder(tf.float32, [None], "td_error")  # TD_error

        with tf.variable_scope('Actor'):    # Policy network
            n = InputLayer(self.s, name='in')
            n = DenseLayer(n, n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden')
            # n = DenseLayer(n, n_units=10, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden2')
            n = DenseLayer(n, n_units=n_actions, name='Pi')
            self.acts_logits = n.outputs
            self.acts_prob = tf.nn.softmax(self.acts_logits)

        ## Hao Dong
        with tf.variable_scope('loss'):
            self.exp_v = tl.rein.cross_entropy_reward_loss(logits=self.acts_logits, actions=self.a, rewards=self.td_error, name='actor_weighted_loss')

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.exp_v)

        ## MorvanZhou (the same)
        # with tf.variable_scope('exp_v'):
        #     # log_prob = tf.log(self.acts_prob[0, self.a[0]])
        #     # self.exp_v = tf.reduce_mean(log_prob * self.td_error[0])  # advantage (TD_error) guided loss
        #     self.exp_v = tl.rein.log_weight(probs=self.acts_prob[0, self.a[0]], weights=self.td_error)
        #
        # with tf.variable_scope('train'):
        #     self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        _, exp_v = self.sess.run([self.train_op, self.exp_v], {self.s: [s], self.a: [a], self.td_error: td[0]})
        return exp_v

    def choose_action(self, s):
        probs = self.sess.run(self.acts_prob, {self.s: [s]})   # get probabilities of all actions
        return tl.rein.choice_action_by_probs(probs.ravel())

    def choose_action_greedy(self, s):
        probs = self.sess.run(self.acts_prob, {self.s: [s]})   # get probabilities of all actions
        return np.argmax(probs.ravel())

class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):   # we use Value-function here, not Action-Value-function
            n = InputLayer(self.s, name='in')
            n = DenseLayer(n, n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden')
            # n = DenseLayer(n, n_units=5, act=tf.nn.relu, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden2')
            n = DenseLayer(n, n_units=1, act=tf.identity, name='V')
            self.v = n.outputs

        with tf.variable_scope('squared_TD_error'):
            # TD_error = r + lambd * V(newS) - V(S)
            self.td_error = self.r + LAMBDA * self.v_ - self.v
            self.loss = tf.square(self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        v_ = self.sess.run(self.v, {self.s: [s_]})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: [s], self.v_: v_, self.r: r})
        return td_error

sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

tl.layers.initialize_global_variables(sess)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    episode_time = time.time()
    s = env.reset()
    t = 0       # number of step in this episode
    all_r = []  # rewards of all steps
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_new, r, done, info = env.step(a)

        if done: r = -20
        ## these may helpful in some tasks
        # if abs(s_new[0]) >= env.observation_space.high[0]:
        # #  cart moves more than 2.4 units from the center
        #     r = -20
        # reward for the distance between cart to the center
        # r -= abs(s_new[0])  * .1

        all_r.append(r)

        td_error = critic.learn(s, r, s_new)   # learn Value-function : gradient = grad[r + lambda * V(s_new) - V(s)]
        actor.learn(s, a, td_error)            # learn Policy         : true_gradient = grad[logPi(s, a) * td_error]

        s = s_new
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(all_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            ## start rending if running_reward greater than a threshold
            # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True
            print("Episode: %d reward: %f running_reward %f took: %.5f" %
                (i_episode, ep_rs_sum, running_reward, time.time()-episode_time))

            ## Early Stopping for quick check
            if t >= MAX_EP_STEPS:
                print("Early Stopping")
                s = env.reset()
                rall = 0
                while True:
                    env.render()
                    # a = actor.choose_action(s)
                    a = actor.choose_action_greedy(s)   # Hao Dong: it is important for this task
                    s_new, r, done, info = env.step(a)
                    s_new = np.concatenate((s_new[0:N_F], s[N_F:]), axis=0)
                    rall += r
                    s = s_new
                    if done:
                        print("reward", rall)
                        s = env.reset()
                        rall = 0
            break
