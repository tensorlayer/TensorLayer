"""
Asynchronous Advantage Actor Critic (A3C) with Continuous Action Space.

Actor Critic History
----------------------
A3C > DDPG (for continuous action space) > AC

Advantage
----------
Train faster and more stable than AC.

Disadvantage
-------------
Have bias.

Reference
----------
Original Paper: https://arxiv.org/pdf/1602.01783.pdf
MorvanZhou's tutorial: https://morvanzhou.github.io/tutorials/
MorvanZhou's code: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/experiments/Solve_BipedalWalker/A3C.py

Environment
-----------
BipedalWalker-v2 : https://gym.openai.com/envs/BipedalWalker-v2

Reward is given for moving forward, total 300+ points up to the far end.
If the robot falls, it gets -100. Applying motor torque costs a small amount of
points, more optimal agent will get better score. State consists of hull angle
speed, angular velocity, horizontal speed, vertical speed, position of joints
and joints angular speed, legs contact with ground, and 10 lidar rangefinder
measurements. There's no coordinates in the state vector.

Prerequisites
--------------
tensorflow 2.0.0a0
tensorflow-probability 0.6.0
tensorlayer 2.0.0
&&
pip install box2d box2d-kengz --user

"""

import argparse
import multiprocessing
import threading
import time

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import tensorlayer as tl
from common.buffer import *
from common.networks import *
from common.utils import *
from tensorlayer.layers import DenseLayer, InputLayer
from tensorlayer.models import Model

tfd = tfp.distributions

# tl.logging.set_verbosity(tl.logging.DEBUG)


###################  Asynchronous Advantage Actor Critic (A3C)  ####################################
class ACNet(object):

    def __init__(
            self, scope, entropy_beta, action_dim, state_dim, actor_hidden_dim, actor_hidden_layer, critic_hidden_dim,
            critic_hidden_layer, action_bound, globalAC=None
    ):
        self.scope = scope  # the scope is for naming networks for each worker differently
        self.save_path = './model'
        self.ENTROPY_BETA = entropy_beta
        self.N_A = action_dim
        self.N_S = state_dim
        self.A_BOUND = action_bound

        # w_init = tf.keras.initializers.glorot_normal(seed=None)  # initializer, glorot=xavier
        # def get_actor(input_shape):  # policy network
        #     with tf.name_scope(self.scope):
        #         ni = tl.layers.Input(input_shape, name='in')
        #         nn = tl.layers.Dense(n_units=500, act=tf.nn.relu6, W_init=w_init, name='la')(ni)
        #         nn = tl.layers.Dense(n_units=300, act=tf.nn.relu6, W_init=w_init, name='la2')(nn)
        #         mu = tl.layers.Dense(n_units=self.N_A, act=tf.nn.tanh, W_init=w_init, name='mu')(nn)
        #         sigma = tl.layers.Dense(n_units=self.N_A, act=tf.nn.softplus, W_init=w_init, name='sigma')(nn)
        #     return tl.models.Model(inputs=ni, outputs=[mu, sigma], name=scope + '/Actor')

        # self.actor = get_actor([None, self.N_S])
        self.actor = StochasticPolicyNetwork(
            self.N_S, self.N_A, actor_hidden_dim, actor_hidden_layer, scope=self.scope
        ).model()  # call the network model in common functions
        self.actor.train()  # train mode for Dropout, BatchNorm

        # def get_critic(input_shape):  # we use Value-function here, but not Q-function.
        #     with tf.name_scope(self.scope):
        #         ni = tl.layers.Input(input_shape, name='in')
        #         nn = tl.layers.Dense(n_units=500, act=tf.nn.relu6, W_init=w_init, name='lc')(ni)
        #         nn = tl.layers.Dense(n_units=300, act=tf.nn.relu6, W_init=w_init, name='lc2')(nn)
        #         v = tl.layers.Dense(n_units=1, W_init=w_init, name='v')(nn)
        #     return tl.models.Model(inputs=ni, outputs=v, name=scope + '/Critic')

        # self.critic = get_critic([None, self.N_S])
        self.critic = ValueNetwork(self.N_S, critic_hidden_dim, critic_hidden_layer,
                                   scope=self.scope).model()  # call the network model in common functions
        self.critic.train()  # train mode for Dropout, BatchNorm

    @tf.function  # convert numpy functions to tf.Operations in the TFgraph, return tensor
    def update_global(
            self, buffer_s, buffer_a, buffer_v_target, globalAC
    ):  # refer to the global Actor-Crtic network for updating it with samples
        ''' update the global critic '''
        with tf.GradientTape() as tape:
            self.v = self.critic(buffer_s)
            self.v_target = buffer_v_target
            td = tf.subtract(self.v_target, self.v, name='TD_error')
            self.c_loss = tf.reduce_mean(tf.square(td))
        self.c_grads = tape.gradient(self.c_loss, self.critic.trainable_weights)
        OPT_C.apply_gradients(zip(self.c_grads, globalAC.critic.trainable_weights))  # local grads applies to global net
        # del tape # Drop the reference to the tape
        ''' update the global actor '''
        with tf.GradientTape() as tape:
            self.mu, self.sigma = self.actor(buffer_s)
            self.test = self.sigma[0]
            self.mu, self.sigma = self.mu * self.A_BOUND[1], self.sigma + 1e-5

            normal_dist = tfd.Normal(self.mu, self.sigma)  # no tf.contrib for tf2.0
            self.a_his = buffer_a  # float32
            log_prob = normal_dist.log_prob(self.a_his)
            exp_v = log_prob * td  # td is from the critic part, no gradients for it
            entropy = normal_dist.entropy()  # encourage exploration
            self.exp_v = self.ENTROPY_BETA * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)
        self.a_grads = tape.gradient(self.a_loss, self.actor.trainable_weights)
        OPT_A.apply_gradients(zip(self.a_grads, globalAC.actor.trainable_weights))  # local grads applies to global net
        return self.test  # for test purpose

    @tf.function
    def pull_global(self, globalAC):  # run by a local, pull weights from the global nets
        for l_p, g_p in zip(self.actor.trainable_weights, globalAC.actor.trainable_weights):
            l_p.assign(g_p)
        for l_p, g_p in zip(self.critic.trainable_weights, globalAC.critic.trainable_weights):
            l_p.assign(g_p)

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        self.mu, self.sigma = self.actor(s)

        with tf.name_scope('wrap_a_out'):
            self.mu, self.sigma = self.mu * self.A_BOUND[1], self.sigma + 1e-5
        normal_dist = tfd.Normal(self.mu, self.sigma)  # for continuous action space
        self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *self.A_BOUND)
        return self.A.numpy()[0]

    def save_ckpt(self):  # save trained weights
        save_model(self.actor, 'model_actor', 'A3C')
        save_model(self.critic, 'model_critic', 'A3C')

    def load_ckpt(self):  # load trained weights
        load_model(self.actor, 'model_actor', 'A3C')
        load_model(self.critic, 'model_critic', 'A3C')


class Worker(object):

    def __init__(
            self, env_id, name, globalAC, train_episodes, gamma, update_itr, entropy_beta, action_dim, state_dim,
            actor_hidden_dim, actor_hidden_layer, critic_hidden_dim, critic_hidden_layer, action_bound
    ):
        self.env = make_env(env_id)
        self.name = name
        self.AC = ACNet(
            name, entropy_beta, action_dim, state_dim, actor_hidden_dim, actor_hidden_layer, critic_hidden_dim,
            critic_hidden_layer, action_bound, globalAC
        )
        self.MAX_GLOBAL_EP = train_episodes
        self.UPDATE_GLOBAL_ITER = update_itr
        self.GAMMA = gamma

    # def work(self):
    def work(self, globalAC):
        global COORD, GLOBAL_RUNNING_R, GLOBAL_EP, OPT_A, OPT_C, t0
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < self.MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                # visualize Worker_0 during training
                if self.name == 'Worker_0' and total_step % 30 == 0:
                    self.env.render()
                s = s.astype('float32')  # double to float
                a = self.AC.choose_action(s)
                s_, r, done, _info = self.env.step(a)

                s_ = s_.astype('float32')  # double to float
                # set robot falls reward to -2 instead of -100
                if r == -100: r = -2

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % self.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net

                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.AC.critic(s_[np.newaxis, :])[0, 0]  # reduce dim from 2 to 0

                    buffer_v_target = []

                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + self.GAMMA * v_s_
                        buffer_v_target.append(v_s_)

                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = (
                        np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    )
                    # update gradients on global network
                    self.AC.update_global(buffer_s, buffer_a, buffer_v_target.astype('float32'), globalAC)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    # update local network from global network
                    self.AC.pull_global(globalAC)

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:  # moving average
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)

                    print('{}, Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'\
                    .format(self.name, GLOBAL_EP, self.MAX_GLOBAL_EP, ep_r, time.time()-t0 ))
                    GLOBAL_EP += 1
                    break


def learn(
        env_id, train_episodes, test_episodes=1000, max_steps=150, number_workers=0, update_itr=10, gamma=0.99,
        entropy_beta=0.005, actor_lr=5e-5, critic_lr=1e-4, actor_hidden_dim=300, actor_hidden_layer=2,
        critic_hidden_dim=300, critic_hidden_layer=2, seed=2, save_interval=500, mode='train'
):
    '''
    parameters
    -----------
    env: learning environment
    train_episodes:  total number of episodes for training
    test_episodes:  total number of episodes for testing
    max_steps:  maximum number of steps for one episode
    number_workers: manually set number of workers
    update_itr: update global policy after several episodes
    gamma: reward discount factor
    entropy_beta: factor for entropy boosted exploration
    actor_lr: learning rate for actor
    critic_lr: learning rate for critic
    mode: train or test

    '''
    global COORD, GLOBAL_RUNNING_R, GLOBAL_EP, OPT_A, OPT_C, t0
    COORD = tf.train.Coordinator()
    GLOBAL_NET_SCOPE = 'Global_Net'
    GLOBAL_RUNNING_R = []
    GLOBAL_EP = 0  # will increase during training, stop training when it >= MAX_GLOBAL_EP
    N_WORKERS = number_workers if number_workers > 0 else multiprocessing.cpu_count()

    env = make_env(env_id)
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.shape[0]

    np.random.seed(seed)
    tf.random.set_seed(seed)  # reproducible

    A_BOUND = [env.action_space.low, env.action_space.high]
    A_BOUND[0] = A_BOUND[0].reshape(1, N_A)
    A_BOUND[1] = A_BOUND[1].reshape(1, N_A)
    # print(A_BOUND)
    if mode == 'train':
        # ============================= TRAINING ===============================
        t0 = time.time()
        with tf.device("/cpu:0"):

            OPT_A = tf.optimizers.RMSprop(actor_lr, name='RMSPropA')
            OPT_C = tf.optimizers.RMSprop(critic_lr, name='RMSPropC')

            GLOBAL_AC = ACNet(
                GLOBAL_NET_SCOPE, entropy_beta, N_A, N_S, actor_hidden_dim, actor_hidden_layer, critic_hidden_dim,
                critic_hidden_layer, A_BOUND
            )  # we only need its params
            workers = []
            # Create worker
            for i in range(N_WORKERS):
                i_name = 'Worker_%i' % i  # worker name
                workers.append(
                    Worker(
                        env_id, i_name, GLOBAL_AC, train_episodes, gamma, update_itr, entropy_beta, N_A, N_S,
                        actor_hidden_dim, actor_hidden_layer, critic_hidden_dim, critic_hidden_layer, A_BOUND
                    )
                )

        # start TF threading
        worker_threads = []
        for worker in workers:
            # t = threading.Thread(target=worker.work)
            job = lambda: worker.work(GLOBAL_AC)
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)
        import matplotlib.pyplot as plt
        plt.plot(GLOBAL_RUNNING_R)
        plt.xlabel('episode')
        plt.ylabel('global running reward')
        plt.savefig('a3c.png')
        plt.show()

        GLOBAL_AC.save_ckpt()

    if mode == 'test':
        # ============================= EVALUATION =============================
        GLOBAL_AC.load_ckpt()
        frame_idx = 0
        for eps in range(test_episodes):
            s = env.reset()
            rall = 0
            for step in range(max_steps):
                env.render()
                frame_idx += 1
                s = s.astype('float32')  # double to float
                a = GLOBAL_AC.choose_action(s)
                s, r, d, _ = env.step(a)
                rall += r
                if d:
                    print("reward", rall)
                    break
