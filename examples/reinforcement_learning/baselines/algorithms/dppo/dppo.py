"""
Distributed Proximal Policy Optimization (DPPO)
----------------------------
A distributed version of OpenAI's Proximal Policy Optimization (PPO).
Workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

Reference
---------
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

"""

import argparse
import os
import queue
import threading
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import tensorlayer as tl
from common.buffer import *
from common.networks import *
from common.utils import *

EPS = 1e-8  # epsilon
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
][1]  # choose the method for optimization

###############################  PPO  ####################################


class StochasticPolicyNetwork(Model):
    ''' stochastic continuous policy network for generating action according to the state '''

    def __init__(
            self, state_dim, action_dim, hidden_list, a_bound, log_std_min=-20, log_std_max=2, scope=None,
            trainable=True
    ):

        # w_init = tf.keras.initializers.glorot_normal(
        #     seed=None
        # )  # glorot initialization is better than uniform in practice
        # init_w=3e-3
        # w_init = tf.random_uniform_initializer(-init_w, init_w)
        w_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        l = inputs = Input([None, state_dim])
        for each_dim in hidden_list:
            # l = Dense(n_units=each_dim, act=tf.nn.relu, W_init=w_init, b_init=b_init,)(l)
            l = Dense(n_units=each_dim, act=tf.nn.relu)(l)
        # mean_linear = Dense(n_units=action_dim, act=tf.nn.tanh, W_init=w_init, b_init=b_init,)(l)
        mean_linear = Dense(n_units=action_dim, act=tf.nn.tanh)(l)
        mu = tl.layers.Lambda(lambda x: x * a_bound)(mean_linear)

        # log_std_linear = Dense(n_units=action_dim, act=tf.nn.softplus, W_init=w_init, b_init=b_init,)(l)
        log_std_linear = Dense(
            n_units=action_dim,
            act=tf.nn.softplus,
        )(l)

        # log_std_linear = tl.layers.Lambda(lambda x: tf.clip_by_value(x, log_std_min, log_std_max))(log_std_linear)
        super().__init__(inputs=inputs, outputs=[mu, log_std_linear])
        if trainable:
            self.train()
        else:
            self.eval()


class ValueNetwork(Model):
    '''
    network for estimating V(s),
    one input layer, one output layer, others are hidden layers.
    '''

    def __init__(self, state_dim, hidden_list, trainable=True):

        # w_init = tf.keras.initializers.glorot_normal(
        #     seed=None
        # )  # glorot initialization is better than uniform in practice
        # init_w=3e-3
        # w_init = tf.random_uniform_initializer(-init_w, init_w)
        # w_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        # b_init = tf.constant_initializer(0.1)

        l = inputs = Input([None, state_dim], tf.float32)
        for each_dim in hidden_list:
            l = Dense(n_units=each_dim, act=tf.nn.relu)(l)
        outputs = Dense(n_units=1)(l)

        super().__init__(inputs=inputs, outputs=outputs)
        if trainable:
            self.train()
        else:
            self.eval()


###############################  DPPO  ####################################


class PPO(object):
    '''
    PPO class
    '''

    def __init__(
            self, a_dim, s_dim, hidden_list, a_max, actor_lr, critic_lr, a_update_steps, c_update_steps, save_interval
    ):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.bound = a_max
        self.a_update_steps = a_update_steps
        self.c_update_steps = c_update_steps
        self.save_interval = save_interval

        self.critic = ValueNetwork(s_dim, hidden_list)

        self.actor = StochasticPolicyNetwork(s_dim, a_dim, hidden_list, a_max, trainable=True)
        self.actor_old = StochasticPolicyNetwork(s_dim, a_dim, hidden_list, a_max, trainable=False)
        self.actor_opt = tf.optimizers.Adam(actor_lr)
        self.critic_opt = tf.optimizers.Adam(critic_lr)

    def a_train(self, tfs, tfa, tfadv):
        '''
        Update policy network
        :param tfs: state
        :param tfa: act
        :param tfadv: advantage
        :return:
        '''
        tfs = np.array(tfs, np.float32)
        tfa = np.array(tfa, np.float32)
        tfadv = np.array(tfadv, np.float32)
        with tf.GradientTape() as tape:
            mu, sigma = self.actor(tfs)
            pi = tfp.distributions.Normal(mu, sigma)

            mu_old, sigma_old = self.actor_old(tfs)
            oldpi = tfp.distributions.Normal(mu_old, sigma_old)

            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            ratio = pi.prob(tfa) / (oldpi.prob(tfa) + EPS)
            surr = ratio * tfadv
            if METHOD['name'] == 'kl_pen':
                tflam = METHOD['lam']
                kl = tfp.distributions.kl_divergence(oldpi, pi)
                kl_mean = tf.reduce_mean(kl)
                aloss = -(tf.reduce_mean(surr - tflam * kl))
            else:  # clipping method, find this is better
                aloss = -tf.reduce_mean(
                    tf.minimum(surr,
                               tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * tfadv)
                )
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)

        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if METHOD['name'] == 'kl_pen':
            return kl_mean

    def update_old_pi(self):
        '''
        Update old policy parameter
        :return: None
        '''
        for p, oldp in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldp.assign(p)

    def c_train(self, tfdc_r, s):
        '''
        Update actor network
        :param tfdc_r: cumulative reward
        :param s: state
        :return: None
        '''
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = tfdc_r - self.critic(s)
            closs = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(closs, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def cal_adv(self, tfs, tfdc_r):
        '''
        Calculate advantage
        :param tfs: state
        :param tfdc_r: cumulative reward
        :return: advantage
        '''
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        advantage = tfdc_r - self.critic(tfs)
        return advantage.numpy()

    def update(self):
        '''
        Update parameter with the constraint of KL divergent
        :return: None
        '''
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()  # wait until get batch of data
                self.update_old_pi()  # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
                data = np.vstack(data)

                s, a, r = data[:, :self.s_dim].astype(np.float32), \
                          data[:, self.s_dim: self.s_dim + self.a_dim].astype(np.float32), \
                          data[:, -1:].astype(np.float32)

                adv = self.cal_adv(s, r)
                # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

                # update actor
                if METHOD['name'] == 'kl_pen':
                    for _ in range(self.a_update_steps):
                        kl = self.a_train(s, a, adv)
                        if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                            break
                    if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                        METHOD['lam'] /= 2
                    elif kl > METHOD['kl_target'] * 1.5:
                        METHOD['lam'] *= 2

                    # sometimes explode, this clipping is MorvanZhou's solution
                    METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)

                else:  # clipping method, find this is better (OpenAI's paper)
                    for _ in range(self.a_update_steps):
                        self.a_train(s, a, adv)

                # update critic
                for _ in range(self.c_update_steps):
                    self.c_train(r, s)

                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available

                if GLOBAL_EP and not GLOBAL_EP % self.save_interval:
                    self.save_ckpt()

    def choose_action(self, s):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''
        s = s[np.newaxis, :].astype(np.float32)
        mu, sigma = self.actor(s)
        pi = tfp.distributions.Normal(mu, sigma)
        a = tf.squeeze(pi.sample(1), axis=0)[0]  # choosing action
        return np.clip(a, -self.bound, self.bound)

    def get_v(self, s):
        '''
        Compute value
        :param s: state
        :return: value
        '''
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.critic(s)[0, 0]

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        save_model(
            self.actor,
            'actor',
            'ppo',
        )
        save_model(
            self.actor_old,
            'actor_old',
            'ppo',
        )
        save_model(
            self.critic,
            'critic',
            'ppo',
        )

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        load_model(
            self.actor,
            'actor',
            'ppo',
        )
        load_model(
            self.actor_old,
            'actor_old',
            'ppo',
        )
        load_model(
            self.critic,
            'critic',
            'ppo',
        )


class Worker(object):
    '''
    Worker class for distributional running
    '''

    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.env.seed(wid * 100 + RANDOMSEED)
        self.ppo = GLOBAL_PPO

    def work(self):
        '''
        Define a worker
        :return: None
        '''
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            t0 = time.time()
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # stop collecting data
                        UPDATE_EVENT.set()  # globalPPO update

                    if GLOBAL_EP >= EP_MAX:  # stop training
                        COORD.request_stop()
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1

            print(
                'Episode: {}/{}  | Worker: {} | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    GLOBAL_EP, EP_MAX, self.wid, ep_r,
                    time.time() - t0
                )
            )


def learn(
        env_id='Pendulum-v0', train_episodes=1000, test_episodes=100, max_steps=200, save_interval=10, actor_lr=1e-4,
        critic_lr=2e-4, gamma=0.9, hidden_dim=100, num_hidden_layer=1, seed=1, mode='train', batch_size=32,
        a_update_steps=10, c_update_steps=10, n_worker=4
):
    """
    learn function
    :param env_id: learning environment
    :param train_episodes: total number of episodes for training
    :param test_episodes: total number of episodes for testing
    :param max_steps:  maximum number of steps for one episode
    :param save_interval: timesteps for saving
    :param actor_lr: actor learning rate
    :param critic_lr: critic learning rate
    :param gamma: reward discount factor
    :param hidden_dim: dimension for each hidden layer
    :param num_hidden_layer: number of hidden layer
    :param seed: random seed
    :param mode: train or test
    :param batch_size: udpate batchsize
    :param a_update_steps: actor update iteration steps
    :param c_update_steps: critic update iteration steps
    :param n_worker: number of workers
    :return: None
    """

    env = gym.make(env_id).unwrapped
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_max = env.action_space.high

    global GLOBAL_PPO, UPDATE_EVENT, ROLLING_EVENT, GLOBAL_UPDATE_COUNTER, GLOBAL_EP, GLOBAL_RUNNING_R, COORD, QUEUE
    global GAME, RANDOMSEED, EP_LEN, MIN_BATCH_SIZE, GAMMA, EP_MAX
    GAME, RANDOMSEED, EP_LEN, MIN_BATCH_SIZE, GAMMA, EP_MAX = env_id, seed, max_steps, batch_size, gamma, train_episodes
    GLOBAL_PPO = PPO(
        a_dim, s_dim, [hidden_dim] * num_hidden_layer, a_max, actor_lr, critic_lr, a_update_steps, c_update_steps,
        save_interval
    )

    if mode == 'train':  # train
        UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
        UPDATE_EVENT.clear()  # not update now
        ROLLING_EVENT.set()  # start to roll out
        workers = [Worker(wid=i) for i in range(n_worker)]

        GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
        GLOBAL_RUNNING_R = []
        COORD = tf.train.Coordinator()
        QUEUE = queue.Queue()  # workers putting data in this queue
        threads = []
        for worker in workers:  # worker threads
            t = threading.Thread(target=worker.work, args=())
            t.start()  # training
            threads.append(t)
        # add a PPO updating thread
        threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
        threads[-1].start()
        COORD.join(threads)

        GLOBAL_PPO.save_ckpt()

        # plot reward change
        plot(GLOBAL_RUNNING_R, 'DPPO', env_id)

    elif mode is not 'test':
        print('unknow mode type, activate test mode as default')

    # test
    GLOBAL_PPO.load_ckpt()
    env = gym.make(env_id)
    for _ in range(test_episodes):
        s = env.reset()
        for t in range(EP_LEN):
            env.render()
            s, r, done, info = env.step(GLOBAL_PPO.choose_action(s))
            if done:
                break


if __name__ == '__main__':
    learn()
