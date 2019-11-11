"""
Proximal Policy Optimization (PPO)
----------------------------
A simple version of Proximal Policy Optimization (PPO) using single thread.
PPO is a family of first-order methods that use a few other tricks to keep new policies close to old.
PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

Reference
---------
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

"""
import argparse
import os
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

        l = inputs = Input([None, state_dim], name='policy_input')
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


class PPO(object):
    '''
    PPO class
    '''

    def __init__(self, a_dim, s_dim, hidden_list, a_max, actor_lr, critic_lr, a_update_steps, c_update_steps):
        self.bound = a_max
        self.a_update_steps = a_update_steps
        self.c_update_steps = c_update_steps

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
            v = self.critic(s)
            advantage = tfdc_r - v
            closs = tf.reduce_mean(tf.square(advantage))
        # print('tfdc_r value', tfdc_r)
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

    def update(self, s, a, r):
        '''
        Update parameter with the constraint of KL divergent
        :param s: state
        :param a: act
        :param r: reward
        :return: None
        '''
        s, a, r = s.astype(np.float32), a.astype(np.float32), r.astype(np.float32)

        self.update_old_pi()
        adv = self.cal_adv(s, r)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful

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
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution
        else:  # clipping method, find this is better (OpenAI's paper)
            for _ in range(self.a_update_steps):
                self.a_train(s, a, adv)

        # update critic
        for _ in range(self.c_update_steps):
            self.c_train(r, s)

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


#####################  hyper parameters  ####################

EPS = 1e-8  # epsilon
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
][1]  # choose the method for optimization


def learn(
        env_id='Pendulum-v0', train_episodes=1000, test_episodes=100, max_steps=200, save_interval=10, actor_lr=1e-4,
        critic_lr=2e-4, gamma=0.9, hidden_dim=100, num_hidden_layer=1, seed=1, mode='train', render=False,
        batch_size=32, a_update_steps=10, c_update_steps=10
):
    """
    learn function
    :param env_id: learning environment
    :param train_episodes: total number of episodes for training
    :param test_episodes: total number of episodes for testing
    :param max_steps: maximum number of steps for one episode
    :param save_interval: timesteps for saving
    :param actor_lr: actor learning rate
    :param critic_lr: critic learning rate
    :param gamma: reward discount factor
    :param hidden_dim: dimension for each hidden layer
    :param num_hidden_layer: number of hidden layer
    :param seed: random seed
    :param mode: train or test
    :param render: render each step
    :param batch_size: udpate batchsize
    :param a_update_steps: actor update iteration steps
    :param c_update_steps: critic update iteration steps
    :return: None
    """

    env = gym.make(env_id).unwrapped

    # reproducible
    env.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_max = env.action_space.high
    a_min = env.action_space.low

    ppo = PPO(a_dim, s_dim, [hidden_dim] * num_hidden_layer, a_max, actor_lr, critic_lr, a_update_steps, c_update_steps)

    if mode == 'train':
        all_ep_r = []
        for ep in range(train_episodes):
            s = env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            t0 = time.time()
            for t in range(max_steps):  # in one episode
                if render:
                    env.render()
                a = ppo.choose_action(s)
                s_, r, done, _ = env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
                s = s_
                ep_r += r

                # update ppo
                if (t + 1) % batch_size == 0 or t == max_steps - 1:
                    v_s_ = ppo.get_v(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    ppo.update(bs, ba, br)
            if ep == 0:
                all_ep_r.append(ep_r)
            else:
                all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
            print(
                'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    ep, train_episodes, ep_r,
                    time.time() - t0
                )
            )

            if ep and not ep % save_interval:
                ppo.save_ckpt()
        plot(all_ep_r, 'PPO', env_id)

    elif mode is not 'test':
        print('unknow mode type, activate test mode as default')
    # test
    ppo.load_ckpt()
    for _ in range(test_episodes):
        s = env.reset()
        for i in range(max_steps):
            env.render()
            s, r, done, _ = env.step(ppo.choose_action(s))
            if done:
                break


if __name__ == '__main__':
    learn()
