"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.

Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

"""

import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl
from common.buffer import *
from common.networks import *
from common.utils import *

#####################  hyper parameters  ####################

TAU = 0.01  # soft replacement
VAR = 3  # control exploration

###############################  DDPG  ####################################


class QNetwork(Model):
    ''' network for estimating Q(s,a) '''

    def __init__(self, state_dim, state_hidden_list, action_dim, action_hidden_list, hidden_list, scope=None):

        # w_init = tf.keras.initializers.glorot_normal(
        #     seed=None
        # )  # glorot initialization is better than uniform in practice
        # init_w=3e-3
        # w_init = tf.random_uniform_initializer(-init_w, init_w)
        w_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        s = s_input = Input([None, state_dim])
        for each_dim in state_hidden_list:
            s = Dense(
                n_units=each_dim,
                act=tf.nn.relu,
                W_init=w_init,
                b_init=b_init,
            )(s)

        a = a_input = Input([None, action_dim])
        for each_dim in action_hidden_list:
            a = Dense(
                n_units=each_dim,
                act=tf.nn.relu,
                W_init=w_init,
                b_init=b_init,
            )(a)
        x = tl.layers.Concat(1)([s, a])

        for each_dim in hidden_list:
            x = Dense(
                n_units=each_dim,
                act=tf.nn.relu,
                W_init=w_init,
                b_init=b_init,
            )(x)
        outputs = Dense(
            n_units=1,
            W_init=w_init,
        )(x)

        super().__init__(inputs=[s_input, a_input], outputs=outputs)


class DeterministicPolicyNetwork(Model):
    ''' stochastic continuous policy network for generating action according to the state '''

    def __init__(self, state_dim, action_dim, hidden_list, scope=None):
        # w_init = tf.keras.initializers.glorot_normal(
        #     seed=None
        # )  # glorot initialization is better than uniform in practice
        # init_w=3e-3
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

        w_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)
        l = inputs = Input([None, state_dim])
        for each_dim in hidden_list:
            l = Dense(
                n_units=each_dim,
                act=tf.nn.relu,
                W_init=w_init,
                b_init=b_init,
            )(l)
        action_linear = Dense(
            n_units=action_dim,
            act=tf.nn.tanh,
            W_init=w_init,
            b_init=b_init,
        )(l)
        if scope:
            action_linear = tl.layers.Lambda(lambda x: np.array(scope) * x)(action_linear)
        super().__init__(inputs=inputs, outputs=action_linear)


class DDPG(object):
    """
    DDPG class
    """

    def __init__(
            self, a_dim, s_dim, hidden_dim, num_hidden_layer, a_bound, gamma, lr_a, lr_c, replay_buffer_size,
            batch_size=32
    ):
        self.memory = np.zeros((replay_buffer_size, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.actor = DeterministicPolicyNetwork(s_dim, a_dim, [hidden_dim] * num_hidden_layer, a_bound)
        self.critic = QNetwork(s_dim, [], a_dim, [], [2 * hidden_dim] * num_hidden_layer)
        self.actor.train()
        self.critic.train()

        self.actor_target = DeterministicPolicyNetwork(s_dim, a_dim, [hidden_dim] * num_hidden_layer, a_bound)
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        self.critic_target = QNetwork(s_dim, [], a_dim, [], [2 * hidden_dim] * num_hidden_layer)
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(lr_a)
        self.critic_opt = tf.optimizers.Adam(lr_c)

    def ema_update(self):
        """
        Soft updating by exponential smoothing
        :return: None
        """
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        return self.actor(np.array([s], dtype=np.float32))[0]

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(len(self.memory), size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1:-self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + self.gamma * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # maximize the q
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.ema_update()

    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % len(self.memory)  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        save_model(
            self.actor,
            'actor',
            'ddpg',
        )
        save_model(
            self.actor_target,
            'actor_target',
            'ddpg',
        )
        save_model(
            self.critic,
            'critic',
            'ddpg',
        )
        save_model(
            self.critic_target,
            'critic_target',
            'ddpg',
        )

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        load_model(
            self.actor,
            'actor',
            'ddpg',
        )
        load_model(
            self.actor_target,
            'actor_target',
            'ddpg',
        )
        load_model(
            self.critic,
            'critic',
            'ddpg',
        )
        load_model(
            self.critic_target,
            'critic_target',
            'ddpg',
        )


def learn(
        env_id='Pendulum-v0', train_episodes=200, test_episodes=100, max_steps=200, save_interval=10, actor_lr=1e-3,
        critic_lr=2e-3, gamma=0.9, hidden_dim=30, num_hidden_layer=1, seed=1, mode='train', render=False,
        replay_buffer_size=10000, batch_size=32
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
    :param replay_buffer_size: size of replay buffer
    :param batch_size: udpate batchsize
    :return: None
    """

    env = gym.make(env_id)
    env = env.unwrapped

    # reproducible
    env.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_max = env.action_space.high
    a_min = env.action_space.low

    ddpg = DDPG(
        a_dim, s_dim, hidden_dim, num_hidden_layer, a_max, gamma, actor_lr, critic_lr, replay_buffer_size, batch_size
    )

    if mode == 'train':  # train

        reward_buffer = []
        t0 = time.time()
        for i in range(train_episodes):
            t1 = time.time()
            s = env.reset()
            if render:
                env.render()
            ep_reward = 0
            for j in range(max_steps):
                # Add exploration noise
                a = ddpg.choose_action(s)
                a = np.clip(
                    np.random.normal(a, VAR), a_min, a_max
                )  # add randomness to action selection for exploration
                s_, r, done, info = env.step(a)

                ddpg.store_transition(s, a, r / 10, s_)

                if ddpg.pointer > replay_buffer_size:
                    ddpg.learn()

                s = s_
                ep_reward += r
                if j == max_steps - 1:
                    print(
                        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i, train_episodes, ep_reward,
                            time.time() - t1
                        ), end=''
                    )
                plt.show()
            # test
            if i and not i % save_interval:
                t1 = time.time()
                s = env.reset()
                ep_reward = 0
                for j in range(max_steps):

                    a = ddpg.choose_action(s)  # without exploration noise
                    s_, r, done, info = env.step(a)

                    s = s_
                    ep_reward += r
                    if j == max_steps - 1:
                        print(
                            '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                                i, train_episodes, ep_reward,
                                time.time() - t1
                            )
                        )

                        reward_buffer.append(ep_reward)
                ddpg.save_ckpt()
        print('\nRunning time: ', time.time() - t0)

    elif mode is not 'test':
        print('unknow mode type, activate test mode as default')

    # test
    ddpg.load_ckpt()
    for i in range(test_episodes):
        s = env.reset()
        for i in range(max_steps):
            env.render()
            s, r, done, info = env.step(ddpg.choose_action(s))
            if done:
                break


if __name__ == '__main__':
    learn()
