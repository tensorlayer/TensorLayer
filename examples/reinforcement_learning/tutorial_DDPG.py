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

Env
---
Openai Gym Pendulum-v0, continual action space

To run
------
python *.py

"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np

#####################  hyper parameters  ####################

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

###############################  DDPG  ####################################


class DDPG(object):
    '''
    DDPG class
    '''
    def __init__(self, a_dim, s_dim, a_bound, ):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def get_actor(input_state_shape, name=''):
            '''
            Build actor network
            :param input_state_shape: state
            :return: act
            '''
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            # x = tl.layers.Lambda(lambda x: np.array(a_bound)*x)(x)
            # x = tf.multiply(x, a_bound, name='A_scaled_a')
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)

        def get_critic(input_state_shape, input_action_shape, name=''):
            '''
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :return: Q value Q(s,a)
            '''
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        def copy_para(from_model, to_model):
            '''
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            '''
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        self.critic_target = get_critic([None, s_dim], [None, a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

    def ema_update(self):
        '''
        Soft updating by exponential smoothing
        :return: None
        '''
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def choose_action(self, s):
        '''
        Choose action
        :param s: state
        :return: act
        '''
        return self.actor(np.array([s], dtype=np.float32))[0]

    def learn(self):
        '''
        Update parameters
        :return: None
        '''
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + GAMMA * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = - tf.reduce_mean(q)  # maximize the q
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.ema_update()

    def store_transition(self, s, a, r, s_):
        '''
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        '''
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        tl.files.save_npz(self.actor.trainable_weights, name='model/actor.npz')
        tl.files.save_npz(self.actor_target.trainable_weights, name='model/actor_target.npz')
        tl.files.save_npz(self.critic.trainable_weights, name='model/critic.npz')
        tl.files.save_npz(self.critic_target.trainable_weights, name='model/critic_target.npz')

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_and_assign_npz(name='model/actor.npz', network=self.actor)
        tl.files.load_and_assign_npz(name='model/actor_target.npz', network=self.actor_target)
        tl.files.load_and_assign_npz(name='model/critic.npz', network=self.critic)
        tl.files.load_and_assign_npz(name='model/critic_target.npz', network=self.critic_target)


if __name__ == '__main__':
    import gym
    import time
    import matplotlib.pyplot as plt

    MAX_EPISODES = 200
    MAX_EP_STEPS = 200
    TEST_PER_EPISODES = 10
    ENV_NAME = 'Pendulum-v0'

    ###############################  training  ####################################

    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound)

    var = 3  # control exploration
    reward_buffer = []
    t0 = time.time()
    for i in range(MAX_EPISODES):
        t1 = time.time()
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                # var *= .9995  # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print("\rEpisode [%d/%d] \tReward: %i \tExplore: %.2f \ttook: %.5fs " %
                      (i, MAX_EPISODES, ep_reward, var, time.time() - t1), end='')

        # test
        if i and not i % TEST_PER_EPISODES:
            t1 = time.time()
            s = env.reset()
            ep_reward = 0
            for j in range(MAX_EP_STEPS):

                a = ddpg.choose_action(s)       # without exploration noise
                s_, r, done, info = env.step(a)

                s = s_
                ep_reward += r
                if j == MAX_EP_STEPS - 1:
                    print("\rEpisode [%d/%d] \tReward: %i \tExplore: %.2f \ttook: %.5fs " %
                          (i, MAX_EPISODES, ep_reward, var, time.time() - t1))

                    reward_buffer.append(ep_reward)

        if reward_buffer:
            plt.ion()
            plt.title('DDPG')
            plt.plot(np.array(range(len(reward_buffer)))*TEST_PER_EPISODES, reward_buffer)  # plot the episode vt
            plt.xlabel('episode steps')
            plt.ylabel('normalized state-action value')
            plt.ylim(-2000, 0)
            plt.show()
            plt.pause(0.1)
            plt.cla()
            plt.ioff()

    print('\nRunning time: ', time.time() - t0)
    s = env.reset()
    while True:
        s = env.reset()
        for i in range(MAX_EP_STEPS):
            env.render()
            a = ddpg.choose_action(s)
            s_, r, done, info = env.step(a)
            if done:
                break
            s = s_
