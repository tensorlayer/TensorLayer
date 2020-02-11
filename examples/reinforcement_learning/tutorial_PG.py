"""
Vanilla Policy Gradient(VPG or REINFORCE)
-----------------------------------------
The policy gradient algorithm works by updating policy parameters via stochastic gradient ascent on policy performance.
It's an on-policy algorithm can be used for environments with either discrete or continuous action spaces.
Here is an example on discrete action space game CartPole-v0.
To apply it on continuous action space, you need to change the last softmax layer and the get_action function.

Reference
---------
Cookbook: Barto A G, Sutton R S. Reinforcement Learning: An Introduction[J]. 1998.
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Environment
-----------
Openai Gym CartPole-v0, discrete action space

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

To run
------
python tutorial_PG.py --train/test

"""
import argparse
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'CartPole-v1'  # environment id
RANDOM_SEED = 1  # random seed, can be either an int number or None
RENDER = False  # render while training

ALG_NAME = 'PG'
TRAIN_EPISODES = 200
TEST_EPISODES = 10
MAX_STEPS = 500

###############################  PG  ####################################


class PolicyGradient:
    """
    PG class
    """

    def __init__(self, state_dim, action_num, learning_rate=0.02, gamma=0.99):
        self.gamma = gamma

        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []

        input_layer = tl.layers.Input([None, state_dim], tf.float32)
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.tanh, W_init=tf.random_normal_initializer(mean=0, stddev=0.3),
            b_init=tf.constant_initializer(0.1)
        )(input_layer)
        all_act = tl.layers.Dense(
            n_units=action_num, act=None, W_init=tf.random_normal_initializer(mean=0, stddev=0.3),
            b_init=tf.constant_initializer(0.1)
        )(layer)

        self.model = tl.models.Model(inputs=input_layer, outputs=all_act)
        self.model.train()
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def get_action(self, s, greedy=False):
        """
        choose action with probabilities.
        :param s: state
        :param greedy: choose action greedy or not
        :return: act
        """
        _logits = self.model(np.array([s], np.float32))
        _probs = tf.nn.softmax(_logits).numpy()
        if greedy:
            return np.argmax(_probs.ravel())
        return tl.rein.choice_action_by_probs(_probs.ravel())

    def store_transition(self, s, a, r):
        """
        store data in memory buffer
        :param s: state
        :param a: act
        :param r: reward
        :return:
        """
        self.state_buffer.append(np.array([s], np.float32))
        self.action_buffer.append(a)
        self.reward_buffer.append(r)

    def learn(self):
        """
        update policy parameters via stochastic gradient ascent
        :return: None
        """
        discounted_reward_buffer_norm = self._discount_and_norm_rewards()

        with tf.GradientTape() as tape:
            _logits = self.model(np.vstack(self.state_buffer))
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=_logits, labels=np.array(self.action_buffer)
            )
            loss = tf.reduce_mean(neg_log_prob * discounted_reward_buffer_norm)

        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []  # empty episode data
        return discounted_reward_buffer_norm

    def _discount_and_norm_rewards(self):
        """
        compute discount_and_norm_rewards
        :return: discount_and_norm_rewards
        """
        # discount episode rewards
        discounted_reward_buffer = np.zeros_like(self.reward_buffer)
        running_add = 0
        for t in reversed(range(0, len(self.reward_buffer))):
            running_add = running_add * self.gamma + self.reward_buffer[t]
            discounted_reward_buffer[t] = running_add

        # normalize episode rewards
        discounted_reward_buffer -= np.mean(discounted_reward_buffer)
        discounted_reward_buffer /= np.std(discounted_reward_buffer)
        return discounted_reward_buffer

    def save(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'pg_policy.hdf5'), self.model)

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'pg_policy.hdf5'), self.model)


if __name__ == '__main__':
    env = gym.make(ENV_ID).unwrapped

    # reproducible
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)

    agent = PolicyGradient(
        action_num=env.action_space.n,
        state_dim=env.observation_space.shape[0],
    )

    t0 = time.time()

    if args.train:
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):

            state = env.reset()
            episode_reward = 0

            for step in range(MAX_STEPS):  # in one episode
                if RENDER:
                    env.render()

                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward)
                state = next_state
                episode_reward += reward
                if done:
                    break
            agent.learn()
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                    episode + 1, TRAIN_EPISODES, episode_reward,
                    time.time() - t0
                )
            )

            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)

        agent.save()
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    if args.test:
        # test
        agent.load()
        for episode in range(TEST_EPISODES):
            state = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                env.render()
                state, reward, done, info = env.step(agent.get_action(state, True))
                episode_reward += reward
                if done:
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward,
                    time.time() - t0
                )
            )
