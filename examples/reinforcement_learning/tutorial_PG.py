"""
Vanilla Policy Gradient(VPG or REINFORCE)
-----------------------------------------
The policy gradient algorithm works by updating policy parameters via stochastic gradient ascent on policy performance.
It's an on-policy algorithm can be used for environments with either discrete or continuous action spaces.
Here is an example on discrete action space game CartPole-v0.
To apply it on continuous action space, you need to change the last softmax layer and the choose_action function.

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
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false')
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_NAME = 'CartPole-v0'  # environment name
RANDOMSEED = 1  # random seed

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time
num_episodes = 3000

###############################  PG  ####################################


class PolicyGradient:
    """
    PG class
    """

    def __init__(self, n_features, n_actions, learning_rate=0.01, reward_decay=0.95):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        def get_model(inputs_shape):
            """
            Build a neural network model.
            :param inputs_shape: state_shape
            :return: act
            """
            with tf.name_scope('inputs'):
                self.tf_obs = tl.layers.Input(inputs_shape, tf.float32, name="observations")
                self.tf_acts = tl.layers.Input([
                    None,
                ], tf.int32, name="actions_num")
                self.tf_vt = tl.layers.Input([
                    None,
                ], tf.float32, name="actions_value")
            # fc1
            layer = tl.layers.Dense(
                n_units=30, act=tf.nn.tanh, W_init=tf.random_normal_initializer(mean=0, stddev=0.3),
                b_init=tf.constant_initializer(0.1), name='fc1'
            )(self.tf_obs)
            # fc2
            all_act = tl.layers.Dense(
                n_units=self.n_actions, act=None, W_init=tf.random_normal_initializer(mean=0, stddev=0.3),
                b_init=tf.constant_initializer(0.1), name='all_act'
            )(layer)
            return tl.models.Model(inputs=self.tf_obs, outputs=all_act, name='PG model')

        self.model = get_model([None, n_features])
        self.model.train()
        self.optimizer = tf.optimizers.Adam(self.lr)

    def choose_action(self, s):
        """
        choose action with probabilities.
        :param s: state
        :return: act
        """
        _logits = self.model(np.array([s], np.float32))
        _probs = tf.nn.softmax(_logits).numpy()
        return tl.rein.choice_action_by_probs(_probs.ravel())

    def choose_action_greedy(self, s):
        """
        choose action with greedy policy
        :param s: state
        :return: act
        """
        _probs = tf.nn.softmax(self.model(np.array([s], np.float32))).numpy()
        return np.argmax(_probs.ravel())

    def store_transition(self, s, a, r):
        """
        store data in memory buffer
        :param s: state
        :param a: act
        :param r: reward
        :return:
        """
        self.ep_obs.append(np.array([s], np.float32))
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        """
        update policy parameters via stochastic gradient ascent
        :return: None
        """
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        with tf.GradientTape() as tape:

            _logits = self.model(np.vstack(self.ep_obs))
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_logits, labels=np.array(self.ep_as))
            # this is negative log of chosen action

            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)

            loss = tf.reduce_mean(neg_log_prob * discounted_ep_rs_norm)  # reward guided loss

        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        """
        compute discount_and_norm_rewards
        :return: discount_and_norm_rewards
        """
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        tl.files.save_weights_to_hdf5('model/pg_policy.hdf5', self.model)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/pg_policy.hdf5', self.model)


if __name__ == '__main__':

    # reproducible
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    tl.logging.set_verbosity(tl.logging.DEBUG)

    env = gym.make(ENV_NAME)
    env.seed(RANDOMSEED)  # reproducible, general Policy gradient has high variance
    env = env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    RL = PolicyGradient(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.02,
        reward_decay=0.99,
        # output_graph=True,
    )

    if args.train:
        reward_buffer = []

        for i_episode in range(num_episodes):

            episode_time = time.time()
            observation = env.reset()

            while True:
                if RENDER:
                    env.render()

                action = RL.choose_action(observation)

                observation_, reward, done, info = env.step(action)

                RL.store_transition(observation, action, reward)

                if done:
                    ep_rs_sum = sum(RL.ep_rs)

                    if 'running_reward' not in globals():
                        running_reward = ep_rs_sum
                    else:
                        running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

                    if running_reward > DISPLAY_REWARD_THRESHOLD:
                        RENDER = True  # rendering

                    # print("episode:", i_episode, "  reward:", int(running_reward))

                    print(
                        "Episode [%d/%d] \tsum reward: %d  \trunning reward: %f \ttook: %.5fs " %
                        (i_episode, num_episodes, ep_rs_sum, running_reward, time.time() - episode_time)
                    )
                    reward_buffer.append(running_reward)

                    vt = RL.learn()

                    plt.ion()
                    plt.cla()
                    plt.title('PG')
                    plt.plot(reward_buffer, )  # plot the episode vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                    plt.pause(0.1)

                    break

                observation = observation_
        RL.save_ckpt()
        plt.ioff()
        plt.show()

    # test
    RL.load_ckpt()
    observation = env.reset()
    while True:
        env.render()
        action = RL.choose_action(observation)
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
