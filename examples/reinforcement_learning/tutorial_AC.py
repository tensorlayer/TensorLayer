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

To run
------
python tutorial_AC.py --train/test

"""
import argparse
import time

import gym
import numpy as np
import tensorflow as tf

import tensorlayer as tl

tl.logging.set_verbosity(tl.logging.DEBUG)

np.random.seed(2)
tf.random.set_seed(2)  # reproducible

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################

OUTPUT_GRAPH = False
MAX_EPISODE = 3000  # number of overall episodes for training
DISPLAY_REWARD_THRESHOLD = 100  # renders environment if running reward is greater then this threshold
MAX_EP_STEPS = 1000  # maximum time step in one episode
RENDER = False  # rendering wastes time
LAMBDA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

###############################  Actor-Critic  ####################################


class Actor(object):

    def __init__(self, n_features, n_actions, lr=0.001):

        def get_model(inputs_shape):
            ni = tl.layers.Input(inputs_shape, name='state')
            nn = tl.layers.Dense(
                n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden'
            )(ni)
            nn = tl.layers.Dense(
                n_units=10, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden2'
            )(nn)
            nn = tl.layers.Dense(n_units=n_actions, name='actions')(nn)
            return tl.models.Model(inputs=ni, outputs=nn, name="Actor")

        self.model = get_model([None, n_features])
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
        tl.files.save_npz(self.model.trainable_weights, name='model_actor.npz')

    def load_ckpt(self):  # load trained weights
        tl.files.load_and_assign_npz(name='model_actor.npz', network=self.model)


class Critic(object):

    def __init__(self, n_features, lr=0.01):

        def get_model(inputs_shape):
            ni = tl.layers.Input(inputs_shape, name='state')
            nn = tl.layers.Dense(
                n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden'
            )(ni)
            nn = tl.layers.Dense(
                n_units=5, act=tf.nn.relu, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden2'
            )(nn)
            nn = tl.layers.Dense(n_units=1, act=None, name='value')(nn)
            return tl.models.Model(inputs=ni, outputs=nn, name="Critic")

        self.model = get_model([1, n_features])
        self.model.train()

        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, s, r, s_):
        v_ = self.model(np.array([s_]))
        with tf.GradientTape() as tape:
            v = self.model(np.array([s]))
            ## TD_error = r + lambd * V(newS) - V(S)
            td_error = r + LAMBDA * v_ - v
            loss = tf.square(td_error)
        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

        return td_error

    def save_ckpt(self):  # save trained weights
        tl.files.save_npz(self.model.trainable_weights, name='model_critic.npz')

    def load_ckpt(self):  # load trained weights
        tl.files.load_and_assign_npz(name='model_critic.npz', network=self.model)


if __name__ == '__main__':
    ''' 
    choose environment
    1. Openai gym:
    env = gym.make()
    2. DeepMind Control Suite:
    env = dm_control2gym.make()
    '''
    env = gym.make('CartPole-v1')
    # dm_control2gym.create_render_mode('example mode', show=True, return_pixel=False, height=240, width=320, camera_id=-1, overlays=(),
    #              depth=False, scene_option=None)
    # env = dm_control2gym.make(domain_name="cartpole", task_name="balance")
    env.seed(2)  # reproducible
    # env = env.unwrapped
    N_F = env.observation_space.shape[0]
    # N_A = env.action_space.shape[0]
    N_A = env.action_space.n

    print("observation dimension: %d" % N_F)  # 4
    print("observation high: %s" % env.observation_space.high)  # [ 2.4 , inf , 0.41887902 , inf]
    print("observation low : %s" % env.observation_space.low)  # [-2.4 , -inf , -0.41887902 , -inf]
    print("num of actions: %d" % N_A)  # 2 : left or right

    actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
    # we need a good teacher, so the teacher should learn faster than the actor
    critic = Critic(n_features=N_F, lr=LR_C)

    if args.train:
        t0 = time.time()
        for i_episode in range(MAX_EPISODE):
            # episode_time = time.time()
            s = env.reset().astype(np.float32)
            t = 0  # number of step in this episode
            all_r = []  # rewards of all steps
            while True:

                if RENDER: env.render()

                a = actor.choose_action(s)

                s_new, r, done, info = env.step(a)
                s_new = s_new.astype(np.float32)

                if done: r = -20
                # these may helpful in some tasks
                # if abs(s_new[0]) >= env.observation_space.high[0]:
                # #  cart moves more than 2.4 units from the center
                #     r = -20
                # reward for the distance between cart to the center
                # r -= abs(s_new[0])  * .1

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

                if done or t >= MAX_EP_STEPS:
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
                    .format(i_episode, MAX_EPISODE, ep_rs_sum, time.time()-t0 ))

                    # Early Stopping for quick check
                    if t >= MAX_EP_STEPS:
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
                                print("reward", rall)
                                s = env.reset().astype(np.float32)
                                rall = 0
                    break
        actor.save_ckpt()
        critic.save_ckpt()

    if args.test:
        actor.load_ckpt()
        critic.load_ckpt()
        t0 = time.time()

        for i_episode in range(MAX_EPISODE):
            episode_time = time.time()
            s = env.reset().astype(np.float32)
            t = 0  # number of step in this episode
            all_r = []  # rewards of all steps
            while True:
                if RENDER: env.render()
                a = actor.choose_action(s)
                s_new, r, done, info = env.step(a)
                s_new = s_new.astype(np.float32)
                if done: r = -20
                # these may helpful in some tasks
                # if abs(s_new[0]) >= env.observation_space.high[0]:
                # #  cart moves more than 2.4 units from the center
                #     r = -20
                # reward for the distance between cart to the center
                # r -= abs(s_new[0])  * .1

                all_r.append(r)
                s = s_new
                t += 1

                if done or t >= MAX_EP_STEPS:
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
                    .format(i_episode, MAX_EPISODE, ep_rs_sum, time.time()-t0 ))

                    # Early Stopping for quick check
                    if t >= MAX_EP_STEPS:
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
                                print("reward", rall)
                                s = env.reset().astype(np.float32)
                                rall = 0
                    break
