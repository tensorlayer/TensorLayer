#! /usr/bin/python
# -*- coding: utf-8 -*-
"""Monte-Carlo Policy Network π(a|s)  (REINFORCE).
To understand Reinforcement Learning, we let computer to learn how to play
Pong game from the original screen inputs. Before we start, we highly recommend
you to go through a famous blog called “Deep Reinforcement Learning: Pong from
Pixels” which is a minimalistic implementation of deep reinforcement learning by
using python-numpy and OpenAI gym environment.
The code here is the reimplementation of Karpathy's Blog by using TensorLayer.
Compare with Karpathy's code, we store observation for a batch, but he store
observation for only one episode and gradients. (so we will use
more memory if the observation is very large.)

TODO
-----
- update grads every step rather than storing all observation!
- tensorlayer@gmail.com

References
------------
- http://karpathy.github.io/2016/05/31/rl/
"""
import time

import gym
import numpy as np
import tensorflow as tf

import tensorlayer as tl

tl.logging.set_verbosity(tl.logging.DEBUG)

# hyper-parameters
image_size = 80
D = image_size * image_size
H = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
render = False  # display the game environment
# resume = True         # load existing policy network
model_file_name = "model_pong"
np.set_printoptions(threshold=np.inf)


def prepro(I):
    """Prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector."""
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
running_reward = None
reward_sum = 0
episode_number = 0

xs, ys, rs = [], [], []


# policy network
def get_model(inputs_shape):
    ni = tl.layers.Input(inputs_shape)
    nn = tl.layers.Dense(n_units=H, act=tf.nn.relu, name='hidden')(ni)
    nn = tl.layers.Dense(n_units=3, name='output')(nn)
    M = tl.models.Model(inputs=ni, outputs=nn, name="mlp")
    return M


model = get_model([None, D])
train_weights = model.trainable_weights

optimizer = tf.optimizers.RMSprop(lr=learning_rate, decay=decay_rate)

model.train()  # set model to train mode (in case you add dropout into the model)

start_time = time.time()
game_number = 0
while True:
    if render:
        env.render()

    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D, dtype=np.float32)
    x = x.reshape(1, D)
    prev_x = cur_x

    _prob = model(x)
    prob = tf.nn.softmax(_prob)

    # action. 1: STOP  2: UP  3: DOWN
    # action = np.random.choice([1,2,3], p=prob.flatten())
    # action = tl.rein.choice_action_by_probs(prob.flatten(), [1, 2, 3])
    action = tl.rein.choice_action_by_probs(prob[0].numpy(), [1, 2, 3])

    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    xs.append(x)  # all observations in an episode
    ys.append(action - 1)  # all fake labels in an episode (action begins from 1, so minus 1)
    rs.append(reward)  # all rewards in an episode

    if done:
        episode_number += 1
        game_number = 0

        if episode_number % batch_size == 0:
            print('batch over...... updating parameters......')
            epx = np.vstack(xs)
            epy = np.asarray(ys)
            epr = np.asarray(rs)
            disR = tl.rein.discount_episode_rewards(epr, gamma)
            disR -= np.mean(disR)
            disR /= np.std(disR)

            xs, ys, rs = [], [], []

            with tf.GradientTape() as tape:
                _prob = model(epx)
                _loss = tl.rein.cross_entropy_reward_loss(_prob, epy, disR)
            grad = tape.gradient(_loss, train_weights)
            optimizer.apply_gradients(zip(grad, train_weights))

        ## TODO
        # if episode_number % (batch_size * 100) == 0:
        #     tl.files.save_npz(network.all_params, name=model_file_name + '.npz')

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was {}. running mean: {}'.format(reward_sum, running_reward))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    if reward != 0:
        print(
            (
                'episode %d: game %d took %.5fs, reward: %f' %
                (episode_number, game_number, time.time() - start_time, reward)
            ), ('' if reward == -1 else ' !!!!!!!!')
        )
        start_time = time.time()
        game_number += 1
