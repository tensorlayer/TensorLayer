"""
DQN and its variants
------------------------
We support Double DQN, Dueling DQN and PER here.

Reference:
------------------------
1. Double DQN
    Van Hasselt H, Guez A, Silver D. Deep reinforcement learning with double
    q-learning[C]//Thirtieth AAAI Conference on Artificial Intelligence. 2016.
2. Dueling DQN
    Wang Z, Schaul T, Hessel M, et al. Dueling network architectures for deep
    reinforcement learning[J]. arXiv preprint arXiv:1511.06581, 2015.
3. PER
    Schaul T, Quan J, Antonoglou I, et al. Prioritized experience replay[J]. arXiv
    preprint arXiv:1511.05952, 2015.

Usage:
------------------------
python3 main.py --algorithm=dqn --env=CartPole-v0 --env_type=classic
"""
import os
import random

import numpy as np
import tensorflow as tf

from common.buffer import PrioritizedReplayBuffer, ReplayBuffer
from common.wrappers import build_env


def learn(env_id, env_type, seed, mode, **kwargs):
    env = build_env(env_id, seed=seed)
    if env_type.lower == 'atari':
        return core_learn(env, mode=mode, **atari_parameters(env, **kwargs))
    else:
        return core_learn(env, mode=mode, **other_parameters(env, **kwargs))


def core_learn(
        env, mode, number_timesteps, network, optimizer, ob_scale, gamma, double_q, exploration_fraction,
        exploration_final_eps, batch_size, learning_starts, target_network_update_freq, buffer_size, prioritized_replay,
        prioritized_replay_alpha, prioritized_replay_beta0, save_path='dqn', save_interval=0, **kwargs
):
    """
    Parameters:
    ----------
    double_q (bool): if True double DQN will be used
    param_noise (bool): whether or not to use parameter space noise
    dueling (bool): if True dueling value estimation will be used
    exploration_fraction (float): fraction of entire training period over which
                                  the exploration rate is annealed
    exploration_final_eps (float): final value of random action probability
    batch_size (int): size of a batched sampled from replay buffer for training
    learning_starts (int): how many steps of the model to collect transitions
                           for before learning starts
    target_network_update_freq (int): update the target network every
                                      `target_network_update_freq` steps
    buffer_size (int): size of the replay buffer
    prioritized_replay (bool): if True prioritized replay buffer will be used.
    prioritized_replay_alpha (float): alpha parameter for prioritized replay
    prioritized_replay_beta0 (float): beta parameter for prioritized replay
    """
    out_dim = env.action_space.n
    if mode == 'train':
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        explore_steps = exploration_fraction * number_timesteps
        epsilon = lambda i_iter: 1 - (1 - exploration_final_eps) * min(1, i_iter / explore_steps)
        qnet = network
        targetqnet = tf.keras.models.clone_model(qnet)
        if prioritized_replay:
            buffer = PrioritizedReplayBuffer(buffer_size, prioritized_replay_alpha, prioritized_replay_beta0)
        else:
            buffer = ReplayBuffer(buffer_size)

        o = env.reset()
        for i in range(1, number_timesteps + 1):
            eps = epsilon(i)
            if prioritized_replay:
                buffer.beta += (1 - prioritized_replay_beta0) / number_timesteps

            # select action
            if random.random() < eps:
                a = int(random.random() * out_dim)
            else:
                obv = np.expand_dims(o, 0).astype('float32') * ob_scale
                a = qnet(obv).numpy().argmax(1)[0]

            # execute action and feed to replay buffer
            # note that `_` tail in var name means next
            o_, r, done, info = env.step(a)
            buffer.push(o, a, r, o_, done)

            if i >= learning_starts:
                # sync q net and target q net
                if i % target_network_update_freq == 0:
                    sync(qnet, targetqnet)

                # sample from replay buffer
                if prioritized_replay:
                    b_o, b_a, b_r, b_o_, b_d, weights, idxs = buffer.sample(batch_size)
                else:
                    b_o, b_a, b_r, b_o_, b_d = buffer.sample(batch_size)

                if double_q:
                    b_a_ = tf.one_hot(tf.argmax(qnet(b_o_), 1), out_dim)
                    b_q_ = (1 - b_d) * tf.reduce_sum(targetqnet(b_o_) * b_a_, 1).numpy()
                else:
                    b_q_ = (1 - b_d) * tf.reduce_max(targetqnet(b_o_), 1)
                b_tarq = b_r + gamma * b_q_

                # calculate loss
                with tf.GradientTape() as q_tape:
                    b_q = tf.reduce_sum(qnet(b_o) * tf.one_hot(b_a, out_dim), 1)
                    abs_td_error = tf.abs(b_q - b_tarq)
                    if prioritized_replay:
                        loss = tf.reduce_mean(weights * huber_loss(abs_td_error))
                    else:
                        loss = tf.reduce_mean(huber_loss(b_q - b_tarq))
                if prioritized_replay:
                    priorities = np.clip(abs_td_error.numpy(), 1e-6, None)
                    buffer.update_priorities(idxs, priorities)

                # backward gradients
                q_grad = q_tape.gradient(loss, qnet.trainable_variables)
                optimizer.apply_gradients(zip(q_grad, qnet.trainable_variables))

            if done:
                o = env.reset()
            else:
                o = o_

            if save_interval != 0:
                qnet.save_weights(save_path)

            # episode in info is real (unwrapped) message
            if info.get('episode'):
                reward, length = info['episode']['r'], info['episode']['l']
                print(
                    'timesteps sofar {}, epsilon {:.2f}, episode reward {:.4f}, episode length {}'.format(
                        i, eps, reward, length
                    )
                )
    else:
        qnet = network
        qnet.load_weights(save_path)

        o = env.reset()
        for i in range(1, number_timesteps + 1):
            obv = np.expand_dims(o, 0).astype('float32') * ob_scale
            a = qnet(obv).numpy().argmax(1)[0]

            # execute action
            # note that `_` tail in var name means next
            o_, r, done, info = env.step(a)

            if done:
                o = env.reset()
            else:
                o = o_

            # episode in info is real (unwrapped) message
            if info.get('episode'):
                reward, length = info['episode']['r'], info['episode']['l']
                print('timesteps sofar {}, episode reward {:.4f}, episode length {}'.format(i, reward, length))


def atari_parameters(env, **kwargs):
    in_dim = env.observation_space.shape
    policy_dim = env.action_space.n
    params = dict(
        lr=1e-4, number_timesteps=int(1e6), grad_norm=10, batch_size=32, double_q=True, buffer_size=10000,
        exploration_fraction=0.1, exploration_final_eps=0.01, learning_starts=10000, target_network_update_freq=1000,
        gamma=0.99, prioritized_replay=True, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, dueling=True,
        ob_scale=1 / 255.0
    )
    params.update(kwargs)
    network = get_cnn_model(in_dim, policy_dim, params.pop('dueling'))
    norm = params.pop('grad_norm')
    if norm:
        optimizer = tf.optimizers.Adam(learning_rate=params.pop('lr'), clipnorm=norm)
    else:
        optimizer = tf.optimizers.Adam(learning_rate=params.pop('lr'))
    params.update(network=network, optimizer=optimizer)
    return params


def other_parameters(env, **kwargs):
    in_dim = env.observation_space.shape[0]
    policy_dim = env.action_space.n
    params = dict(
        lr=5e-3, grad_norm=None, batch_size=100, number_timesteps=10000, double_q=True, buffer_size=1000,
        exploration_fraction=0.1, exploration_final_eps=0.01, learning_starts=100, target_network_update_freq=50,
        gamma=0.99, prioritized_replay=False, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, dueling=True,
        ob_scale=1
    )
    params.update(kwargs)
    network = get_mlp_model(in_dim, policy_dim, params.pop('dueling'))
    norm = params.pop('grad_norm')
    if norm:
        optimizer = tf.optimizers.Adam(learning_rate=params.pop('lr'), clipnorm=norm)
    else:
        optimizer = tf.optimizers.Adam(learning_rate=params.pop('lr'))
    params.update(network=network, optimizer=optimizer)
    return params


def get_mlp_model(in_dim, out_dim, dueling):
    """MLP model"""
    inputs = tf.keras.layers.Input(in_dim)
    net = tf.keras.layers.Dense(units=64, activation=tf.nn.tanh)(inputs)
    qvalue = tf.keras.layers.Dense(units=out_dim)(net)
    if dueling:
        svalue = tf.keras.layers.Dense(units=1)(net)
        output = svalue + qvalue - tf.reduce_mean(qvalue, 1, keepdims=True)
    else:
        output = qvalue
    return tf.keras.Model(inputs=inputs, outputs=output)


def get_cnn_model(in_dim, out_dim, dueling):
    """CNN model"""
    inputs = tf.keras.layers.Input(in_dim)
    conv1 = tf.keras.layers.Conv2D(32, (8, 8), (4, 4), activation=tf.nn.relu)(inputs)
    conv2 = tf.keras.layers.Conv2D(64, (4, 4), (2, 2), activation=tf.nn.relu)(conv1)
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), activation=tf.nn.relu)(conv2)
    flatten = tf.keras.layers.Flatten()(conv3)
    preq = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)(flatten)
    qvalue = tf.keras.layers.Dense(units=out_dim)(preq)
    if dueling:
        pres = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)(flatten)
        svalue = tf.keras.layers.Dense(units=1)(pres)
        output = svalue + qvalue - tf.reduce_mean(qvalue, 1, keepdims=True)
    else:
        output = qvalue
    return tf.keras.Model(inputs=inputs, outputs=output)


def huber_loss(x):
    """Loss function for value"""
    return tf.where(tf.abs(x) < 1, tf.square(x) * 0.5, tf.abs(x) - 0.5)


def sync(net, net_tar):
    """Copy q network to target q network"""
    for var, var_tar in zip(net.trainable_weights, net_tar.trainable_weights):
        var_tar.assign(var)


def dqn_loss(act_qtar, qvalues):
    """DQN loss"""
    act, qtar = tf.split(act_qtar, 2, axis=-1)
    act = tf.squeeze(tf.cast(act, tf.int32), 1)
    qtar = tf.squeeze(qtar, 1)
    qpre = tf.reduce_sum(qvalues * tf.one_hot(act, tf.shape(qvalues)[-1]), 1)
    return tf.reduce_mean(huber_loss(qpre - qtar))
