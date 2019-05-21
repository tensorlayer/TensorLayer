"""Implement following enhanced deep q-learning algorithms
1. Double DQN
    Van Hasselt H, Guez A, Silver D. Deep reinforcement learning with double
    q-learning[C]//Thirtieth AAAI Conference on Artificial Intelligence. 2016.

2. Dueling DQN
    Wang Z, Schaul T, Hessel M, et al. Dueling network architectures for deep
    reinforcement learning[J]. arXiv preprint arXiv:1511.06581, 2015.

3. Noisy DQN
    Plappert M, Houthooft R, Dhariwal P, et al. Parameter space noise for
    exploration[J]. arXiv preprint arXiv:1706.01905, 2017.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""
import random
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tutorial_wrappers import build_env


seed = 0
env_id = 'CartPole-v0'  # CartPole-v0, PongNoFrameskip-v4
if env_id == 'CartPole-v0':
    qnet_type = 'MLP'
    number_timesteps = 10000  # total number of time steps to train on
    explore_timesteps = 100
    # epsilon-greedy schedule, final exploit prob is 0.99
    epsilon = lambda i_iter: 1 - 0.99 * min(1, i_iter / explore_timesteps)
    lr = 5e-3  # learning rate
    buffer_size = 1000  # replay buffer size
    target_q_update_freq = 50  # how frequency target q net update
    ob_scale = 1.0  # scale observations
else:
    # reward will increase obviously after 1e5 time steps
    qnet_type = 'CNN'
    number_timesteps = int(1e6)  # total number of time steps to train on
    explore_timesteps = 1e5
    # epsilon-greedy schedule, final exploit prob is 0.99
    epsilon = lambda i_iter: 1 - 0.99 * min(1, i_iter / explore_timesteps)
    lr = 1e-4  # learning rate
    buffer_size = 10000  # replay buffer size
    target_q_update_freq = 200  # how frequency target q net update
    ob_scale = 1.0 / 255  # scale observations

env = build_env(env_id, seed=seed)
in_dim = env.observation_space.shape
out_dim = env.action_space.n
reward_gamma = 0.99  # reward discount
batch_size = 32  # batch size for sampling from replay buffer
warm_start = buffer_size / 10  # sample times befor learning
noise_update_freq = 50  # how frequency param noise net update


class MLP(tl.models.Model):
    def __init__(self, name):
        super(MLP, self).__init__(name=name)
        self.h1 = tl.layers.Dense(64, tf.nn.tanh, in_channels=in_dim[0])
        self.qvalue = tl.layers.Dense(out_dim, in_channels=64, name='q',
                                      W_init=tf.initializers.GlorotUniform())
        self.svalue = tl.layers.Dense(1, in_channels=64, name='s',
                                      W_init=tf.initializers.GlorotUniform())
        self.noise_scale = 0

    def forward(self, ni):
        feature = self.h1(ni)

        # apply noise to all linear layer
        if self.noise_scale != 0:
            noises = []
            for layer in [self.qvalue, self.svalue]:
                for var in layer.trainable_weights:
                    noise = tf.random.normal(tf.shape(var), 0, self.noise_scale)
                    noises.append(noise)
                    var.assign_add(noise)

        qvalue = self.qvalue(feature)
        svalue = self.svalue(feature)

        if self.noise_scale != 0:
            idx = 0
            for layer in [self.qvalue, self.svalue]:
                for var in layer.trainable_weights:
                    var.assign_sub(noises[idx])
                    idx += 1

        # dueling network
        out = svalue + qvalue - tf.reduce_mean(qvalue, 1, keepdims=True)
        return out


class CNN(tl.models.Model):
    def __init__(self, name):
        super(CNN, self).__init__(name=name)
        h, w, in_channels = in_dim
        dense_in_channels = 64 * ((h - 28) // 8) * ((w - 28) // 8)
        self.conv1 = tl.layers.Conv2d(32, (8, 8), (4, 4), tf.nn.relu, 'VALID',
                                      in_channels=in_channels, name='conv2d_1',
                                      W_init=tf.initializers.GlorotUniform())
        self.conv2 = tl.layers.Conv2d(64, (4, 4), (2, 2), tf.nn.relu, 'VALID',
                                      in_channels=32, name='conv2d_2',
                                      W_init=tf.initializers.GlorotUniform())
        self.conv3 = tl.layers.Conv2d(64, (3, 3), (1, 1), tf.nn.relu, 'VALID',
                                      in_channels=64, name='conv2d_3',
                                      W_init=tf.initializers.GlorotUniform())
        self.flatten = tl.layers.Flatten(name='flatten')
        self.preq = tl.layers.Dense(256, tf.nn.relu,
                                    in_channels=dense_in_channels, name='pre_q',
                                    W_init=tf.initializers.GlorotUniform())
        self.qvalue = tl.layers.Dense(out_dim, in_channels=256, name='q',
                                      W_init=tf.initializers.GlorotUniform())
        self.pres = tl.layers.Dense(256, tf.nn.relu,
                                    in_channels=dense_in_channels, name='pre_s',
                                    W_init=tf.initializers.GlorotUniform())
        self.svalue = tl.layers.Dense(1, in_channels=256, name='state',
                                      W_init=tf.initializers.GlorotUniform())
        self.noise_scale = 0

    def forward(self, ni):
        feature = self.flatten(self.conv3(self.conv2(self.conv1(ni))))

        # apply noise to all linear layer
        if self.noise_scale != 0:
            noises = []
            for layer in [self.preq, self.qvalue, self.pres, self.svalue]:
                for var in layer.trainable_weights:
                    noise = tf.random.normal(tf.shape(var), 0, self.noise_scale)
                    noises.append(noise)
                    var.assign_add(noise)

        qvalue = self.qvalue(self.preq(feature))
        svalue = self.svalue(self.pres(feature))

        if self.noise_scale != 0:
            idx = 0
            for layer in [self.preq, self.qvalue, self.pres, self.svalue]:
                for var in layer.trainable_weights:
                    var.assign_sub(noises[idx])
                    idx += 1

        # dueling network
        return svalue + qvalue - tf.reduce_mean(qvalue, 1, keepdims=True)


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, *args):
        if self._next_idx >= len(self._storage):
            self._storage.append(args)
        else:
            self._storage[self._next_idx] = args
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        b_o, b_a, b_r, b_o_, b_d = [], [], [], [], []
        for i in idxes:
            o, a, r, o_, d = self._storage[i]
            b_o.append(o)
            b_a.append(a)
            b_r.append(r)
            b_o_.append(o_)
            b_d.append(d)
        return (
            np.stack(b_o).astype('float32') * ob_scale,
            np.stack(b_a).astype('int32'),
            np.stack(b_r).astype('float32'),
            np.stack(b_o_).astype('float32') * ob_scale,
            np.stack(b_d).astype('float32'),
        )

    def sample(self, batch_size):
        indexes = range(len(self._storage))
        idxes = [random.choice(indexes) for _ in range(batch_size)]
        return self._encode_sample(idxes)


def huber_loss(x):
    """Loss function for value"""
    return tf.where(tf.abs(x) < 1, tf.square(x) * 0.5, tf.abs(x) - 0.5)


def sync(net, net_tar):
    """Copy q network to target q network"""
    for var, var_tar in zip(net.trainable_weights, net_tar.trainable_weights):
        var_tar.assign(var)


def log_softmax(x, dim):
    temp = x - np.max(x, dim, keepdims=True)
    return temp - np.log(np.exp(temp).sum(dim, keepdims=True))


def softmax(x, dim):
    temp = np.exp(x - np.max(x, dim, keepdims=True))
    return temp / temp.sum(dim, keepdims=True)


qnet = MLP('q') if qnet_type == 'MLP' else CNN('q')
qnet.train()
trainabel_weights = qnet.trainable_weights
targetqnet = MLP('targetq') if qnet_type == 'MLP' else CNN('targetq')
targetqnet.infer()
sync(qnet, targetqnet)
optimizer = tf.optimizers.Adam(learning_rate=lr)
buffer = ReplayBuffer(buffer_size)

o = env.reset()
nepisode = 0
t = time.time()
noise_scale = 1e-2
for i in range(1, number_timesteps + 1):
    eps = epsilon(i)

    # select action
    if random.random() < eps:
        a = int(random.random() * out_dim)
    else:
        # noise schedule is based on KL divergence between perturbed and
        # non-perturbed policy, see https://arxiv.org/pdf/1706.01905.pdf
        obv = np.expand_dims(o, 0).astype('float32') * ob_scale
        if i < explore_timesteps:
            qnet.noise_scale = noise_scale
            q_ptb = qnet(obv).numpy()
            qnet.noise_scale = 0
            if i % noise_update_freq == 0:
                q = qnet(obv).numpy()
                kl_ptb = (log_softmax(q, 1) - log_softmax(q_ptb, 1))
                kl_ptb = np.sum(kl_ptb * softmax(q, 1), 1).mean()
                kl_explore = -np.log(1 - eps + eps / out_dim)
                if kl_ptb < kl_explore:
                    noise_scale *= 1.01
                else:
                    noise_scale /= 1.01
            a = q_ptb.argmax(1)[0]
        else:
            a = qnet(obv).numpy().argmax(1)[0]

    # execute action and feed to replay buffer
    # note that `_` tail in var name means next
    o_, r, done, info = env.step(a)
    buffer.add(o, a, r, o_, done)

    if i >= warm_start:
        # sync q net and target q net
        if i % target_q_update_freq == 0:
            sync(qnet, targetqnet)

        # sample from replay buffer
        b_o, b_a, b_r, b_o_, b_d = buffer.sample(batch_size)

        # double q estimation
        b_a_ = tf.one_hot(tf.argmax(qnet(b_o_), 1), out_dim)
        b_q_ = (1 - b_d) * tf.reduce_sum(targetqnet(b_o_) * b_a_, 1)

        # calculate loss
        with tf.GradientTape() as q_tape:
            b_q = tf.reduce_sum(qnet(b_o) * tf.one_hot(b_a, out_dim), 1)
            loss = tf.reduce_mean(huber_loss(b_q - (b_r + reward_gamma * b_q_)))

        # backward gradients
        q_grad = q_tape.gradient(loss, trainabel_weights)
        optimizer.apply_gradients(zip(q_grad, trainabel_weights))

    if done:
        o = env.reset()
    else:
        o = o_

    # episode in info is real (unwrapped) message
    if info.get('episode'):
        nepisode += 1
        reward, length = info['episode']['r'], info['episode']['l']
        fps = int(length / (time.time() - t))
        print('Time steps so far: {}, episode so far: {}, '
              'episode reward: {:.4f}, episode length: {}, FPS: {}'
              .format(i, nepisode, reward, length, fps))
        t = time.time()
