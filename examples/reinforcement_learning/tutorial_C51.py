"""Implement C51 algorithm
Bellemare M G, Dabney W, Munos R. A distributional perspective on reinforcement
learning[C]//Proceedings of the 34th International Conference on Machine
Learning-Volume 70. JMLR. org, 2017: 449-458.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""
import argparse
import os
import random
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tutorial_wrappers import build_env


parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='train or test', default='train')
parser.add_argument('--save_path', default='c51',
                    help='folder to save if mode == train else model path,'
                         'qnet will be saved once target net update')
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--env_id', default='CartPole-v0',
                    help='CartPole-v0 or PongNoFrameskip-v4')
args = parser.parse_args()
print(args)

if args.mode == 'train':
    os.makedirs(args.save_path, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)  # reproducible
env_id = args.env_id
env = build_env(env_id, seed=args.seed)

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

in_dim = env.observation_space.shape
out_dim = env.action_space.n
reward_gamma = 0.99  # reward discount
batch_size = 32  # batch size for sampling from replay buffer
warm_start = buffer_size / 10  # sample times before learning
atom_num = 51
min_value = -10
max_value = 10
vrange = np.linspace(min_value, max_value, atom_num)
deltaz = float(max_value - min_value) / (atom_num - 1)


class MLP(tl.models.Model):
    def __init__(self, name):
        super(MLP, self).__init__(name=name)
        self.h1 = tl.layers.Dense(64, tf.nn.tanh, in_channels=in_dim[0],
                                  W_init=tf.initializers.GlorotUniform())
        self.qvalue = tl.layers.Dense(out_dim * atom_num,
                                      in_channels=64, name='q',
                                      W_init=tf.initializers.GlorotUniform())
        self.reshape = tl.layers.Reshape((-1, out_dim, atom_num))

    def forward(self, ni):
        qvalues = self.qvalue(self.h1(ni))
        return tf.nn.log_softmax(self.reshape(qvalues), 2)


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
        self.qvalue = tl.layers.Dense(out_dim * atom_num,
                                      in_channels=256, name='q',
                                      W_init=tf.initializers.GlorotUniform())
        self.reshape = tl.layers.Reshape((-1, out_dim, atom_num))

    def forward(self, ni):
        feature = self.flatten(self.conv3(self.conv2(self.conv1(ni))))
        qvalues = self.qvalue(self.preq(feature))
        return tf.nn.log_softmax(self.reshape(qvalues), 2)


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
        # encode sample to numpy.array with right dtype
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
        # allow sampling with replacement
        idxes = [random.choice(indexes) for _ in range(batch_size)]
        return self._encode_sample(idxes)


def sync(net, net_tar):
    """Copy q network to target q network"""
    for var, var_tar in zip(net.trainable_weights, net_tar.trainable_weights):
        var_tar.assign(var)


if args.mode == 'train':
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
    for i in range(1, number_timesteps + 1):
        eps = epsilon(i)

        # select action
        if random.random() < eps:
            a = int(random.random() * out_dim)
        else:
            obv = np.expand_dims(o, 0).astype('float32') * ob_scale
            qdist = np.exp(qnet(obv).numpy())
            qvalues = (qdist * vrange).sum(-1)
            a = qvalues.argmax(1)[0]

        # execute action and feed to replay buffer
        # note that `_` tail in var name means next
        o_, r, done, info = env.step(a)
        buffer.add(o, a, r, o_, done)

        if i >= warm_start:
            # sync q net and target q net
            if i % target_q_update_freq == 0:
                sync(qnet, targetqnet)
                path = os.path.join(args.save_path, '{}.npz'.format(i))
                tl.files.save_npz(qnet.trainable_weights, name=path)

            # sample from replay buffer
            b_o, b_a, b_r, b_o_, b_d = buffer.sample(batch_size)

            # q estimation, see Algorithm 1 in paper for detail
            b_dist_ = np.exp(targetqnet(b_o_).numpy())
            b_a_ = (b_dist_ * vrange).sum(-1).argmax(1)
            b_tzj = np.clip(reward_gamma * (1 - b_d[:, None]) * vrange[None, :]
                            + b_r[:, None], min_value, max_value)
            b_i = (b_tzj - min_value) / deltaz
            b_l = np.floor(b_i).astype('int64')
            b_u = np.ceil(b_i).astype('int64')
            templ = b_dist_[range(batch_size), b_a_, :] * (b_u - b_i)
            tempu = b_dist_[range(batch_size), b_a_, :] * (b_i - b_l)
            b_m = np.zeros((batch_size, atom_num))
            # TODO: aggregate value by index and batch update (scatter_add)
            for j in range(batch_size):
                for k in range(atom_num):
                    b_m[j][b_l[j][k]] += templ[j][k]
                    b_m[j][b_u[j][k]] += tempu[j][k]
            b_m = tf.convert_to_tensor(b_m, dtype='float32')

            # calculate loss
            with tf.GradientTape() as q_tape:
                b_index = np.stack([range(batch_size), b_a], 1)
                b_index = tf.convert_to_tensor(b_index, 'int64')
                b_dist_a = tf.gather_nd(qnet(b_o), b_index)
                loss = -tf.reduce_mean(tf.reduce_sum(b_dist_a * b_m, 1))

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
else:
    qnet = MLP('q') if qnet_type == 'MLP' else CNN('q')
    tl.files.load_and_assign_npz(name=args.save_path, network=qnet)
    qnet.eval()

    nepisode = 0
    o = env.reset()
    for i in range(1, number_timesteps + 1):
        obv = np.expand_dims(o, 0).astype('float32') * ob_scale
        qdist = np.exp(qnet(obv).numpy())
        qvalues = (qdist * vrange).sum(-1)
        a = qvalues.argmax(1)[0]

        # execute action and feed to replay buffer
        # note that `_` tail in var name means next
        o_, r, done, info = env.step(a)

        if done:
            o = env.reset()
        else:
            o = o_

        # episode in info is real (unwrapped) message
        if info.get('episode'):
            nepisode += 1
            reward, length = info['episode']['r'], info['episode']['l']
            print('Time steps so far: {}, episode so far: {}, '
                  'episode reward: {:.4f}, episode length: {}'
                  .format(i, nepisode, reward, length))

