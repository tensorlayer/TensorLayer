"""
C51 Algorithm
------------------------
Categorical 51 distributional RL algorithm, 51 means the number of atoms. In
this algorithm, instead of estimating actual expected value, value distribution
over a series of continuous sub-intervals (atoms) is considered.
Reference:
------------------------
Bellemare M G, Dabney W, Munos R. A distributional perspective on reinforcement
learning[C]//Proceedings of the 34th International Conference on Machine
Learning-Volume 70. JMLR. org, 2017: 449-458.
Environment:
------------------------
Cartpole and Pong in OpenAI Gym
Requirements:
------------------------
tensorflow>=2.0.0a0
tensorlayer>=2.0.0
To run:
------------------------
python tutorial_C51.py --mode=train
python tutorial_C51.py --mode=test --save_path=c51/8000.npz
"""
import argparse
import os
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
parser.add_argument(
    '--save_path', default=None, help='folder to save if mode == train else model path,'
    'qnet will be saved once target net update'
)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--env_id', default='CartPole-v0', help='CartPole-v0 or PongNoFrameskip-v4')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)  # reproducible
env_id = args.env_id
env = gym.make(env_id)
env.seed(args.seed)
alg_name = 'C51'

# ####################  hyper parameters  ####################
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
    clipnorm = None
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
    clipnorm = 10

in_dim = env.observation_space.shape
out_dim = env.action_space.n
reward_gamma = 0.99  # reward discount
batch_size = 32  # batch size for sampling from replay buffer
warm_start = buffer_size / 10  # sample times befor learning
atom_num = 51
min_value = -10
max_value = 10
vrange = np.linspace(min_value, max_value, atom_num)
deltaz = float(max_value - min_value) / (atom_num - 1)


# ##############################  Network  ####################################
class MLP(tl.models.Model):

    def __init__(self, name):
        super(MLP, self).__init__(name=name)
        self.h1 = tl.layers.Dense(64, tf.nn.tanh, in_channels=in_dim[0], W_init=tf.initializers.GlorotUniform())
        self.qvalue = tl.layers.Dense(
            out_dim * atom_num, in_channels=64, name='q', W_init=tf.initializers.GlorotUniform()
        )
        self.reshape = tl.layers.Reshape((-1, out_dim, atom_num))

    def forward(self, ni):
        qvalues = self.qvalue(self.h1(ni))
        return tf.nn.log_softmax(self.reshape(qvalues), 2)


class CNN(tl.models.Model):

    def __init__(self, name):
        super(CNN, self).__init__(name=name)
        h, w, in_channels = in_dim
        dense_in_channels = 64 * ((h - 28) // 8) * ((w - 28) // 8)
        self.conv1 = tl.layers.Conv2d(
            32, (8, 8), (4, 4), tf.nn.relu, 'VALID', in_channels=in_channels, name='conv2d_1',
            W_init=tf.initializers.GlorotUniform()
        )
        self.conv2 = tl.layers.Conv2d(
            64, (4, 4), (2, 2), tf.nn.relu, 'VALID', in_channels=32, name='conv2d_2',
            W_init=tf.initializers.GlorotUniform()
        )
        self.conv3 = tl.layers.Conv2d(
            64, (3, 3), (1, 1), tf.nn.relu, 'VALID', in_channels=64, name='conv2d_3',
            W_init=tf.initializers.GlorotUniform()
        )
        self.flatten = tl.layers.Flatten(name='flatten')
        self.preq = tl.layers.Dense(
            256, tf.nn.relu, in_channels=dense_in_channels, name='pre_q', W_init=tf.initializers.GlorotUniform()
        )
        self.qvalue = tl.layers.Dense(
            out_dim * atom_num, in_channels=256, name='q', W_init=tf.initializers.GlorotUniform()
        )
        self.reshape = tl.layers.Reshape((-1, out_dim, atom_num))

    def forward(self, ni):
        feature = self.flatten(self.conv3(self.conv2(self.conv1(ni))))
        qvalues = self.qvalue(self.preq(feature))
        return tf.nn.log_softmax(self.reshape(qvalues), 2)


# ##############################  Replay  ####################################
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


# #############################  Functions  ###################################
def huber_loss(x):
    """Loss function for value"""
    return tf.where(tf.abs(x) < 1, tf.square(x) * 0.5, tf.abs(x) - 0.5)


def sync(net, net_tar):
    """Copy q network to target q network"""
    for var, var_tar in zip(net.trainable_weights, net_tar.trainable_weights):
        var_tar.assign(var)


# ###############################  DQN  #####################################
class DQN(object):

    def __init__(self):
        model = MLP if qnet_type == 'MLP' else CNN
        self.qnet = model('q')
        if args.train:
            self.qnet.train()
            self.targetqnet = model('targetq')
            self.targetqnet.infer()
            sync(self.qnet, self.targetqnet)
        else:
            self.qnet.infer()
            self.load(args.save_path)
        self.niter = 0
        if clipnorm is not None:
            self.optimizer = tf.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
        else:
            self.optimizer = tf.optimizers.Adam(learning_rate=lr)

    def get_action(self, obv):
        eps = epsilon(self.niter)
        if args.train and random.random() < eps:
            return int(random.random() * out_dim)
        else:
            obv = np.expand_dims(obv, 0).astype('float32') * ob_scale
            qdist = np.exp(self._qvalues_func(obv).numpy())
            qvalues = (qdist * vrange).sum(-1)
            return qvalues.argmax(1)[0]

    @tf.function
    def _qvalues_func(self, obv):
        return self.qnet(obv)

    def train(self, b_o, b_a, b_r, b_o_, b_d):
        # TODO: move q_estimation in tf.function
        b_dist_ = np.exp(self.targetqnet(b_o_).numpy())
        b_a_ = (b_dist_ * vrange).sum(-1).argmax(1)
        b_tzj = np.clip(reward_gamma * (1 - b_d[:, None]) * vrange[None, :] + b_r[:, None], min_value, max_value)
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
        b_index = np.stack([range(batch_size), b_a], 1)
        b_index = tf.convert_to_tensor(b_index, 'int64')

        self._train_func(b_o, b_index, b_m)

        self.niter += 1
        if self.niter % target_q_update_freq == 0:
            sync(self.qnet, self.targetqnet)
            self.save(args.save_path)

    def save(self, path):
        if path is None:
            path = os.path.join('model', '_'.join([alg_name, env_id]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'q_net.hdf5'), self.qnet)

    def load(self, path):
        if path is None:
            path = os.path.join('model', '_'.join([alg_name, env_id]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'q_net.hdf5'), self.qnet)

    @tf.function
    def _train_func(self, b_o, b_index, b_m):
        with tf.GradientTape() as tape:
            b_dist_a = tf.gather_nd(self.qnet(b_o), b_index)
            loss = tf.reduce_mean(tf.negative(tf.reduce_sum(b_dist_a * b_m, 1)))

        grad = tape.gradient(loss, self.qnet.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.qnet.trainable_weights))


# #############################  Trainer  ###################################
if __name__ == '__main__':
    dqn = DQN()
    t0 = time.time()
    if args.train:
        buffer = ReplayBuffer(buffer_size)
        nepisode = 0
        all_episode_reward = []
        for i in range(1, number_timesteps + 1):
            o = env.reset()
            episode_reward = 0
            while True:
                a = dqn.get_action(o)
                # execute action and feed to replay buffer
                # note that `_` tail in var name means next
                o_, r, done, info = env.step(a)
                buffer.add(o, a, r, o_, done)
                episode_reward += r

                if i >= warm_start:
                    transitions = buffer.sample(batch_size)
                    dqn.train(*transitions)

                if done:
                    break
                else:
                    o = o_

            if nepisode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            nepisode += 1
            print(
                'Training  | Episode: {}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    nepisode, episode_reward,
                    time.time() - t0
                )
            )  # episode num starts from 1 in print

        dqn.save(args.save_path)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([alg_name, env_id])))

    if args.test:
        nepisode = 0
        for i in range(1, number_timesteps + 1):
            o = env.reset()
            episode_reward = 0
            while True:
                env.render()
                a = dqn.get_action(o)
                o_, r, done, info = env.step(a)
                episode_reward += r
                if done:
                    break
                else:
                    o = o_
            nepisode += 1
            print(
                'Testing  | Episode: {}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    nepisode, episode_reward,
                    time.time() - t0
                )
            )
