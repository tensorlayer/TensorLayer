"""
Prioritized Experience Replay
------------------------
Prioritized experience replay is an efficient replay method that replay
important transitions more frequently. Segment tree data structure is used to
speed up indexing.
Reference:
------------------------
Schaul T, Quan J, Antonoglou I, et al. Prioritized experience replay[J]. arXiv
preprint arXiv:1511.05952, 2015.
Dhariwal P, Hesse C, Klimov O, et al. Openai baselines (2017)[J]. URL
https://github. com/opfenai/baselines.
Environment:
------------------------
Cartpole and Pong in OpenAI Gym
Requirements:
------------------------
tensorflow>=2.0.0a0
tensorlayer>=2.0.0
To run:
------------------------
python tutorial_prioritized_replay.py --mode=train
python tutorial_prioritized_replay.py --mode=test --save_path=per/8000.npz
"""
import argparse
import operator
import os
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl

parser = argparse.ArgumentParser()
# add arguments in command  --train/test
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
alg_name = 'prioritized_replay'

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
prioritized_replay_alpha = 0.6  # alpha in PER
prioritized_replay_beta0 = 0.4  # initial beta in PER


# ##############################  Network  ####################################
class MLP(tl.models.Model):

    def __init__(self, name):
        super(MLP, self).__init__(name=name)
        self.h1 = tl.layers.Dense(64, tf.nn.tanh, in_channels=in_dim[0])
        self.qvalue = tl.layers.Dense(out_dim, in_channels=64, name='q', W_init=tf.initializers.GlorotUniform())

    def forward(self, ni):
        return self.qvalue(self.h1(ni))


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
        self.qvalue = tl.layers.Dense(out_dim, in_channels=256, name='q', W_init=tf.initializers.GlorotUniform())

    def forward(self, ni):
        feature = self.flatten(self.conv3(self.conv2(self.conv1(ni))))
        return self.qvalue(self.preq(feature))


# ##############################  Replay  ####################################
class SegmentTree(object):

    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, \
            "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx], self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity=capacity, operation=operator.add, neutral_element=0.0)

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(capacity=capacity, operation=min, neutral_element=float('inf'))

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


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


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, size, alpha, beta):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self.beta = beta

    def add(self, *args):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args)
        self._it_sum[idx] = self._max_priority**self._alpha
        self._it_min[idx] = self._max_priority**self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        idxes = self._sample_proportional(batch_size)

        it_sum = self._it_sum.sum()
        p_min = self._it_min.min() / it_sum
        max_weight = (p_min * len(self._storage))**(-self.beta)

        p_samples = np.asarray([self._it_sum[idx] for idx in idxes]) / it_sum
        weights = (p_samples * len(self._storage))**(-self.beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample + (weights.astype('float32'), idxes)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions"""
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority**self._alpha
            self._it_min[idx] = priority**self._alpha

            self._max_priority = max(self._max_priority, priority)


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
            return self._qvalues_func(obv).numpy().argmax(1)[0]

    @tf.function
    def _qvalues_func(self, obv):
        return self.qnet(obv)

    def train(self, b_o, b_a, b_r, b_o_, b_d, weights=None):
        if weights is None:
            weights = np.ones_like(b_r)
        td_errors = self._train_func(b_o, b_a, b_r, b_o_, b_d, weights)

        self.niter += 1
        if self.niter % target_q_update_freq == 0:
            sync(self.qnet, self.targetqnet)
            self.save(args.save_path)
        return td_errors.numpy()

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
    def _train_func(self, b_o, b_a, b_r, b_o_, b_d, weights):
        with tf.GradientTape() as tape:
            td_errors = self._tderror_func(b_o, b_a, b_r, b_o_, b_d)
            loss = tf.reduce_mean(huber_loss(td_errors) * weights)

        grad = tape.gradient(loss, self.qnet.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.qnet.trainable_weights))

        return td_errors

    @tf.function
    def _tderror_func(self, b_o, b_a, b_r, b_o_, b_d):
        b_q_ = (1 - b_d) * tf.reduce_max(self.targetqnet(b_o_), 1)
        b_q = tf.reduce_sum(self.qnet(b_o) * tf.one_hot(b_a, out_dim), 1)
        return b_q - (b_r + reward_gamma * b_q_)


# #############################  Trainer  ###################################
if __name__ == '__main__':
    dqn = DQN()
    t0 = time.time()
    if args.train:
        buffer = PrioritizedReplayBuffer(buffer_size, prioritized_replay_alpha, prioritized_replay_beta0)
        nepisode = 0
        all_episode_reward = []
        for i in range(1, number_timesteps + 1):
            o = env.reset()
            episode_reward = 0
            while True:
                buffer.beta += (1 - prioritized_replay_beta0) / number_timesteps

                a = dqn.get_action(o)

                # execute action and feed to replay buffer
                # note that `_` tail in var name means next
                o_, r, done, info = env.step(a)
                buffer.add(o, a, r, o_, done)
                episode_reward += r

                if i >= warm_start:
                    *transitions, idxs = buffer.sample(batch_size)
                    priorities = dqn.train(*transitions)
                    priorities = np.clip(np.abs(priorities), 1e-6, None)
                    buffer.update_priorities(idxs, priorities)

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
