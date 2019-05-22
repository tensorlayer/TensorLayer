"""Implement prioritized replay
Schaul T, Quan J, Antonoglou I, et al. Prioritized experience replay[J]. arXiv
preprint arXiv:1511.05952, 2015.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""
import operator
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
prioritized_replay_alpha = 0.6
prioritized_replay_beta0 = 0.4


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
                return self._reduce_helper(start, end,
                                           2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid,
                                        2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end,
                                        2 * node + 1, mid + 1, node_end)
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
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

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
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

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
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

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
        max_weight = (p_min * len(self._storage)) ** (-self.beta)

        p_samples = np.asarray([self._it_sum[idx] for idx in idxes]) / it_sum
        weights = (p_samples * len(self._storage)) ** (-self.beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample + (weights, idxes)

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions"""
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


def huber_loss(x):
    """Loss function for value"""
    return tf.where(tf.abs(x) < 1, tf.square(x) * 0.5, tf.abs(x) - 0.5)


def sync(net, net_tar):
    """Copy q network to target q network"""
    for var, var_tar in zip(net.trainable_weights, net_tar.trainable_weights):
        var_tar.assign(var)


qnet = MLP('q') if qnet_type == 'MLP' else CNN('q')
qnet.train()
trainabel_weights = qnet.trainable_weights
targetqnet = MLP('targetq') if qnet_type == 'MLP' else CNN('targetq')
targetqnet.infer()
sync(qnet, targetqnet)
optimizer = tf.optimizers.Adam(learning_rate=lr)
buffer = PrioritizedReplayBuffer(
    buffer_size, prioritized_replay_alpha, prioritized_replay_beta0)

o = env.reset()
nepisode = 0
t = time.time()
for i in range(1, number_timesteps + 1):
    eps = epsilon(i)
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
    buffer.add(o, a, r, o_, done)

    if i >= warm_start:
        # sync q net and target q net
        if i % target_q_update_freq == 0:
            sync(qnet, targetqnet)

        # sample from replay buffer
        b_o, b_a, b_r, b_o_, b_d, weights, idxs = buffer.sample(batch_size)

        # q estimation
        b_q_ = (1 - b_d) * tf.reduce_max(targetqnet(b_o_), 1)

        # calculate loss
        with tf.GradientTape() as q_tape:
            b_q = tf.reduce_sum(qnet(b_o) * tf.one_hot(b_a, out_dim), 1)
            abs_td_error = tf.abs(b_q - (b_r + reward_gamma * b_q_))
            priorities = np.clip(abs_td_error.numpy(), 1e-6, None)
            buffer.update_priorities(idxs, priorities)
            loss = tf.reduce_mean(weights * huber_loss(abs_td_error))

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
