"""
Trust Region Policy Optimization (TRPO)
---------------------------------------
PG method with a large step can collapse the policy performance,
even with a small step can lead a large differences in policy.
TRPO constraint the step in policy space using KL divergence (rather than in parameter space),
which can monotonically improve performance and avoid a collapsed update.

Reference
---------
Trust Region Policy Optimization, Schulman et al. 2015
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
Approximately Optimal Approximate Reinforcement Learning, Kakade and Langford 2002
openai/spinningup : http://spinningup.openai.com/en/latest/algorithms/trpo.html

Environment
-----------
Openai Gym Pendulum-v0, continual action space

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

To run
------
python tutorial_TRPO.py --train/test

"""
import argparse
import copy
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow_probability as tfp
from gym.spaces import Box, Discrete

import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false')

parser.add_argument('--env', type=str, default='Pendulum-v0')  # environment name
parser.add_argument('--hid', type=int, default=64)  # size of each hidden layer
parser.add_argument('--l', type=int, default=2)  # hidden layer length
parser.add_argument('--gamma', type=float, default=0.99)  # reward discount
parser.add_argument('--seed', '-s', type=int, default=1)  # random seed
parser.add_argument('--steps', type=int, default=4000)  # total number of steps for each episode
parser.add_argument('--epochs', type=int, default=500)  # total number of episodes for training
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_NAME = args.env  # environment name
HIDDEN_SIZES = [args.hid] * args.l  # hidden layer size
SEED = args.seed  # random seed
STEPS_PER_EPOCH = args.steps  # total number of steps for each episode
EPOCHS = args.epochs  # total number of episodes for training
GAMMA = args.gamma  # reward discount

DELTA = 0.01  # KL-divergence limit for TRPO update.
VF_LR = 1e-3  # Learning rate for value function optimizer
TRAIN_V_ITERS = 80  # Number of gradient descent steps to take on value function per epoch
DAMPING_COEFF = 0.1  # Artifact for numerical stability
CG_ITERS = 10  # Number of iterations of conjugate gradient to perform
BACKTRACK_ITERS = 10  # Maximum number of steps allowed in the backtracking line search
BACKTRACK_COEFF = 0.8  # How far back to step during backtracking line search
LAM = 0.97  # Lambda for GAE-Lambda
MAX_EP_LEN = 1000  # Maximum length of trajectory
SAVE_FREQ = 10  # How often (in terms of gap between epochs) to save the current policy and value function
EPS = 1e-8  # epsilon

#####################  functions  ####################


def combined_shape(length, shape=None):
    """
    combine length and shape based on shape type
    :param length: int length
    :param shape: shape, can be either scalar or array
    :return: shape
    """
    if shape is None:
        return length,
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def keys_as_sorted_list(dict):
    """
    sorted keys of the dict
    :param dict: dict input
    :return: sorted key list
    """
    return sorted(list(dict.keys()))


def values_as_sorted_list(dict):
    """
    sorted values of the dict
    :param dict: dict input
    :return: sorted value list
    """
    return [dict[k] for k in keys_as_sorted_list(dict)]


def input_layer(dim=None):
    """
    create tensorlayer input layer from dimension input
    :param dim: dimension int
    :return: tensorlayer input layer
    """
    return tl.layers.Input(dtype=tf.float32, shape=combined_shape(None, dim))


def input_layers(*args):
    """
    create tensorlayer input layers from a list of dimensions
    :param args: a list of dimensions
    :return: list of input layers
    """
    return [input_layer(dim) for dim in args]


def input_layer_from_space(space):
    """
    create tensorlayer input layers from env.space input
    :param space: env.space
    :return: tensorlayer input layer
    """
    if isinstance(space, Box):
        return input_layer(space.shape)
    elif isinstance(space, Discrete):
        return tl.layers.Input(dtype=tf.int32, shape=(None, ))
    raise NotImplementedError


def input_layers_from_spaces(*args):
    """
    create tensorlayer input layers from a list of env.space inputs
    :param args: a list of env.space inputs
    :return: tensorlayer input layer list
    """
    return [input_layer_from_space(space) for space in args]


def mlp(x, hidden_sizes=(32, ), activation=tf.tanh, output_activation=None):
    """
    create Multi-Layer Perception
    :param x: tensorlayer input layer
    :param hidden_sizes: hidden layer size
    :param activation: hidden layer activation function
    :param output_activation: activation function for the output layer
    :return: output layer
    """
    for h in hidden_sizes[:-1]:
        x = tl.layers.Dense(n_units=h, act=activation)(x)
    return tl.layers.Dense(n_units=hidden_sizes[-1], act=output_activation)(x)


def get_vars(model: tl.models.Model):
    """
    get trainable parameters of the model
    :param model: tensorlayer model
    :return: a list of trainable parameters of the model
    """
    return model.trainable_weights


def count_vars(model: tl.models.Model):
    """
    count trainable parameters of the model
    :param model: tensorlayer model
    :return: counts
    """
    v = get_vars(model)
    return sum([np.prod(var.shape.as_list()) for var in v])


def gaussian_likelihood(x, mu, log_std):
    """
    calculate gaussian likelihood
    :param x: input distribution
    :param mu: mu
    :param log_std: log std
    :return: gaussian likelihood
    """
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS))**2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
    """
    tf symbol for mean KL divergence between two batches of diagonal gaussian distributions,
    where distributions are specified by means and log stds.
    (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
    """
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    pre_sum = 0.5 * (((mu1 - mu0)**2 + var0) / (var1 + EPS) - 1) + log_std1 - log_std0
    all_kls = tf.reduce_sum(pre_sum, axis=1)
    return tf.reduce_mean(all_kls)


def categorical_kl(logp0, logp1):
    """
    tf symbol for mean KL divergence between two batches of categorical probability distributions,
    where the distributions are input as log probs.
    """
    all_kls = tf.reduce_sum(tf.exp(logp1) * (logp1 - logp0), axis=1)
    return tf.reduce_mean(all_kls)


def flat_concat(xs):
    """
    flat concat input
    :param xs: a list of tensor
    :return: flat tensor
    """
    return tf.concat([tf.reshape(x, (-1, )) for x in xs], axis=0)


def assign_params_from_flat(x, params):
    """
    assign params from flat input
    :param x:
    :param params:
    :return: group
    """
    flat_size = lambda p: int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([p.assign(p_new) for p, p_new in zip(params, new_params)])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""


class MlpCategoricalPolicy:
    """
    Categorical Policy for discrete input
    """

    def __init__(self, x, a, hidden_sizes, activation, output_activation):
        self.act_dim = a.n
        x = input_layer_from_space(x)
        logits = mlp(x, list(hidden_sizes) + [self.act_dim], activation, None)
        self.model = tl.models.Model(x, logits)
        self.model.train()

    def cal_outputs_0(self, states):
        states = states.astype(np.float32)
        logits = self.model(states)
        logp_all = tf.nn.log_softmax(logits)
        pi = tf.squeeze(tfp.distributions.Multinomial(1, logits), axis=1)
        logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=self.act_dim) * logp_all, axis=1)
        info = {'logp_all': logp_all}
        return pi, logp_pi, info, logp_all

    def cal_outputs_1(self, states, actions, old_logp_all):
        pi, logp_pi, info, logp_all = self.cal_outputs_0(states)
        logp = tf.reduce_sum(tf.one_hot(actions, depth=self.act_dim) * logp_all, axis=1)
        d_kl = categorical_kl(logp_all, old_logp_all)

        info_phs = {'logp_all': old_logp_all}

        return pi, logp, logp_pi, info, info_phs, d_kl


class MlpGaussianPolicy:
    """
    Gaussian Policy for continuous input
    """

    def __init__(self, x, a, hidden_sizes, activation, output_activation):
        act_dim = a.shape[0]

        x = input_layer_from_space(x)
        mu = mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
        self.model = tl.models.Model(x, mu)
        self.model.train()

        self._log_std = tf.Variable(-0.5 * np.ones(act_dim, dtype=np.float32))
        self.model.trainable_weights.append(self._log_std)

    def cal_outputs_0(self, states):
        states = states.astype(np.float32)
        mu = self.model(states)
        std = tf.exp(self._log_std)
        pi = mu + tf.random.normal(tf.shape(mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, self._log_std)

        info = {'mu': mu, 'log_std': self._log_std}

        return pi, logp_pi, info, mu, self._log_std

    def cal_outputs_1(self, states, actions, old_log_std_ph, old_mu_ph):
        pi, logp_pi, info, mu, log_std = self.cal_outputs_0(states)
        logp = gaussian_likelihood(actions, mu, log_std)
        d_kl = diagonal_gaussian_kl(mu, log_std, old_mu_ph, old_log_std_ph)

        info_phs = {'mu': old_mu_ph, 'log_std': old_log_std_ph}

        return pi, logp, logp_pi, info, info_phs, d_kl


"""
Actor-Critics
"""


def mlp_actor_critic(
        x: 'env.observation_space', a: 'env.action_space', hidden_sizes=(64, 64), activation=tf.tanh,
        output_activation=None
):
    """
    create actor and critic
    :param x: observation space
    :param a: action space
    :param hidden_sizes: hidden layer size
    :param activation: hidden layer activation function
    :param output_activation: activation function for the output layer
    :return: acter class and critic class
    """
    # default policy builder depends on action space
    if isinstance(a, Box):
        actor = MlpGaussianPolicy(x, a, hidden_sizes, activation, output_activation)
    elif isinstance(a, Discrete):
        actor = MlpCategoricalPolicy(x, a, hidden_sizes, activation, output_activation)
    else:
        raise ValueError('action space type error')

    class Critic:

        def __init__(self, obs_space, hidden_layer_sizes, activation_funcs):
            inputs = input_layer_from_space(obs_space)
            self.model = tl.models.Model(inputs, mlp(inputs, list(hidden_layer_sizes) + [1], activation_funcs, None))
            self.model.train()

        def critic_cal_func(self, states):
            states = states.astype(np.float32)
            return tf.squeeze(self.model(states), axis=1)

    critic = Critic(x, hidden_sizes, activation)

    return actor, critic


class GAEBuffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.info_bufs = {k: np.zeros([size] + list(v), dtype=np.float32) for k, v in info_shapes.items()}
        self.sorted_info_keys = keys_as_sorted_list(self.info_bufs)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, info):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        for i, k in enumerate(self.sorted_info_keys):
            self.info_bufs[k][self.ptr] = info[i]
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf
               ] + values_as_sorted_list(self.info_bufs)


#####################  TRPO  ####################
"""

Trust Region Policy Optimization 

(with support for Natural Policy Gradient)

"""


class TRPO:
    """
    trpo class
    """

    def __init__(self, obs_space, act_space):

        obs_dim = obs_space.shape
        act_dim = act_space.shape

        # # Main models and functions
        self.actor, self.critic = mlp_actor_critic(obs_space, act_space, HIDDEN_SIZES)

        if isinstance(act_space, Box):
            act_dim = env.action_space.shape[0]
            info_shapes = {'mu': [act_dim], 'log_std': [act_dim]}

        elif isinstance(env.action_space, Discrete):
            act_dim = env.action_space.n
            info_shapes = {'logp_all': [act_dim]}
        else:
            raise Exception('info_shape error')

        self.buf = GAEBuffer(obs_dim, act_dim, STEPS_PER_EPOCH, info_shapes, GAMMA, LAM)

        # Optimizer for value function
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=VF_LR)

    # Every step, get: action, value, logprob, & info for pdist (for computing kl div)
    def get_action_ops(self, states):
        """
        get action
        :param states: state input
        :return: pi, v, logp_pi and other outputs
        """
        pi, logp_pi, info, *_ = self.actor.cal_outputs_0(states)
        v = self.critic.critic_cal_func(states)
        res0 = [pi, v, logp_pi] + values_as_sorted_list(info)
        res = []
        for i in res0:
            res.append(i + 0)  # transfer to tensor
        return res

    # TRPO losses
    def pi_loss(self, inputs):
        """
        calculate pi loss
        :param inputs: a list of x_ph, a_ph, adv_ph, ret_ph, logp_old_ph and other inputs
        :return: pi loss
        """
        x_ph, a_ph, adv_ph, ret_ph, logp_old_ph, *info_values = inputs

        pi, logp, logp_pi, info, info_phs, d_kl = self.actor.cal_outputs_1(x_ph, a_ph, *info_values)
        ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
        pi_loss = -tf.reduce_mean(ratio * adv_ph)
        return pi_loss

    def v_loss(self, inputs):
        """
        calculate value loss
        :param inputs: a list of x_ph, a_ph, adv_ph, ret_ph, logp_old_ph and other inputs
        :return: v loss
        """
        x_ph, a_ph, adv_ph, ret_ph, logp_old_ph, *info_values = inputs
        v = self.critic.critic_cal_func(x_ph)
        v_loss = tf.reduce_mean((ret_ph - v)**2)
        return v_loss

    def train_vf(self, inputs):
        """
        train v function
        :param inputs: a list of x_ph, a_ph, adv_ph, ret_ph, logp_old_ph and other inputs
        :return: None
        """
        with tf.GradientTape() as tape:
            loss = self.v_loss(inputs)
        grad = tape.gradient(loss, self.critic.model.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.model.trainable_weights))

    # Symbols needed for CG solver
    def gradient(self, inputs):
        """
        pi gradients
        :param inputs: a list of x_ph, a_ph, adv_ph, ret_ph, logp_old_ph and other inputs
        :return: gradient
        """
        pi_params = self.actor.model.trainable_weights
        with tf.GradientTape() as tape:
            loss = self.pi_loss(inputs)
        grad = tape.gradient(loss, pi_params)
        gradient = flat_concat(grad)
        return gradient

    def hvp(self, inputs, v_ph):
        """
        calculate hvp
        :param inputs: a list of x_ph, a_ph, adv_ph, ret_ph, logp_old_ph and other inputs
        :param v_ph: v input
        :return: hvp
        """
        pi_params = self.actor.model.trainable_weights
        x_ph, a_ph, adv_ph, ret_ph, logp_old_ph, *info_values = inputs

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape0:
                pi, logp, logp_pi, info, info_phs, d_kl = self.actor.cal_outputs_1(x_ph, a_ph, *info_values)
            g = flat_concat(tape0.gradient(d_kl, pi_params))
            l = tf.reduce_sum(g * v_ph)
        hvp = flat_concat(tape1.gradient(l, pi_params))

        if DAMPING_COEFF > 0:
            hvp += DAMPING_COEFF * v_ph
        return hvp

    # Symbols for getting and setting params
    def get_pi_params(self):
        """
        get actor trainable parameters
        :return: flat actor trainable parameters
        """
        pi_params = self.actor.model.trainable_weights
        return flat_concat(pi_params)

    def set_pi_params(self, v_ph):
        """
        set actor trainable parameters
        :param v_ph: inputs
        :return: None
        """
        pi_params = self.actor.model.trainable_weights
        assign_params_from_flat(v_ph, pi_params)

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_weights_to_hdf5('model/trpo_actor.hdf5', self.actor.model)
        tl.files.save_weights_to_hdf5('model/trpo_critic.hdf5', self.critic.model)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/trpo_actor.hdf5', self.actor.model)
        tl.files.load_hdf5_to_weights_in_order('model/trpo_critic.hdf5', self.critic.model)

    def cg(self, Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = np.zeros_like(b)
        r = copy.deepcopy(b)  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = copy.deepcopy(r)
        r_dot_old = np.dot(r, r)
        for _ in range(CG_ITERS):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update(self):
        """
        update trpo
        :return:
        """
        # Prepare hessian func, gradient eval
        inputs = self.buf.get()
        Hx = lambda x: self.hvp(inputs, x)
        g, pi_l_old, v_l_old = self.gradient(inputs), self.pi_loss(inputs), self.v_loss(inputs)

        # Core calculations for TRPO or NPG
        x = self.cg(Hx, g)
        alpha = np.sqrt(2 * DELTA / (np.dot(x, Hx(x)) + EPS))
        old_params = self.get_pi_params()

        def set_and_eval(step):
            aa = alpha * x * step
            par = old_params - aa
            self.set_pi_params(par)
            x_ph, a_ph, adv_ph, ret_ph, logp_old_ph, *info_values = inputs
            pi, logp, logp_pi, info, info_phs, d_kl = self.actor.cal_outputs_1(x_ph, a_ph, *info_values)
            loss = self.pi_loss(inputs)
            return [d_kl, loss]

        # trpo augments npg with backtracking line search, hard kl
        for j in range(BACKTRACK_ITERS):
            kl, pi_l_new = set_and_eval(step=BACKTRACK_COEFF**j)
            if kl <= DELTA and pi_l_new <= pi_l_old:
                # Accepting new params at step of line search
                break

            if j == BACKTRACK_ITERS - 1:
                # Line search failed! Keeping old params.
                kl, pi_l_new = set_and_eval(step=0.)

        # Value function updates
        for _ in range(TRAIN_V_ITERS):
            self.train_vf(inputs)


if __name__ == '__main__':

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    env = gym.make(ENV_NAME)
    env.seed(SEED)

    agent = TRPO(env.observation_space, env.action_space)

    if args.train:
        start_time = time.time()
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        reward_list = []
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(EPOCHS):
            t0 = time.time()
            rew = 0
            for t in range(STEPS_PER_EPOCH):
                agent_outs = agent.get_action_ops(o.reshape(1, -1))
                a, v_t, logp_t, info_t = np.array(agent_outs[0][0], np.float32), \
                                         np.array(agent_outs[1], np.float32), \
                                         np.array(agent_outs[2], np.float32), \
                                         np.array(agent_outs[3:], np.float32)

                # save and log
                agent.buf.store(o, a, r, v_t, logp_t, info_t)

                o, r, d, _ = env.step(a)
                ep_ret += r
                ep_len += 1

                terminal = d or (ep_len == MAX_EP_LEN)
                if terminal or (t == STEPS_PER_EPOCH - 1):
                    if not (terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = r if d else agent.critic.critic_cal_func(o.reshape(1, -1))
                    agent.buf.finish_path(last_val)
                    rew = ep_ret
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            # Save model
            if (epoch % SAVE_FREQ == 0) or (epoch == EPOCHS - 1):
                agent.save_ckpt()

            # Perform TRPO or NPG update!
            agent.update()
            print('epoch [{}/{}] ep_ret: {} time: {}'.format(epoch, EPOCHS, rew, time.time() - t0))

            reward_list.append(rew)
            plt.clf()
            plt.ion()
            plt.plot(reward_list)
            plt.title('TRPO ' + str(DELTA))
            plt.ylim(-2000, 0)
            plt.show()
            plt.pause(0.1)
        agent.save_ckpt()
        plt.ioff()
        plt.show()

    # test
    agent.load_ckpt()
    while True:
        o = env.reset()
        for i in range(STEPS_PER_EPOCH):
            env.render()
            agent_outs = agent.get_action_ops(o.reshape(1, -1))
            a, v_t, logp_t, info_t = agent_outs[0][0], agent_outs[1], agent_outs[2], agent_outs[3:]
            o, r, d, _ = env.step(a)
            if d:
                break
