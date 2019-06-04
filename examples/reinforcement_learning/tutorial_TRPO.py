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

Env
---
Openai Gym Pendulum-v0, continual action space

To run
------
python *.py


"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
import gym
import time
import os

import matplotlib.pyplot as plt
import scipy.signal
import copy
from gym.spaces import Box, Discrete

EPS = 1e-8


def combined_shape(length, shape=None):
    if shape is None:
        return length,
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))


def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]


def input_layer(dim=None):
    return tl.layers.Input(dtype=tf.float32, shape=combined_shape(None, dim))


def input_layers(*args):
    return [input_layer(dim) for dim in args]


def input_layer_from_space(space):
    if isinstance(space, Box):
        return input_layer(space.shape)
    elif isinstance(space, Discrete):
        return tl.layers.Input(dtype=tf.int32, shape=(None,))
    raise NotImplementedError


def input_layers_from_spaces(*args):
    return [input_layer_from_space(space) for space in args]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tl.layers.Dense(n_units=h, act=activation)(x)
    return tl.layers.Dense(n_units=hidden_sizes[-1], act=output_activation)(x)


def get_vars(model: tl.models.Model):
    return model.trainable_weights


def count_vars(model: tl.models.Model):
    v = get_vars(model)
    return sum([np.prod(var.shape.as_list()) for var in v])


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
    """
    tf symbol for mean KL divergence between two batches of diagonal gaussian distributions,
    where distributions are specified by means and log stds.
    (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
    """
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    pre_sum = 0.5 * (((mu1 - mu0) ** 2 + var0) / (var1 + EPS) - 1) + log_std1 - log_std0
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
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)


def flat_grad(f, params):
    return flat_concat(tf.gradients(xs=params, ys=f))


def hessian_vector_product(f, params, x):
    # for H = grad**2 f, compute Hx
    g = flat_grad(f, params)
    return flat_grad(tf.reduce_sum(g * x), params)


def assign_params_from_flat(x, params):
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


def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.n

    x = input_layer_from_space(x)
    logits = mlp(x, list(hidden_sizes) + [act_dim], activation, None)
    actor = tl.models.Model(x, logits)

    def cal_outputs_0(states):
        states = states.astype(np.float32)
        logits = actor(states)
        logp_all = tf.nn.log_softmax(logits)
        pi = tf.squeeze(tfp.distributions.Multinomial(1, logits), axis=1)
        logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
        info = {'logp_all': logp_all}
        return pi, logp_pi, info, logp_all

    def cal_outputs_1(states, actions, old_logp_all):
        pi, logp_pi, info, logp_all = cal_outputs_0(states)
        logp = tf.reduce_sum(tf.one_hot(actions, depth=act_dim) * logp_all, axis=1)
        d_kl = categorical_kl(logp_all, old_logp_all)

        info_phs = {'logp_all': old_logp_all}

        return pi, logp, logp_pi, info, info_phs, d_kl

    return actor, cal_outputs_0, cal_outputs_1


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape[0]

    x = input_layer_from_space(x)
    mu = mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
    actor = tl.models.Model(x, mu)

    def cal_outputs_0(states):
        states = states.astype(np.float32)
        mu = actor(states)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        std = tf.exp(log_std)
        pi = mu + tf.random.normal(tf.shape(mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)

        info = {'mu': mu, 'log_std': log_std}

        return pi, logp_pi, info, mu, log_std

    def cal_outputs_1(states, actions, old_log_std_ph, old_mu_ph):
        pi, logp_pi, info, mu, log_std = cal_outputs_0(states)
        logp = gaussian_likelihood(actions, mu, log_std)
        d_kl = diagonal_gaussian_kl(mu, log_std, old_mu_ph, old_log_std_ph)

        info_phs = {'mu': old_mu_ph, 'log_std': old_log_std_ph}

        return pi, logp, logp_pi, info, info_phs, d_kl

    return actor, cal_outputs_0, cal_outputs_1


"""
Actor-Critics
"""


def mlp_actor_critic(x: 'env.observation_space', a: 'env.action_space', hidden_sizes=(64, 64), activation=tf.tanh,
                     output_activation=None, policy=None):
    # default policy builder depends on action space
    if policy is None and isinstance(a, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(a, Discrete):
        policy = mlp_categorical_policy

    actor, actor_cal_func_0, actor_cal_func_1 = policy(x, a, hidden_sizes, activation, output_activation)

    x = input_layer_from_space(x)
    critic = tl.models.Model(x, mlp(x, list(hidden_sizes) + [1], activation, None))

    actor.train()
    critic.train()

    def critic_cal_func(states):
        states = states.astype(np.float32)
        return tf.squeeze(critic(states), axis=1)

    return actor, actor_cal_func_0, actor_cal_func_1, critic, critic_cal_func


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
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
                self.logp_buf] + values_as_sorted_list(self.info_bufs)


"""

Trust Region Policy Optimization 

(with support for Natural Policy Gradient)

"""


def trpo(env_fn, actor_critic=mlp_actor_critic, ac_kwargs=dict(), seed=1,
         steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, save_freq=10, algo='trpo'):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ============  ================  ========================================
            Symbol        Shape             Description
            ============  ================  ========================================
            ``pi``        (batch, act_dim)  | Samples actions from policy given 
                                            | states.
            ``logp``      (batch,)          | Gives log probability, according to
                                            | the policy, of taking actions ``a_ph``
                                            | in states ``x_ph``.
            ``logp_pi``   (batch,)          | Gives log probability, according to
                                            | the policy, of the action sampled by
                                            | ``pi``.
            ``info``      N/A               | A dict of any intermediate quantities
                                            | (from calculating the policy or log 
                                            | probabilities) which are needed for
                                            | analytically computing KL divergence.
                                            | (eg sufficient statistics of the
                                            | distributions)
            ``info_phs``  N/A               | A dict of placeholders for old values
                                            | of the entries in ``info``.
            ``d_kl``      ()                | A symbol for computing the mean KL
                                            | divergence between the current policy
                                            | (``pi``) and the old policy (as 
                                            | specified by the inputs to 
                                            | ``info_phs``) over the batch of 
                                            | states given in ``x_ph``.
            ``v``         (batch,)          | Gives the value estimate for states
                                            | in ``x_ph``. (Critical: make sure 
                                            | to flatten this!)
            ============  ================  ========================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to TRPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        delta (float): KL-divergence limit for TRPO / NPG update. 
            (Should be small for stability. Values like 0.01, 0.05.)

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        damping_coeff (float): Artifact for numerical stability, should be 
            smallish. Adjusts Hessian-vector product calculation:
            
            .. math:: Hv \\rightarrow (\\alpha I + H)v

            where :math:`\\alpha` is the damping coefficient. 
            Probably don't play with this hyperparameter.

        cg_iters (int): Number of iterations of conjugate gradient to perform. 
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down. 

            Also probably don't play with this hyperparameter.

        backtrack_iters (int): Maximum number of steps allowed in the 
            backtracking line search. Since the line search usually doesn't 
            backtrack, and usually only steps back once when it does, this
            hyperparameter doesn't often matter.

        backtrack_coeff (float): How far back to step during backtracking line
            search. (Always between 0 and 1, usually above 0.5.)

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        algo: Either 'trpo' or 'npg': this code supports both, since they are 
            almost the same.

    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Main models and functions
    actor, actor_cal_func_0, actor_cal_func_1, critic, critic_cal_func = \
        actor_critic(env.observation_space, env.action_space)

    # Every step, get: action, value, logprob, & info for pdist (for computing kl div)
    def get_action_ops(states):
        pi, logp_pi, info, *_ = actor_cal_func_0(states)
        v = critic_cal_func(states)
        return [pi, v, logp_pi] + values_as_sorted_list(info)

    # Experience buffer
    local_steps_per_epoch = steps_per_epoch

    if isinstance(env.action_space, Box):
        act_dim = env.action_space.shape[0]
        info_shapes = {'mu': [act_dim], 'log_std': [act_dim]}

    elif isinstance(env.action_space, Discrete):
        act_dim = env.action_space.n
        info_shapes = {'logp_all': [act_dim]}
    else:
        raise Exception('info_shape error')

    buf = GAEBuffer(obs_dim, act_dim, local_steps_per_epoch, info_shapes, gamma, lam)

    # TRPO losses
    def pi_loss(inputs):
        x_ph, a_ph, adv_ph, ret_ph, logp_old_ph, *info_values = inputs

        pi, logp, logp_pi, info, info_phs, d_kl = actor_cal_func_1(x_ph, a_ph, *info_values)
        ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
        pi_loss = -tf.reduce_mean(ratio * adv_ph)
        return pi_loss

    def v_loss(inputs):
        x_ph, a_ph, adv_ph, ret_ph, logp_old_ph, *info_values = inputs
        v = critic_cal_func(x_ph)
        v_loss = tf.reduce_mean((ret_ph - v) ** 2)
        return v_loss

    # Optimizer for value function
    critic_optimizer = tf.optimizers.Adam(learning_rate=vf_lr)

    def train_vf(inputs):
        with tf.GradientTape() as tape:
            loss = v_loss(inputs)
        grad = tape.gradient(loss, critic.trainable_weights)
        critic_optimizer.apply_gradients(zip(grad, critic.trainable_weights))

    # Symbols needed for CG solver
    def gradient(inputs):
        pi_params = actor.trainable_weights
        with tf.GradientTape() as tape:
            loss = pi_loss(inputs)
        grad = tape.gradient(loss, pi_params)
        gradient = flat_concat(grad)
        return gradient

    def hvp(inputs, v_ph):
        pi_params = actor.trainable_weights
        x_ph, a_ph, adv_ph, ret_ph, logp_old_ph, *info_values = inputs

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape0:
                pi, logp, logp_pi, info, info_phs, d_kl = actor_cal_func_1(x_ph, a_ph, *info_values)
            g = flat_concat(tape0.gradient(d_kl, pi_params))
            l = tf.reduce_sum(g * v_ph)
        hvp = flat_concat(tape1.gradient(l, pi_params))

        if damping_coeff > 0:
            hvp += damping_coeff * v_ph
        return hvp

    # Symbols for getting and setting params
    def get_pi_params():
        pi_params = actor.trainable_weights
        return flat_concat(pi_params)

    def set_pi_params(v_ph):
        pi_params = actor.trainable_weights
        assign_params_from_flat(v_ph, pi_params)

    def save_ckpt():
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_weights_to_hdf5('model/trpo_actor.hdf5', actor)
        tl.files.save_weights_to_hdf5('model/trpo_critic.hdf5', critic)

    def load_ckpt():
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/trpo_actor.hdf5', actor)
        tl.files.load_hdf5_to_weights_in_order('model/trpo_critic.hdf5', critic)

    def cg(Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = np.zeros_like(b)
        r = copy.deepcopy(b)  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = copy.deepcopy(r)
        r_dot_old = np.dot(r, r)
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update():
        # Prepare hessian func, gradient eval
        inputs = buf.get()
        ''''all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph] + values_as_sorted_list(info_phs)'''
        Hx = lambda x: hvp(inputs, x)
        g, pi_l_old, v_l_old = gradient(inputs), pi_loss(inputs), v_loss(inputs)

        # Core calculations for TRPO or NPG
        x = cg(Hx, g)
        alpha = np.sqrt(2 * delta / (np.dot(x, Hx(x)) + EPS))
        old_params = get_pi_params()

        def set_and_eval(step):
            set_pi_params(old_params - alpha * x * step)
            x_ph, a_ph, adv_ph, ret_ph, logp_old_ph, *info_values = inputs
            pi, logp, logp_pi, info, info_phs, d_kl = actor_cal_func_1(x_ph, a_ph, *info_values)
            loss = pi_loss(inputs)
            return [d_kl, loss]

        if algo == 'npg':
            # npg has no backtracking or hard kl constraint enforcement
            kl, pi_l_new = set_and_eval(step=1.)

        elif algo == 'trpo':
            # trpo augments npg with backtracking line search, hard kl
            for j in range(backtrack_iters):
                kl, pi_l_new = set_and_eval(step=backtrack_coeff ** j)
                if kl <= delta and pi_l_new <= pi_l_old:
                    # Accepting new params at step of line search
                    break

                if j == backtrack_iters - 1:
                    # Line search failed! Keeping old params.
                    kl, pi_l_new = set_and_eval(step=0.)

        # Value function updates
        for _ in range(train_v_iters):
            train_vf(inputs)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    reward_list = []
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        t0 = time.time()
        rew = 0
        for t in range(local_steps_per_epoch):
            agent_outs = get_action_ops(o.reshape(1, -1))
            a, v_t, logp_t, info_t = np.array(agent_outs[0][0], np.float32), \
                                     np.array(agent_outs[1], np.float32), \
                                     np.array(agent_outs[2], np.float32), \
                                     np.array(agent_outs[3:], np.float32)

            # store
            buf.store(o, a, r, v_t, logp_t, info_t)

            o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == local_steps_per_epoch - 1):
                if not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else critic_cal_func(o.reshape(1, -1))
                buf.finish_path(last_val)
                rew = ep_ret
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            save_ckpt()

        # Perform TRPO or NPG update!
        update()
        print('epoch [{}/{}] ep_ret: {} time: {}'.format(epoch, epochs, rew, time.time() - t0))

        reward_list.append(rew)
        plt.clf()
        plt.ion()
        plt.plot(reward_list)
        plt.title('TRPO' + str(delta))
        plt.ylim(-2000, 0)
        plt.show()
        plt.pause(0.1)

    plt.ioff()
    plt.show()
    while True:
        o = env.reset()
        for i in range(200):
            env.render()
            agent_outs = get_action_ops(o.reshape(1, -1))
            a, v_t, logp_t, info_t = agent_outs[0][0], agent_outs[1], agent_outs[2], agent_outs[3:]
            o, r, d, _ = env.step(a)
            if d:
                break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()

    trpo(lambda: gym.make(args.env), actor_critic=mlp_actor_critic,
         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)
