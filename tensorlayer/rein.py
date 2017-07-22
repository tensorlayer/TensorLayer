#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
import numpy as np
from six.moves import xrange

def discount_episode_rewards(rewards=[], gamma=0.99, mode=0):
    """ Take 1D float array of rewards and compute discounted rewards for an
    episode. When encount a non-zero value, consider as the end a of an episode.

    Parameters
    ----------
    rewards : numpy list
        a list of rewards
    gamma : float
        discounted factor
    mode : int
        if mode == 0, reset the discount process when encount a non-zero reward (Ping-pong game).
        if mode == 1, would not reset the discount process.

    Examples
    ----------
    >>> rewards = np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    >>> gamma = 0.9
    >>> discount_rewards = tl.rein.discount_episode_rewards(rewards, gamma)
    >>> print(discount_rewards)
    ... [ 0.72899997  0.81        0.89999998  1.          0.72899997  0.81
    ... 0.89999998  1.          0.72899997  0.81        0.89999998  1.        ]
    >>> discount_rewards = tl.rein.discount_episode_rewards(rewards, gamma, mode=1)
    >>> print(discount_rewards)
    ... [ 1.52110755  1.69011939  1.87791049  2.08656716  1.20729685  1.34144104
    ... 1.49048996  1.65610003  0.72899997  0.81        0.89999998  1.        ]
    """
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        if mode == 0:
            if rewards[t] != 0: running_add = 0

        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r


def cross_entropy_reward_loss(logits, actions, rewards, name=None):
    """ Calculate the loss for Policy Gradient Network.

    Parameters
    ----------
    logits : tensor
        The network outputs without softmax. This function implements softmax
        inside.
    actions : tensor/ placeholder
        The agent actions.
    rewards : tensor/ placeholder
        The rewards.

    Examples
    ----------
    >>> states_batch_pl = tf.placeholder(tf.float32, shape=[None, D])
    >>> network = InputLayer(states_batch_pl, name='input')
    >>> network = DenseLayer(network, n_units=H, act=tf.nn.relu, name='relu1')
    >>> network = DenseLayer(network, n_units=3, name='out')
    >>> probs = network.outputs
    >>> sampling_prob = tf.nn.softmax(probs)
    >>> actions_batch_pl = tf.placeholder(tf.int32, shape=[None])
    >>> discount_rewards_batch_pl = tf.placeholder(tf.float32, shape=[None])
    >>> loss = tl.rein.cross_entropy_reward_loss(probs, actions_batch_pl, discount_rewards_batch_pl)
    >>> train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)
    """

    try: # TF 1.0+
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits, name=name)
    except:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, targets=actions)
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, actions)

    try: ## TF1.0+
        loss = tf.reduce_sum(tf.multiply(cross_entropy, rewards))
    except: ## TF0.12
        loss = tf.reduce_sum(tf.mul(cross_entropy, rewards))   # element-wise mul
    return loss

def log_weight(probs, weights, name='log_weight'):
    """Log weight.

    Parameters
    -----------
    probs : tensor
        If it is a network output, usually we should scale it to [0, 1] via softmax.
    weights : tensor
    """
    with tf.variable_scope(name):
        exp_v = tf.reduce_mean(tf.log(probs) * weights)
        return exp_v



def choice_action_by_probs(probs=[0.5, 0.5], action_list=None):
    """Choice and return an an action by given the action probability distribution.

    Parameters
    ------------
    probs : a list of float.
        The probability distribution of all actions.
    action_list : None or a list of action in integer, string or others.
        If None, returns an integer range between 0 and len(probs)-1.

    Examples
    ----------
    >>> for _ in range(5):
    >>>     a = choice_action_by_probs([0.2, 0.4, 0.4])
    >>>     print(a)
    ... 0
    ... 1
    ... 1
    ... 2
    ... 1
    >>> for _ in range(3):
    >>>     a = choice_action_by_probs([0.5, 0.5], ['a', 'b'])
    >>>     print(a)
    ... a
    ... b
    ... b
    """
    if action_list is None:
        n_action = len(probs)
        action_list = np.arange(n_action)
    else:
        assert len(action_list) == len(probs), "Number of actions should equal to number of probabilities."
    return np.random.choice(action_list, p=probs)
