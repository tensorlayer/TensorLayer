"""
Functions for utilization.

# Requirements
tensorflow==2.0.0a0
tensorlayer==2.0.1

"""
import operator
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import tensorlayer as tl
from tensorlayer.layers import Dense, Input
from tensorlayer.models import Model

tfd = tfp.distributions
Normal = tfd.Normal


class ValueNetwork_old(Model):
    ''' Deprecated! network for estimating V(s).'''

    def __init__(self, state_dim, hidden_dim, scope=None):
        super(ValueNetwork_old, self).__init__()
        input_dim = state_dim
        w_init = tf.keras.initializers.glorot_normal(
            seed=None
        )  # glorot initialization is better than uniform in practice
        # init_w=3e-3
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=input_dim, name='v1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='v2')
        self.linear3 = Dense(n_units=1, W_init=w_init, in_channels=hidden_dim, name='v3')

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class ValueNetwork(Model):
    ''' 
    network for estimating V(s),
    one input layer, one output layer, others are hidden layers.
    '''

    def __init__(self, state_dim, hidden_dim, num_hidden_layer, scope=None):
        super(ValueNetwork, self).__init__()
        self.input_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layer = num_hidden_layer
        self.scope = scope

    def model(self):
        w_init = tf.keras.initializers.glorot_normal(
            seed=None
        )  # glorot initialization is better than uniform in practice
        # init_w=3e-3
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

        inputs = Input(
            [None, self.input_dim], name=str(self.scope) + 'q_input' if self.scope is not None else 'q_input'
        )
        l = Dense(
            n_units=self.hidden_dim, act=tf.nn.relu, W_init=w_init,
            name=str(self.scope) + 'v_1' if self.scope is not None else 'v_1'
        )(inputs)
        for i in range(self.num_hidden_layer):
            l = Dense(
                n_units=self.hidden_dim, act=tf.nn.relu, W_init=w_init,
                name=str(self.scope) + 'v_1' + str(i + 2) if self.scope is not None else 'v_1' + str(i + 2)
            )(l)
        outputs = Dense(
            n_units=1, W_init=w_init, name=str(self.scope) + 'v' +
            str(self.num_hidden_layer + 2) if self.scope is not None else 'v' + str(self.num_hidden_layer + 2)
        )(l)

        return tl.models.Model(
            inputs=inputs, outputs=outputs,
            name=str(self.scope) + 'value_network' if self.scope is not None else 'value_network'
        )


class QNetwork_old(Model):
    ''' Deprecated!  network for estimating Q(s,a). '''

    def __init__(self, state_dim, action_dim, hidden_dim, scope=None):
        super(QNetwork_old, self).__init__()
        input_dim = state_dim + action_dim
        w_init = tf.keras.initializers.glorot_normal(
            seed=None
        )  # glorot initialization is better than uniform in practice
        # init_w=3e-3
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='q2')
        self.linear3 = Dense(n_units=1, W_init=w_init, in_channels=hidden_dim, name='q3')

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class QNetwork(Model):
    ''' 
    network for estimating Q(s,a).
    '''

    def __init__(self, state_dim, action_dim, hidden_dim, num_hidden_layer, scope=None):
        super(QNetwork, self).__init__()
        self.input_dim = state_dim + action_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layer = num_hidden_layer

    def model(self):
        w_init = tf.keras.initializers.glorot_normal(
            seed=None
        )  # glorot initialization is better than uniform in practice
        # init_w=3e-3
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

        inputs = Input([None, self.input_dim], name='q_input')
        l = Dense(n_units=self.hidden_dim, act=tf.nn.relu, W_init=w_init, name='q1')(inputs)
        for i in range(self.num_hidden_layer):
            l = Dense(n_units=self.hidden_dim, act=tf.nn.relu, W_init=w_init, name='q' + str(i + 2))(l)
        outputs = Dense(n_units=1, W_init=w_init, name='q' + str(self.num_hidden_layer + 2))(l)

        return tl.models.Model(inputs=inputs, outputs=outputs, name='Q_network')


class StochasticPolicyNetwork_old(Model):
    ''' Deprecated! stochastic continuous policy network for generating action according to the state '''

    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3, log_std_min=-20, log_std_max=2, scope=None):
        super(StochasticPolicyNetwork_old, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        w_init = tf.keras.initializers.glorot_normal(seed=None)
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=state_dim, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')

        self.mean_linear = Dense(n_units=action_dim, W_init=w_init, \
        b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim, name='policy_mean')
        self.log_std_linear = Dense(n_units=action_dim, W_init=w_init, \
        b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim, name='policy_logstd')

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


class StochasticPolicyNetwork(Model):
    ''' stochastic continuous policy network for generating action according to the state '''

    def __init__(self, state_dim, action_dim, hidden_dim, num_hidden_layer, log_std_min=-20, log_std_max=2, scope=None):
        super(StochasticPolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.scope = scope
        self.input_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_hidden_layer = num_hidden_layer

    def model(self):
        w_init = tf.keras.initializers.glorot_normal(
            seed=None
        )  # glorot initialization is better than uniform in practice
        # init_w=3e-3
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

        inputs = Input([None, self.input_dim], name='policy_input')
        l = Dense(
            n_units=self.hidden_dim, act=tf.nn.relu, W_init=w_init,
            name=str(self.scope) + 'policy1' if self.scope is not None else 'policy1'
        )(inputs)
        for i in range(self.num_hidden_layer):
            l = Dense(
                n_units=self.hidden_dim, act=tf.nn.relu, W_init=w_init,
                name=str(self.scope) + 'policy' + str(i + 2) if self.scope is not None else 'policy' + str(i + 2)
            )(l)
        mean_linear = Dense(
            n_units=self.action_dim, W_init=w_init,
            name=str(self.scope) + 'policy_mean' if self.scope is not None else 'policy_mean'
        )(l)
        log_std_linear = Dense(
            n_units=self.action_dim, W_init=w_init,
            name=str(self.scope) + 'policy_std' if self.scope is not None else 'policy_std'
        )(l)
        log_std_linear = tl.layers.Lambda(
            lambda x: tf.clip_by_value(x, self.log_std_min, self.log_std_max),
            name=str(self.scope) + 'lambda' if self.scope is not None else 'lambda'
        )(log_std_linear)
        return tl.models.Model(
            inputs=inputs, outputs=[mean_linear, log_std_linear], name=str(self.scope) +
            'stochastic_policy_network' if self.scope is not None else 'stochastic_policy_network'
        )


class DeterministicPolicyNetwork_old(Model):
    ''' Deprecated! deterministic continuous policy network for generating action according to the state '''

    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3, scope=None):
        super(DeterministicPolicyNetwork_old, self).__init__()

        w_init = tf.keras.initializers.glorot_normal(seed=None)
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=state_dim, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')

        self.action_linear = Dense(n_units=action_dim, W_init=w_init, \
        b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim, name='policy')

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)

        action = self.action_linear(x)

        return action


class DeterministicPolicyNetwork(Model):
    ''' stochastic continuous policy network for generating action according to the state '''

    def __init__(self, state_dim, action_dim, hidden_dim, num_hidden_layer, scope=None):
        super(DeterministicPolicyNetwork, self).__init__()

        self.input_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_hidden_layer = num_hidden_layer

    def model(self):
        w_init = tf.keras.initializers.glorot_normal(
            seed=None
        )  # glorot initialization is better than uniform in practice
        # init_w=3e-3
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

        inputs = Input([None, self.input_dim], name='policy_input')
        l = Dense(n_units=self.hidden_dim, act=tf.nn.relu, W_init=w_init, name='policy1')(inputs)
        for i in range(self.num_hidden_layer):
            l = Dense(n_units=self.hidden_dim, act=tf.nn.relu, W_init=w_init, name='policy' + str(i + 2))(l)
        action_linear = Dense(n_units=self.action_dim, W_init=w_init, name='policy')(l)
        return tl.models.Model(inputs=inputs, outputs=action_linear, name='deterministic_policy_network')
