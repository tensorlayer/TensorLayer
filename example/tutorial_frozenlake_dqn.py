import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

""" Q-Network Q(a, s) - TD Learning, Off-Policy, e-Greedy Exploration (GLIE)

Q(S, A) <- Q(S, A) + alpha * (R + lambda * Q(newS, newA) - Q(S, A))
delta_w = R + lambda * Q(newS, newA)

See David Silver RL Tutorial Lecture 5 - Q-Learning for more details.

EN: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.5m3361vlw
CN: https://zhuanlan.zhihu.com/p/25710327

Note: Policy Network has been proved to be better than Q-Learning, see tutorial_atari_pong.py
"""
## The FrozenLake v0 environment
# https://gym.openai.com/envs/FrozenLake-v0
# The agent controls the movement of a character in a grid world. Some tiles of
# the grid are walkable, and others lead to the agent falling into the water.
# Additionally, the movement direction of the agent is uncertain and only partially
# depends on the chosen direction. The agent is rewarded for finding a walkable
# path to a goal tile.
# SFFF       (S: starting point, safe)
# FHFH       (F: frozen surface, safe)
# FFFH       (H: hole, fall to your doom)
# HFFG       (G: goal, where the frisbee is located)
# The episode ends when you reach the goal or fall in a hole. You receive a reward
# of 1 if you reach the goal, and zero otherwise.
env = gym.make('FrozenLake-v0')

def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a

render = False           # display the game environment
running_reward = None

tf.reset_default_graph()
## Define Q-network q(a,s) that ouput the rewards of 4 actions by given state, i.e. Action-Value Function.
# 4x4 grid can be represented by one-hot vector with 16 integers.
inputs = tf.placeholder(shape=[1, 16], dtype=tf.float32)
net = InputLayer(inputs, name='observation')
net = DenseLayer(net, n_units=4, act=tf.identity,
    W_init=tf.random_uniform_initializer(0, 0.01), b_init=None, name='q_a_s')
y = net.outputs             # action-value / rewards of 4 actions
predict = tf.argmax(y, 1)   # chose action greedily with reward. in Q-Learning, policy is greedy, so we use "max" to select the next action.

## Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tl.cost.mean_squared_error(nextQ, y, is_mean=False) # tf.reduce_sum(tf.square(nextQ - y))
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

## Set learning parameters
lambd = .99    # decay factor
e = 0.1        # e-Greedy Exploration, the larger the more random
num_episodes = 10000
with tf.Session() as sess:
    tl.layers.initialize_global_variables(sess)
    for i in range(num_episodes):
        ## Reset environment and get first new observation
        episode_time = time.time()
        s = env.reset() # observation is state, integer 0 ~ 15
        rAll = 0
        for j in range(99): # step index, maximum step is 99
            if render: env.render()
            ## Choose an action by greedily (with e chance of random action) from the Q-network
            a, allQ = sess.run([predict, y], feed_dict={inputs : [to_one_hot(s, 16)]})
            ## e-Greedy Exploration !!! sample random action
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            ## Get new state and reward from environment
            s1, r, d, _ = env.step(a[0])
            ## Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(y, feed_dict={inputs : [to_one_hot(s1, 16)]})
            ## Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)  # in Q-Learning, policy is greedy, so we use "max" to select the next action.
            targetQ = allQ
            targetQ[0, a[0]] = r + lambd * maxQ1
            ## Train network using target and predicted Q values
            # it is not real target Q value, it is just an estimation,
            # but check the Q-Learning update formula:
            #    Q'(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a))
            # minimizing |r + lambd * maxQ(s',a') - Q(s, a)|^2 equal to force
            #   Q'(s,a) ≈ Q(s,a)
            _ = sess.run(train_op, {inputs : [to_one_hot(s, 16)], nextQ : targetQ})
            rAll += r
            s = s1
            ## Reduce chance of random action if an episode is done.
            if d == True:
                e = 1./((i/50) + 10)    # reduce e, GLIE: Greey in the limit with infinite Exploration
                break

        ## Note that, the rewards here with random action
        running_reward = rAll if running_reward is None else running_reward * 0.99 + rAll * 0.01
        print("Episode [%d/%d] sum reward:%f running reward:%f took:%.5fs %s" %
            (i, num_episodes, rAll, running_reward, time.time()-episode_time, '' if rAll == 0 else ' !!!!!!!!'))
