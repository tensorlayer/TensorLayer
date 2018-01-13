"""
Asynchronous Advantage Actor Critic (A3C) with Continuous Action Space.

Actor Critic History
----------------------
A3C > DDPG (for continuous action space) > AC

Advantage
----------
Train faster and more stable than AC.

Disadvantage
-------------
Have bias.

Reference
----------
MorvanZhou's tutorial: https://morvanzhou.github.io/tutorials/
MorvanZhou's code: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/experiments/Solve_BipedalWalker/A3C.py

Environment
-----------
BipedalWalker-v2 : https://gym.openai.com/envs/BipedalWalker-v2

Reward is given for moving forward, total 300+ points up to the far end.
If the robot falls, it gets -100. Applying motor torque costs a small amount of
points, more optimal agent will get better score. State consists of hull angle
speed, angular velocity, horizontal speed, vertical speed, position of joints
and joints angular speed, legs contact with ground, and 10 lidar rangefinder
measurements. There's no coordinates in the state vector.
"""

import multiprocessing
import os
import shutil
import threading

import gym
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

GAME = 'BipedalWalker-v2' # BipedalWalkerHardcore-v2
OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
# N_WORKERS = 4
MAX_GLOBAL_EP = 20000#8000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.999
ENTROPY_BETA = 0.005
LR_A = 0.00002    # learning rate for actor
LR_C = 0.0001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0    # will increase during training, stop training when it >= MAX_GLOBAL_EP

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]
# print(env.unwrapped.hull.position[0])
# exit()

class ACNet(object):
    def __init__(self, scope, globalAC=None):

        self.scope = scope
        if scope == GLOBAL_NET_SCOPE:
            ## global network only do inference
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net()
                self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
                self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)

                normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma) # for continuous action space

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)

        else:
            ## worker network calculate gradient locally, update on global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    self.test = self.sigma[0]
                    self.mu, self.sigma = self.mu * A_BOUND[1], self.sigma + 1e-5

                normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma) # for continuous action space

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)

                with tf.name_scope('local_grad'):
                    self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
                    self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        w_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('actor'):        # Policy network
            nn = InputLayer(self.s, name='in')
            nn = DenseLayer(nn, n_units=500, act=tf.nn.relu6, W_init=w_init, name='la')
            nn = DenseLayer(nn, n_units=300, act=tf.nn.relu6, W_init=w_init, name='la2')
            mu = DenseLayer(nn, n_units=N_A, act=tf.nn.tanh, W_init=w_init, name='mu')
            sigma = DenseLayer(nn, n_units=N_A, act=tf.nn.softplus, W_init=w_init, name='sigma')
            self.mu = mu.outputs
            self.sigma = sigma.outputs

        with tf.variable_scope('critic'):       # we use Value-function here, but not Q-function.
            nn = InputLayer(self.s, name='in')
            nn = DenseLayer(nn, n_units=500, act=tf.nn.relu6, W_init=w_init, name='lc')
            nn = DenseLayer(nn, n_units=200, act=tf.nn.relu6, W_init=w_init, name='lc2')
            v = DenseLayer(nn, n_units=1, W_init=w_init, name='v')
            self.v = v.outputs

    def update_global(self, feed_dict):  # run by a local
        _, _, t = sess.run([self.update_a_op, self.update_c_op, self.test], feed_dict)  # local grads applies to global net
        return t

    def pull_global(self):  # run by a local
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return sess.run(self.A, {self.s: s})[0]

    def save_ckpt(self):
        tl.files.exists_or_mkdir(self.scope)
        tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', var_list=self.a_params+self.c_params, save_dir=self.scope, printable=True)

    def load_ckpt(self):
        tl.files.load_ckpt(sess=sess, var_list=self.a_params+self.c_params, save_dir=self.scope, printable=True)
        # tl.files.load_ckpt(sess=sess, mode_name='model.ckpt', var_list=self.a_params+self.c_params, save_dir=self.scope, is_latest=False, printable=True)

class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME)
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                ## visualize Worker_0 during training
                if self.name == 'Worker_0' and total_step % 30 == 0:
                    self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)

                ## set robot falls reward to -2 instead of -100
                if r == -100: r = -2

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    ## update gradients on global network
                    test = self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    ## update local network from global network
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    print(
                        self.name,
                        "episode:", GLOBAL_EP,
                        "| pos: %i" % self.env.unwrapped.hull.position[0],  # number of move
                        '| reward: %.1f' % ep_r,
                        "| running_reward: %.1f" % GLOBAL_RUNNING_R[-1],
                        # '| sigma:', test, # debug
                        'WIN '*5 if self.env.unwrapped.hull.position[0] >= 88 else '',
                    )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    sess = tf.Session()

    ###============================= TRAINING ===============================###
    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'Worker_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    tl.layers.initialize_global_variables(sess)

    ## start TF threading
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    GLOBAL_AC.save_ckpt()

    ###============================= EVALUATION =============================###
    # env = gym.make(GAME)
    # GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
    # tl.layers.initialize_global_variables(sess)
    # GLOBAL_AC.load_ckpt()
    # while True:
    #     s = env.reset()
    #     rall = 0
    #     while True:
    #         env.render()
    #         a = GLOBAL_AC.choose_action(s)
    #         s, r, d, _ = env.step(a)
    #         rall += r
    #         if d:
    #             print("reward", rall)
    #             break
