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

tensorflow 2.0.0a0
tensorflow-probability 0.6.0
tensorlayer 2.0.0

&&
pip install box2d box2d-kengz --user

"""

import multiprocessing
import threading

import numpy as np

import gym
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import DenseLayer, InputLayer

tfd = tfp.distributions


# tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

# np.random.seed(2)
# tf.random.set_seed(2)  # reproducible

GAME = 'BipedalWalker-v2'  # BipedalWalkerHardcore-v2   BipedalWalker-v2  LunarLanderContinuous-v2
OUTPUT_GRAPH = False
LOG_DIR = './log'
# N_WORKERS = multiprocessing.cpu_count()
N_WORKERS = 2
MAX_GLOBAL_EP = 8000  # 8000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.99
ENTROPY_BETA = 0.005
LR_A = 0.00005  # learning rate for actor
LR_C = 0.0001  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0  # will increase during training, stop training when it >= MAX_GLOBAL_EP

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
# N_A = env.action_space.n

# A_BOUND = [env.action_space.low, env.action_space.high]
A_BOUND = [env.action_space.low, env.action_space.high]
# A_BOUND[0] = A_BOUND[0].reshape(1, N_A)
# A_BOUND[1] = A_BOUND[1].reshape(1, N_A)
# print(A_BOUND)

# print(env.unwrapped.hull.position[0])
# exit()


class ACNet(object):

    def __init__(self, scope, globalAC=None):  # no need for scope
        self.scope = scope
        self.save_path = './model'
        # if scope == GLOBAL_NET_SCOPE:
        #     ## global network only do inference
        #     with tf.variable_scope(scope):
        #         self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
        #         self._build_net()


        #         normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)  # for continuous action space

        #         with tf.name_scope('choose_a'):  # use local params to choose action
        #             self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)

        # else:
        #     ## worker network calculate gradient locally, update on global network
        #     # with tf.variable_scope(scope):
        #     #     self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
        #     #     self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
        #     #     self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')



        #         self._build_net()

            #     td = tf.subtract(self.v_target, self.v, name='TD_error')
            #     with tf.name_scope('c_loss'):
            #         self.c_loss = tf.reduce_mean(tf.square(td))

            #     with tf.name_scope('wrap_a_out'):
            #         self.test = self.sigma[0]
            #         self.mu, self.sigma = self.mu * A_BOUND[1], self.sigma + 1e-5

            #     normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)  # for continuous action space

            #     with tf.name_scope('a_loss'):
            #         log_prob = normal_dist.log_prob(self.a_his)
            #         exp_v = log_prob * td
            #         entropy = normal_dist.entropy()  # encourage exploration
            #         self.exp_v = ENTROPY_BETA * entropy + exp_v
            #         self.a_loss = tf.reduce_mean(-self.exp_v)

            #     with tf.name_scope('choose_a'):  # use local params to choose action
            #         self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)

            #     with tf.name_scope('local_grad'):
            #         self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
            #         self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)
            #         self.a_grads = tf.gradients(self.a_loss, self.a_params)
            #         self.c_grads = tf.gradients(self.c_loss, self.c_params)

            # with tf.name_scope('sync'):
            #     with tf.name_scope('pull'):
            #         self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
            #         self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
            #     with tf.name_scope('push'):
            #         self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
            #         self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    # def _build_net(self):
        # w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.random_uniform_initializer(0, 0.01)
        w_init = tf.keras.initializers.glorot_normal(seed=None)
        # with tf.variable_scope('actor'):  # Policy network
        #     nn = InputLayer(self.s, name='in')
        #     nn = DenseLayer(nn, n_units=500, act=tf.nn.relu6, W_init=w_init, name='la')
        #     nn = DenseLayer(nn, n_units=300, act=tf.nn.relu6, W_init=w_init, name='la2')
        #     mu = DenseLayer(nn, n_units=N_A, act=tf.nn.tanh, W_init=w_init, name='mu')
        #     sigma = DenseLayer(nn, n_units=N_A, act=tf.nn.softplus, W_init=w_init, name='sigma')
        #     self.mu = mu.outputs
        #     self.sigma = sigma.outputs
        def get_actor(input_shape):
            with tf.name_scope(self.scope):
                ni = tl.layers.Input(input_shape, name='in')
                nn = tl.layers.Dense(n_units=50, act=tf.nn.relu6, W_init=w_init, name='la')(ni)
                nn = tl.layers.Dense(n_units=30, act=tf.nn.relu6, W_init=w_init, name='la2')(nn)
                mu = tl.layers.Dense(n_units=N_A, act=tf.nn.tanh, W_init=w_init, name='mu')(nn)
                sigma = tl.layers.Dense(n_units=N_A, act=tf.nn.softplus, W_init=w_init, name='sigma')(nn)
            return tl.models.Model(inputs=ni, outputs=[mu, sigma], name=scope+'/Actor')
        self.actor = get_actor( [None, N_S])
        self.actor.train() # set training mode, also for workers?

        def get_critic(input_shape):
            with tf.name_scope(self.scope):
                ni = tl.layers.Input(input_shape, name='in')
                nn = tl.layers.Dense(n_units=50, act=tf.nn.relu6, W_init=w_init, name='lc')(ni)
                nn = tl.layers.Dense(n_units=30, act=tf.nn.relu6, W_init=w_init, name='lc2')(nn)
                v = tl.layers.Dense(n_units=1, W_init=w_init, name='v')(nn)
            return tl.models.Model(inputs=ni, outputs=v, name=scope+'/Critic')
        self.critic = get_critic( [None, N_S])
        self.critic.train()

        # self.a_params = tl.layers.get_variables_with_name(scope + '/Actor', True, False)
        # self.c_params = tl.layers.get_variables_with_name(scope + '/Critic', True, False)

        # with tf.variable_scope('critic'):  # we use Value-function here, but not Q-function.
        #     nn = InputLayer(self.s, name='in')
        #     nn = DenseLayer(nn, n_units=500, act=tf.nn.relu6, W_init=w_init, name='lc')
        #     nn = DenseLayer(nn, n_units=200, act=tf.nn.relu6, W_init=w_init, name='lc2')
        #     v = DenseLayer(nn, n_units=1, W_init=w_init, name='v')
        #     self.v = v.outputs

    # def update_global(self, feed_dict):  # run by a local
    def update_global(self, buffer_s, buffer_a, buffer_v_target, globalAC):
        # _, _, t = sess.run(
        #     [self.update_a_op, self.update_c_op, self.test], feed_dict
        # )  # local grads applies to global net
        with tf.GradientTape() as tape:
            self.v = self.critic(buffer_s)
            self.v_target = buffer_v_target  # tensor float?
            td = tf.subtract(self.v_target, self.v, name='TD_error')
            # with tf.name_scope('c_loss'):
            self.c_loss = tf.reduce_mean(tf.square(td))
        self.c_grads = tape.gradient(self.c_loss, self.critic.trainable_weights)
        OPT_C.apply_gradients(zip(self.c_grads, globalAC.critic.trainable_weights))
        del tape # Drop the reference to the tape


        with tf.GradientTape() as tape:
            self.mu, self.sigma = self.actor(buffer_s)
            # print('mu: ', self.mu)
            # print('sigma: ', self.sigma)
            # with tf.name_scope('wrap_a_out'):
            self.test = self.sigma[0]
            self.mu, self.sigma = self.mu * A_BOUND[1], self.sigma + 1e-5


            # normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)  # for continuous action space
            normal_dist = tfd.Normal(self.mu, self.sigma)
            # with tf.name_scope('a_loss'):
            self.a_his = buffer_a # tensor float?
            log_prob = normal_dist.log_prob(self.a_his)
            exp_v = log_prob * td
            entropy = normal_dist.entropy()  # encourage exploration
            self.exp_v = ENTROPY_BETA * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)
        # with tf.name_scope('local_grad'):
            # self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
            # self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)
        #     self.a_grads = tf.gradients(self.a_loss, self.a_params)
        #     self.c_grads = tf.gradients(self.c_loss, self.c_params)
        self.a_grads = tape.gradient(self.a_loss, self.actor.trainable_weights)
        OPT_A.apply_gradients(zip(self.a_grads, globalAC.actor.trainable_weights))
                

        # return t
        return self.test.numpy()

    def pull_global(self, globalAC):  # run by a local
        # sess.run([self.pull_a_params_op, self.pull_c_params_op])
        # with tf.name_scope('sync'):
        #     with tf.name_scope('pull'):
        # self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.actor.trainable_weights, globalAC.a_params)]
        # self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.critic.trainable_weights, globalAC.c_params)]
        # print(self.scope, 'before: ', self.actor.trainable_weights[0][5][2])
        # print(self.scope, 'global: ', globalAC.actor.trainable_weights[0][5][2])
        for l_p, g_p in zip(self.actor.trainable_weights, globalAC.actor.trainable_weights):
            l_p.assign(g_p)
        for l_p, g_p in zip(self.critic.trainable_weights, globalAC.critic.trainable_weights):
            l_p.assign(g_p)
        # print(self.scope, 'after: ', self.actor.trainable_weights[0][5][2])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        self.mu, self.sigma = self.actor(s)
        # print('mu1: ', self.mu)
        # print('sigma1: ', self.sigma)
        with tf.name_scope('wrap_a_out'):
            # self.test = self.sigma[0]
            self.mu, self.sigma = self.mu * A_BOUND[1], self.sigma + 1e-5
        # normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        normal_dist = tfd.Normal(self.mu, self.sigma)
        self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)
        # return sess.run(self.A, {self.s: s})[0]
        return self.A.numpy()[0]

    def save_ckpt(self): 
        tl.files.save_npz(self.actor.trainable_weights, name='model_actor.npz')
        tl.files.save_npz(self.critic.trainable_weights, name='model_critic.npz')
        # tl.files.save_ckpt(
        #     sess=sess, mode_name='model.ckpt', var_list=self.a_params + self.c_params, save_dir=self.scope,
        #     printable=True
        # )

    def load_ckpt(self):
        # tl.files.load_hdf5_to_trainable_weights(self.save_path+'/actor', self.actor)
        # tl.files.load_hdf5_to_trainable_weights(self.save_path+'/critic', self.critic)
        tl.files.load_and_assign_npz(name='model_actor.npz', network=self.actor)
        tl.files.load_and_assign_npz(name='model_critic.npz', network=self.critic)

class Worker(object):

    def __init__(self, name, globalAC):
        self.env = gym.make(GAME)
        self.name = name
        self.AC = ACNet(name, globalAC)

    # def work(self):
    def work(self, globalAC):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                # visualize Worker_0 during training
                if self.name == 'Worker_0' and total_step % 30 == 0:
                    self.env.render()
                s = s.astype('float32') # double to float
                a = self.AC.choose_action(s) 
                s_, r, done, _info = self.env.step(a)
                s_ = s_.astype('float32') # double to float
                # print('s:',s)
                # print('a:', a)
                # print('r:',r)

                # set robot falls reward to -2 instead of -100
                if r == -100: r = -2

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net

                    if done:
                        v_s_ = 0  # terminal
                    else:
                        # v_s_ = sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                        v_s_ = self.AC.critic(s_[np.newaxis, :])[0,0] # reduce dim from 2 to 0

                    buffer_v_target = []

                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)

                    buffer_v_target.reverse()
                    
                    buffer_s, buffer_a, buffer_v_target = (
                        np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    )
                    # print(buffer_s, buffer_a, buffer_v_target)
                    # feed_dict = {self.AC.s: buffer_s, self.AC.a_his: buffer_a, self.AC.v_target: buffer_v_target}
                    # update gradients on global network
                    # self.AC.update_global(feed_dict)
                    self.AC.update_global(buffer_s, buffer_a, buffer_v_target, globalAC)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    # update local network from global network
                    self.AC.pull_global(globalAC)

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:  # moving average
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    print(
                        self.name,
                        "episode:",
                        GLOBAL_EP,
                        # "| pos: %i" % self.env.unwrapped.hull.position[0],  # number of move
                        '| reward: %.1f' % ep_r,
                        "| running_reward: %.1f" % GLOBAL_RUNNING_R[-1],
                        # '| sigma:', test, # debug
                        # 'WIN ' * 5 if self.env.unwrapped.hull.position[0] >= 88 else '',
                    )
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    # sess = tf.Session()
    # ============================= TRAINING ===============================
    with tf.device("/cpu:0"):
        
        OPT_A = tf.optimizers.RMSprop(LR_A, name='RMSPropA')
        OPT_C = tf.optimizers.RMSprop(LR_C, name='RMSPropC')

        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'Worker_%i' % i  # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    # sess.run(tf.global_variables_initializer())

    # start TF threading
    worker_threads = []
    for worker in workers:
        # t = threading.Thread(target=worker.work)
        job = lambda: worker.work(GLOBAL_AC)
        t = threading.Thread(target=job)  
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
    import matplotlib.pyplot as plt
    plt.plot(GLOBAL_RUNNING_R)
    plt.xlabel('episode')
    plt.ylabel('global running reward')
    plt.savefig('a3c.png')
    plt.show()

    GLOBAL_AC.save_ckpt()

    # ============================= EVALUATION =============================
    # env = gym.make(GAME)
    # GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
    # sess.run(tf.global_variables_initializer())
    GLOBAL_AC.load_ckpt()
    while True:
        s = env.reset()
        rall = 0
        while True:
            env.render()
            s = s.astype('float32') # double to float
            a = GLOBAL_AC.choose_action(s)
            s, r, d, _ = env.step(a)
            rall += r
            if d:
                print("reward", rall)
                break
