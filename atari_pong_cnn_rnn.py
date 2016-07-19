
"""
To understand Reinforcement Learning, we let computer to learn how to play
Pong game from the original screen inputs. Before we start, we highly recommend
you to go through a famous blog called “Deep Reinforcement Learning: Pong from
Pixels” which is a minimalistic implementation of deep reinforcement learning by
using python-numpy and OpenAI gym environment.

The code here is the reimplementation of Karpathy's Blog by using TensorLayer.

Compare with Karpathy's code, we store observation for one batch, he store
observation for one episode only, they store gradients instead. (so we will use
more memory if the observation is very large.)

Link
-----
http://karpathy.github.io/2016/05/31/rl/

"""

import gym
import tensorflow as tf
import numpy as np
import tensorlayer as tl
import time



# hyperparameters
image_size = 80
D = image_size * image_size
H = 200
batch_size = 1
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
render = False
resume = False   # load existing policy network
model_file_name = "model_pong_cnn_rnn"
np.set_printoptions(threshold=np.nan)


def prepro(I, is_reshape=True):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]
    I = I[::2,::2,0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    if is_reshape:
        return I.astype(np.float).ravel()
    else:
        I = I[np.newaxis, :, :, np.newaxis]
        return I.astype(np.float)



from six.moves import xrange
def iterate_reward_seq_minibatches(inputs, targets, rewards, batch_size, seq_length, stride):
    """
    Generate a generator that return a batch of sequence inputs and targets.
    Many to Many
    """
    assert len(inputs) == len(targets) == len(rewards)
    n_loads = (batch_size * stride) + (seq_length - stride)
    for start_idx in range(0, len(inputs) - n_loads + 1, (batch_size * stride)):
        seq_inputs = np.zeros((batch_size, seq_length) + inputs.shape[1:],
                              dtype=inputs.dtype)
        seq_targets = np.zeros((batch_size, seq_length) + targets.shape[1:],
                               dtype=targets.dtype)
        seq_rewards = np.zeros((batch_size, seq_length) + targets.shape[1:],
                               dtype=targets.dtype)
        for b_idx in xrange(batch_size):
            start_seq_idx = start_idx + (b_idx * stride)
            end_seq_idx = start_seq_idx + seq_length
            seq_inputs[b_idx] = inputs[start_seq_idx:end_seq_idx]
            seq_targets[b_idx] = targets[start_seq_idx:end_seq_idx]
            seq_rewards[b_idx] = rewards[start_seq_idx:end_seq_idx]
        flatten_inputs = seq_inputs.reshape((-1,) + inputs.shape[1:])
        flatten_targets = seq_targets.reshape((-1,) + targets.shape[1:])
        flatten_rewards = seq_rewards.reshape((-1,) + rewards.shape[1:])
        yield flatten_inputs, flatten_targets, flatten_rewards


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
running_reward = None
reward_sum = 0
episode_number = 0

## RNN
num_steps = 5       # sequence length
return_last = True  #

xs, ys, rs = [], [], []

## CNN + RNN
states_batch_pl = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 1])
network = tl.layers.InputLayer(states_batch_pl, name='input_layer')
network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name ='cnn_layer1')
network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool_layer1',)
network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [5, 5, 32, 10], # 10 features for each 5x5 patch
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name ='cnn_layer2')
network = tl.layers.PoolLayer(network,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME',
                    pool = tf.nn.max_pool,
                    name ='pool_layer2',)
network = tl.layers.FlattenLayer(network, name='flatten_layer')
network = tl.layers.RNNLayer(network,
                    cell_fn=tf.nn.rnn_cell.LSTMCell,
                    cell_init_args={},
                    n_hidden=200,
                    n_steps=num_steps,
                    return_last=True,
                    is_reshape=True,
                    name='rnn_layer')
network = tl.layers.DenseLayer(network, n_units=3,
                    act = tl.activation.identity, name='output_layer')

probs = network.outputs
sampling_prob = tf.nn.softmax(probs)

actions_batch_pl = tf.placeholder(tf.int32, shape=[None])
discount_rewards_batch_pl = tf.placeholder(tf.float32, shape=[None])
loss = tl.rein.cross_entropy_reward_loss(probs, actions_batch_pl, discount_rewards_batch_pl)
train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)#, var_list=[])


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    if resume:
        load_params = tl.files.load_npz(name=model_file_name+'.npz')
        tl.files.assign_params(sess, load_params, network)
    network.print_params()
    network.print_layers()

    start_time = time.time()
    game_number = 0
    frame_idx = 0
    x_agame = np.empty(shape=[0, image_size, image_size, 1])     # num_steps frames
    while True:
        if render: env.render()

        x = prepro(observation, False)
        frame_idx += 1

        if frame_idx > num_steps:
            prob = sess.run(
                sampling_prob,
                feed_dict={states_batch_pl: x_agame[-num_steps:, :]}
            )
            x_agame = np.vstack((x_agame, x))
            # action. 1: STOP  2: UP  3: DOWN
            action = np.random.choice([1,2,3], p=prob.flatten())
        else:
            x_agame = np.vstack((x_agame, x))
            action = 1

        observation, reward, done, _ = env.step(action)
        reward_sum += reward
        xs.append(x)            # all observations in a episode
        ys.append(action - 1)   # all fake labels in a episode (action begins from 1, so minus 1)
        rs.append(reward)       # all rewards in a episode
        if done:
            frame_idx = 0
            x_agame = np.empty(shape=[0, image_size, image_size, 1])
            episode_number += 1
            game_number = 0

            if episode_number % batch_size == 0:
                print('batch over...... updating parameters......')
                epx = np.vstack(xs)
                epy = np.asarray(ys)
                epr = np.asarray(rs)
                disR = tl.rein.discount_episode_rewards(epr, gamma)
                disR -= np.mean(disR)
                disR /= np.std(disR)

                xs, ys, rs = [], [], []
                x_seq, y_seq, d_seq = [], [], []
                for batch in iterate_reward_seq_minibatches(inputs=epx, targets=epy, rewards=disR, batch_size=100, seq_length=num_steps, stride=1):
                    x, y, r = batch
                    if return_last:
                        tmp_y = y.reshape((-1, num_steps) + y.shape[1:])
                        tmp_r = r.reshape((-1, num_steps) + r.shape[1:])
                    y = tmp_y[:, -1]
                    r = tmp_r[:, -1]

                    x_seq.append(x)
                    y_seq.append(y)
                    d_seq.append(r)

                x_seq = np.vstack(x_seq)
                y_seq = np.hstack(y_seq)
                d_seq = np.hstack(d_seq)

                sess.run(
                    train_op,
                    feed_dict={
                        states_batch_pl: x_seq,
                        actions_batch_pl: y_seq,
                        discount_rewards_batch_pl: d_seq
                    }
                )

            if episode_number % (batch_size * 10) == 0:
                tl.files.save_npz(network.all_params, name=model_file_name+'.npz')

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            observation = env.reset() # reset env
            prev_x = None

        if reward != 0:
            print(('episode %d: game %d took %.5fs, reward: %f' % (episode_number, game_number, time.time()-start_time, reward)), ('' if reward == -1 else ' !!!!!!!!'))
            start_time = time.time()
            game_number += 1
