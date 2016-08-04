
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
import tunelayer as tl
import time



# hyperparameters
image_size = 80
D = image_size * image_size
H = 200
batch_size_episode = 10
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
        return I.astype(np.float32).ravel()
    else:
        I = I[np.newaxis, :, :, np.newaxis]
        return I.astype(np.float32)

def reward_ptb_iterator(raw_data, raw_data2, raw_data3, batch_size, num_steps):
    # raw_data = np.array(raw_data, dtype=np.int32)
    assert len(raw_data) == len(raw_data2) == len(raw_data3)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len, raw_data.shape[1], raw_data.shape[2], raw_data.shape[3]])#, dtype=np.int32)
    data2 = np.zeros([batch_size, batch_len])
    data3 = np.zeros([batch_size, batch_len])
    # print(data.shape)
    # print(data2.shape)
    # exit()
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
        data2[i] = raw_data2[batch_len * i:batch_len * (i + 1)]
        data3[i] = raw_data3[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        x2 = data2[:, i*num_steps:(i+1)*num_steps]
        x3 = data3[:, i*num_steps:(i+1)*num_steps]
        # y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, x2, x3)

# x = y = z = [i for i in range(20)]
# for batch in reward_ptb_iterator(x, y, z, batch_size=2, num_steps=3):
#     a, b, c = batch
#     print(a)
#     print(b)
#     print(c)
#     print()
# exit()

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
running_reward = None
reward_sum = 0
episode_number = 0

## RNN
num_steps = 5       # sequence length
batch_size_rnn = 10            # number of concurrent processes
batch_size = batch_size_rnn * num_steps
                               # different with PTB tutorial, the batch_size
                               # here is not RNN batch_size, it is the number
                               # of observations, so we need to multiply
                               # num_steps.

xs, ys, rs = [], [], []

"""We considered this problem as Synced sequence input and output.
Many to one (Sequence input and single output), predict a single output by
given a num_steps of observations, it initializes the states at the begining of
the observation i.e. it only look at the previous num_steps of observations,
the advantage of LSTM is wasted.
Human do not make decision only base on previous few observations, the
strategy information can be extracted from all previous observations.
So Synced sequence input and output can better to model this problem.
"""

def inference(x, num_steps, reuse=None):
    """If reuse is True, share the same parameters.
    """
    print('\n\nnum_steps: %d' % num_steps)
    with tf.variable_scope("model", reuse=reuse):
        # if reuse:
        tl.layers.set_name_reuse(True)
        network = tl.layers.InputLayer(x, name='input_layer')
        network = tl.layers.Conv2dLayer(network,
                            act = tf.nn.relu,
                            shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name ='cnn_layer1')
        network = tl.layers.PoolLayer(network,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            pool = tf.nn.max_pool,
                            name ='pool_layer1')
        network = tl.layers.Conv2dLayer(network,
                            act = tf.nn.relu,
                            shape = [5, 5, 32, 10], # 10 features for each 5x5 patch
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name ='cnn_layer2')
        network = tl.layers.PoolLayer(network,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            pool = tf.nn.max_pool,
                            name ='pool_layer2')
        network = tl.layers.FlattenLayer(network, name='flatten_layer')
        # Akara double check
        network = tl.layers.ReshapeLayer(network, shape=[-1, num_steps, int(network.outputs._shape[-1])])
        # network = tl.layers.ReshapeLayer(network, shape=[batch_size, num_steps, int(network.outputs._shape[-1])])
        rnn1 = tl.layers.RNNLayer(network,
                            cell_fn=tf.nn.rnn_cell.LSTMCell,
                            cell_init_args={},
                            n_hidden=200,
                            initializer=tf.random_uniform_initializer(-0.1, 0.1),
                            n_steps=num_steps,
                            return_last=False,
                            # is_reshape=True,
                            return_seq_2d=True,
                            name='rnn_layer')
        # print(rnn1.outputs)
        # exit()
        network = tl.layers.DenseLayer(rnn1, n_units=3,
                            act = tl.activation.identity, name='output_layer')
        probs = network.outputs
        sampling_prob = tf.nn.softmax(probs)
    return network, rnn1, probs, sampling_prob

def loss_function(probs, actions, rewards, batch_size):
    # loss = tl.rein.cross_entropy_reward_loss(probs, actions, rewards)
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [probs],
        [actions],
        [rewards])
    loss = tf.reduce_sum(loss) / batch_size
    return loss


states_batch_pl = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 1])
states_batch_pl_eval = tf.placeholder(tf.float32, shape=[1, image_size, image_size, 1])

# Inference for Training
network, rnn1, probs, sampling_prob = \
                                inference(states_batch_pl, num_steps)
# Inference for Predicting
network_eval, rnn1_eval, probs_eval, sampling_prob_eval = \
                        inference(states_batch_pl_eval, 1, reuse=True)

network.print_layers()
tl.layers.print_all_variables()

# exit()

# Loss and Optimizer for Training
actions_batch_pl = tf.placeholder(tf.int32, shape=[None])
discount_rewards_batch_pl = tf.placeholder(tf.float32, shape=[None])
loss = loss_function(network.outputs, actions_batch_pl, discount_rewards_batch_pl, batch_size_rnn * num_steps)

train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)#, var_list=[])

# exit()

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
    # For prediction (testing), initialize the RNN state to zero in the begining
    state = tl.layers.initialize_rnn_state(rnn1_eval.initial_state)
    while True:
        if render: env.render()

        x = prepro(observation, False)

        prob, state = sess.run(
            [sampling_prob_eval, rnn1_eval.final_state],
            feed_dict={
                states_batch_pl_eval : np.asarray(x),
                rnn1_eval.initial_state : state,
                    }
                )
        # action. 1: STOP  2: UP  3: DOWN
        action = np.random.choice([1,2,3], p=prob.flatten())

        observation, reward, done, _ = env.step(action)
        reward_sum += reward
        xs.append(x)            # all observations in a episode
        ys.append(action - 1)   # all fake labels in a episode (action begins from 1, so minus 1)
        rs.append(reward)       # all rewards in a episode

        if reward != 0:
            # reinitialize the RNN state, at the begining of each game.
            # rnn1_eval_state = rnn1_eval.cell.zero_state(1, dtype=tf.float32).eval()
            state = tl.layers.initialize_rnn_state(rnn1_eval.initial_state)

        if done:
            frame_idx = 0
            episode_number += 1
            game_number = 0

            if episode_number % batch_size_episode == 0:
                print('batch over...... updating parameters......')
                epx = np.vstack(xs)
                epy = np.asarray(ys)
                epr = np.asarray(rs)
                disR = tl.rein.discount_episode_rewards(epr, gamma)
                disR -= np.mean(disR)
                disR /= np.std(disR)
                # print(epx.shape)
                # exit()

                xs, ys, rs = [], [], []
                # For training,
                # initize the state at the begining
                state = tl.layers.initialize_rnn_state(rnn1.initial_state)
                for (x_seq, y_seq, d_seq) in reward_ptb_iterator(epx, epy, disR, batch_size=batch_size_rnn, num_steps=num_steps):
                    x_seq = np.reshape(x_seq, [-1, image_size, image_size, 1])
                    y_seq = np.reshape(y_seq, [-1])
                    d_seq = np.reshape(d_seq, [-1])
                    state, _ = sess.run(
                            [rnn1.final_state, train_op],
                            feed_dict={
                                states_batch_pl: x_seq,
                                actions_batch_pl: y_seq,
                                discount_rewards_batch_pl: d_seq,
                                # if you know batch_size, you can initize the states of each training examples as follow.
                                # rnn1.initial_state: rnn1.initial_state.eval()
                                # if you don't know the batch_size, you can initize them as follow
                                # Note: in RL, we don't know the batch_size.
                                rnn1.initial_state: state
                                }
                            )

                # initize state for predicting
                state = tl.layers.initialize_rnn_state(rnn1_eval.initial_state)

            if episode_number % (batch_size_episode * 10) == 0:
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
