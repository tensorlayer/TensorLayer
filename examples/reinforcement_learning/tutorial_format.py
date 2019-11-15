# the format of turorial algorithm #
# please heavily annotate the code #
'''
Algorithm Name
------------------------
Briefly describe the algorithms, add some details.

Reference
---------
original paper: e.g. https://arxiv.org/pdf/1802.09477.pdf
website: ...


Environment
-----------
e.g. Openai Gym Pendulum-v0, continuous action space

Prerequisites
---------------
tensorflow >=2.0.0a0
tensorlayer >=2.0.0
...

To run
-------
python tutorial_***.py --train/test

'''

import argparse
import time

import numpy as np
import tensorflow as tf

# import 'other package name'

np.random.seed(2)
tf.random.set_seed(2)  # reproducible

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################
A = a  # description of hyper parameter
B = b  # description of hyper parameter

###############################  Algorithm Name  ####################################


class C():  # algorithm-specific classes
    ''' description of class '''

    def C1():
        ''' description of function'''


def D():  # some common functions, could be extracted into utils afterwards
    ''' description of function '''


if __name__ == '__main__':
    '''initialization of env, buffer, networks in algorithms'''
    env = 'env model'
    buffer = 'buffer model'
    network1 = 'network model1'
    network2 = 'network model2'

    # training loop
    if args.train:
        t0 = time.time()
        while NOT_FINISHED:  # loop of episodes
            while NOT_DONE:  # loop of steps in episode
                ''' step '''
                ''' train '''

            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'\
            .format(episode, all_episodes, episode_reward, time.time()-t0 ))
        ''' plot , following the format of ./baselines/utils/plot()'''
        plot(rewards, Algorithm_name='SAC', Env_name='Pendulum-v0')
        ''' save weights, implemented in defined classes above, following the format of ./baselines/utils/save_model()  '''
        model.save_weights()

    # testing loop
    if args.test:
        t0 = time.time()
        ''' save weights, implemented in defined classes above, following the format of ./baselines/utils/load_model()  '''
        model.load_weights()

        while NOT_FINISHED:  # loop of episodes
            while NOT_DONE:  # loop of steps in episode
                ''' step '''

            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'\
            .format(episode, all_episodes, episode_reward, time.time()-t0 ) )
