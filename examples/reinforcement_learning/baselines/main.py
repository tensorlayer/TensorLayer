'''
main function, call different algorithms and different environments
to train with specified settings
'''

import argparse
import os

from common.utils import learn, parse_all_args

# """ Parse arguments """
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--env', type=str, required=True, help='environment ID')
parser.add_argument('--algorithm', type=str, required=True, help='Algorithm')
# parser.add_argument('--nenv', type=int, default=0, help='parrallel number')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--train_episodes', type=int, default=1e6)
# parser.add_argument('--reward_scale', type=float, default=1.0)
parser.add_argument('--mode', help='train or test', default='train')
parser.add_argument('--save_interval', type=int, default=100, help='save model every x episodes (0 = disabled)')
# parser.add_argument('--log_path', type=str, default='../logs',
#                     help='save model every x steps (0 = disabled)')
common_options, other_options = parse_all_args(parser)
""" Learn """
model = learn(
    # device=device,
    env_id=common_options.env,
    algorithm=common_options.algorithm,
    # nenv=common_options.nenv,
    seed=common_options.seed,
    train_episodes=int(common_options.train_episodes),
    # reward_scale=common_options.reward_scale,
    mode=common_options.mode,
    save_interval=common_options.save_interval,
    # log_path=common_options.log_path,
    **other_options
)
