# Reinforcement Learning Tutorial with Tensorlayer

This repository contains implementation of most popular reinforcement learning algorithms with Tensorlayer 2.0, supporting [Tensorflow 2.0](https://www.tensorflow.org/alpha/guide/effective_tf2). We aim to make the reinforcement learning tutorial for each algorithm simple and straight-forward to use, as this would not only benefits new learners of reinforcement learning, but also provide convenience for senior researchers to testify their new ideas quickly.

## Prerequisites:

* python 3.5
* tensorflow >= 2.0.0
* tensorlayer >= 2.0.1
* tensorflow-probability
* tf-nightly-2.0-preview

*** If you meet the error`AttributeError: module 'tensorflow' has no attribute 'contrib'` when running the code after installing tensorflow-probability, try:

`pip install --upgrade tf-nightly-2.0-preview tfp-nightly`

## Status: Beta

We are currently open to any suggestions or pull requests from you to make the reinforcement learning tutorial with TensorLayer2.0 a better code repository for both new learners and senior researchers. Some of the algorithms mentioned in the this markdown may be not yet available, since we are still trying to implement more RL algorithms and optimize their performances. However, those algorithms listed above will come out in a few weeks, and the repository will keep updating more advanced RL algorithms in the future.

## To Use:

For each tutorial, open a terminal and run:

`python ***.py` 

or `python ***.py --train` for training and `python ***.py --test` for testing.

## Table of Contents:

| Algorithms      | Observation Space | Action Space | Tutorial Env   |
| --------------- | ----------------- | ------------ | -------------- |
| Q-learning      | Discrete          | Discrete     | FrozenLake     |
| DQN             | Discrete          | Discrete     | FrozenLake     |
| Variants of DQN | Continuous        | Discrete     | Pong, CartPole |
| Actor-Critic    | Continuous        | Discrete     | CartPole       |
| A3C             | Continuous        | Continuous   | BipedalWalker  |
| SAC             | Continuous        | Continuous   | Pendulum       |
| PG              | Continuous        | Discrete     | CartPole       |
| DDPG            | Continuous        | Continuous   | Pendulum       |
| TD3             | Continuous        | Continuous   | Pendulum       |
| C51             | Continuous        | Discrete     | CartPole       |

## Examples of RL Algorithms:

* Q-learning

  Code: `./tutorial_frozenlake_q_table.py`

  

* Deep Q-Network (DQN)

  Code: `./tutorial_frozenlake_dqn.py`

  

* Double DQN / Dueling DQN / Noisy DQN

  Code: `./tutorial_double_dueling_noisy_dqn.py`

  Experiment Environments: Pong and Cartpole

  


* Prioritized replay

  Code: `./tutorial_prioritized_replay.py`

  Experiment Environments: Pong and Cartpole

  

* Distributed DQN

  Code: `./tutorial_c51.py`

  Experiment Environments: Pong and Cartpole

  


* Retrace(lambda) DQN

  Code: `./tutorial_retrace.py`

  Experiment Environments: Pong and Cartpole

  


* Actor-Critic (AC)

  Code:`./tutorial_cartpole_ac.py`

  

* Asynchronous Advantage Actor-Critic (A3C)

  Code: `./tutorial_bipedalwalker_a3c_continuous_action.py`

  

* Soft Actor-Critic (SAC)

  Code: `./tutorial_sac.py`

  Paper: [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)

  


* Policy Gradient (PG/REINFORCE) 

  Code: `./tutorial_PG.py`

  Paper: [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

  

* Deep Deterministic Policy Gradient (DDPG)

  Code: `./tutorial_DDPG.py`

  Paper: [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1509.02971.pdf)

  


* Twin Delayed DDPG (TD3)

  Code: `./tutorial_td3.py`

  Paper: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)

  

* Trust Region Policy Optimization (TRPO)

  Code: `./tutorial_TRPO.py`

  Paper: [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)

  

* Proximal Policy Optimization (PPO)

  Code: `./tutorial_PPO.py`

  Paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)

  

* Distributed Proximal Policy Optimization (PPO)

  Code: `./tutorial_DPPO.py`

  Paper: [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf)

  

* Hindsight Experience Replay (HER)

  To do.

* etc

## Environment:

We typically apply game environments in [Openai Gym](https://gym.openai.com/) for our tutorials. For other environment sources like [DeepMind Control Suite](https://github.com/deepmind/dm_control) and [Marathon-Envs in Unity](https://github.com/Unity-Technologies/marathon-envs), they all have wrappers to convert into format of Gym environments, see [here](https://github.com/martinseilair/dm_control2gym) and [here](https://github.com/Unity-Technologies/marathon-envs/tree/master/gym-unity).

Our env wrapper: `./tutorial_wrappers.py` 



### More examples can be found in [example List](https://tensorlayer.readthedocs.io/en/stable/user/examples.html)
