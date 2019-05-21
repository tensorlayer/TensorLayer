# Reinforcement Learning Tutorial with Tensorlayer

This folder contains implementation of most popular reinforcement learning algorithms with Tensorlayer 2.0.

## Prerequisites:

* python 3.5
* tensorflow >= 2.0.0
* tensorlayer >= 2.0.1
* tensorflow-probability
* tf-nightly-2.0-preview

*** If you meet problem `AttributeError: module 'tensorflow' has no attribute 'contrib'` when running the code after install tensorflow-probability, try:

`pip install --upgrade tf-nightly-2.0-preview tfp-nightly`

## To Use:

`python ***.py` 

or `python ***.py --train` for training and `python ***.py --test` for testing.

## Table of Contents:

| Algorithms   | Observation Space | Action Space |
| ------------ | ----------------- | ------------ |
| Q-learning   | Discrete          | Discrete     |
| DQN          | Discrete          | Discrete     |
| Actor-Critic | Continuous        | Discrete     |
| A3C          | Continuous        | Continuous   |
| SAC          | Continuous        | Continuous   |
| DDPG         | Continuous        | Continuous   |
| TD3          | Continuous        | Continuous   |
| HER          |                   |              |
| TRPO         |                   |              |
| PPO          |                   |              |
|              |                   |              |
|              |                   |              |
|              |                   |              |
|              |                   |              |



* Q-learning

  Code: `./tutorial_frozenlake_q_table.py`

  

* Deep Q-Network (DQN)

  Code: `./tutorial_frozenlake_dqn.py`

  

* Double DQN / Dueling DQN / Noisy DQN

  To do.

  

* Distributed DQN

  To do.

  

* Actor-Critic (AC)

  Code:`./tutorial_cartpole_ac.py`

  

* Asynchronous Advantage Actor-Critic (A3C)

  Code: `./tutorial_bipedalwalker_a3c_continuous_action.py`

  

* Soft Actor-Critic (SAC)

  Code: `./tutorial_sac.py`

  Paper: [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)

  

* Deep Deterministic Policy Gradient (DDPG)

  To do.

  

* Twin Delayed DDPG (TD3)

  Code: `./tutorial_td3.py`

  Paper: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)

  

* Hindsight Experience Replay (HER)

  To do.

  

* Trust Region Policy Optimization (TRPO)

  To do.

  

* Proximal Policy Optimization (PPO)

  To do.

  

* etc

## Environment:

[Openai Gym](https://gym.openai.com/)

Our env wrapper: `./tutorial_wrappers.py` 



### More examples can be found in [example List](https://tensorlayer.readthedocs.io/en/stable/user/examples.html)
