# Reinforcement Learning Tutorial with Tensorlayer

This folder contains implementation of most popular reinforcement learning algorithms with Tensorlayer 2.0.

## Prerequisites:

* python 3.5
* tensorflow >= 2.0.0
* tensorlayer >= 2.0.1
* tensorflow-probability
* tf-nightly-2.0-preview

*** If you meet the error`AttributeError: module 'tensorflow' has no attribute 'contrib'` when running the code after installing tensorflow-probability, try:

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
| PG           | Continuous        | Discrete     |
| TRPO         | Continuous        | Continuous   |
| PPO          | Continuous        | Continuous   |
|              |                   |              |
|              |                   |              |
|              |                   |              |
|              |                   |              |



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

  
* Hindsight Experience Replay (HER)

  To do.

  
* Trust Region Policy Optimization (TRPO)

  Code: `./tutorial_TRPO.py`
  
  Paper: [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)

  
* Proximal Policy Optimization (PPO)

  Code: `./tutorial_PPO.py`
  
  Paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)

* Distributed Proximal Policy Optimization (PPO)

  Code: `./tutorial_DPPO.py`
  
  Paper: [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf)

  
* etc

## Environment:

[Openai Gym](https://gym.openai.com/)

Our env wrapper: `./tutorial_wrappers.py` 



### More examples can be found in [example List](https://tensorlayer.readthedocs.io/en/stable/user/examples.html)
