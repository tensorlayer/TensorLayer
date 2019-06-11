# Reinforcement Learning Tutorial with Tensorlayer

<br/>

<a href="https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc" target="\_blank">
	<div align="center">
		<img src="../../img/join_slack.png" width="40%"/>
	</div>
</a>

<br/>

This repository contains implementation of most popular reinforcement learning algorithms with Tensorlayer 2.0, supporting [Tensorflow 2.0](https://www.tensorflow.org/alpha/guide/effective_tf2). We aim to make the reinforcement learning tutorial for each algorithm simple and straight-forward to use, as this would not only benefits new learners of reinforcement learning, but also provide convenience for senior researchers to testify their new ideas quickly.

## Prerequisites:

* python 3.5
* tensorflow >= 2.0.0 or tensorflow-gpu >= 2.0.0a0
* tensorlayer >= 2.0.1
* tensorflow-probability
* tf-nightly-2.0-preview

*** If you meet the error`AttributeError: module 'tensorflow' has no attribute 'contrib'` when running the code after installing tensorflow-probability, try:

`pip install --upgrade tf-nightly-2.0-preview tfp-nightly`

## Status: Beta

We are currently open to any suggestions or pull requests from you to make the reinforcement learning tutorial with TensorLayer2.0 a better code repository for both new learners and senior researchers. Some of the algorithms mentioned in the this markdown may be not yet available, since we are still trying to implement more RL algorithms and optimize their performances. However, those algorithms listed above will come out in a few weeks, and the repository will keep updating more advanced RL algorithms in the future.

## To Use:

For each tutorial, open a terminal and run:

 `python ***.py --train` for training and `python ***.py --test` for testing.

The tutorial algorithms follow the same basic structure, as shown in file: [`./tutorial_format.py`](https://github.com/tensorlayer/tensorlayer/blob/reinforcement-learning/examples/reinforcement_learning/tutorial_format.py)

## Table of Contents:

| Algorithms      | Observation Space | Action Space | Tutorial Env   |
| --------------- | ----------------- | ------------ | -------------- |
| Q-learning      | Discrete          | Discrete     | FrozenLake     |
| C51             | Discrete          | Discrete     | Pong, CartPole |
| DQN             | Discrete          | Discrete     | FrozenLake     |
| Variants of DQN | Discrete          | Discrete     | Pong, CartPole |
| Retrace         | Discrete          | Discrete     | Pong, CartPole |
| PER             | Discrete          | Discrete     | Pong, CartPole |
| Actor-Critic    | Continuous        | Discrete     | CartPole       |
| A3C             | Continuous        | Continuous   | BipedalWalker  |
| DDPG            | Continuous        | Continuous   | Pendulum       |
| TD3             | Continuous        | Continuous   | Pendulum       |
| SAC             | Continuous        | Continuous   | Pendulum       |
| PG              | Continuous        | Discrete     | CartPole       |
| TRPO            | Continuous        | Continuous   | Pendulum       |
| PPO             | Continuous        | Continuous   | Pendulum       |
| DPPO            | Continuous        | Continuous   | Pendulum       |


## Examples of RL Algorithms:

* Q-learning

  Code: `./tutorial_Qlearning.py`

  Paper: [Technical  Note Q-Learning](http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)

  

* Deep Q-Network (DQN)

  Code: `./tutorial_DQN.py`

  Paper: [Human-level control through deep reinforcementlearning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

  [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

  

* Double DQN / Dueling DQN / Noisy DQN

  Code: `./tutorial_DQN_variants.py`

  Paper: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

  


* PER (Prioritized Experience Replay)

  Code: `./tutorial_prioritized_replay.py`

  Paper: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

  

* Distributed DQN

  Code: `./tutorial_C51.py`

  Paper: [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf)

  


* Retrace(lambda) DQN

  Code: `./tutorial_Retrace.py`

  Paper: [Safe and Efficient Off-Policy Reinforcement Learning](https://arxiv.org/abs/1606.02647)

  


* Actor-Critic (AC)

  Code:`./tutorial_AC.py`

  Paper: [Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)

  

* Asynchronous Advantage Actor-Critic (A3C)

  Code: `./tutorial_A3C.py`

  Paper: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)

  

* Soft Actor-Critic (SAC)

  Code: `./tutorial_SAC.py`

  Paper: [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)

  


* Policy Gradient (PG/REINFORCE) 

  Code: `./tutorial_PG.py`

  Paper: [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

  

* Deep Deterministic Policy Gradient (DDPG)

  Code: `./tutorial_DDPG.py`

  Paper: [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)

  


* Twin Delayed DDPG (TD3)

  Code: `./tutorial_TD3.py`

  Paper: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)

  

* Trust Region Policy Optimization (TRPO)

  Code: `./tutorial_TRPO.py`

  Paper: [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)

  

* Proximal Policy Optimization (PPO)

  Code: `./tutorial_PPO.py`

  Paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)

  

* Distributed Proximal Policy Optimization (DPPO)

  Code: `./tutorial_DPPO.py`

  Paper: [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf)

  

* Hindsight Experience Replay (HER)

  To do.

* etc

## Environment:

We typically apply game environments in [Openai Gym](https://gym.openai.com/) for our tutorials. For other environment sources like [DeepMind Control Suite](https://github.com/deepmind/dm_control) and [Marathon-Envs in Unity](https://github.com/Unity-Technologies/marathon-envs), they all have wrappers to convert into format of Gym environments, see [here](https://github.com/martinseilair/dm_control2gym) and [here](https://github.com/Unity-Technologies/marathon-envs/tree/master/gym-unity).

Our env wrapper: `./tutorial_wrappers.py` 

## Authors
- @xxxx XXXXX : AC, A3C
- @quantumiracle Zihan Ding: SAC, TD3.
- @Tokarev-TT-33 Tianyang Yu @initial-h Hongming Zhang : PG, DDPG, PPO, DPPO, TRPO
- @Officium Yanhua Huang: C51, Retrace, DQN_variants, prioritized_replay, wrappers.

### More examples can be found in the [example list](https://tensorlayer.readthedocs.io/en/stable/user/examples.html)
