# Reinforcement Learning Tutorial with Tensorlayer

<br/>
<a href="https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc" target="\_blank">
	<div align="center">
		<img src="../../img/join_slack.png" width="40%"/>
	</div>
	<div align="center"><caption>Slack Invitation Link</caption></div>
</a>
<br/>

<!--
<br/>
<a href="https://github.com/tensorlayer/tensorlayer-chinese/blob/master/docs/images/RL_Group_QR.jpeg" target="\_blank">
	<div align="center">
		<img src="https://github.com/tensorlayer/tensorlayer-chinese/blob/master/docs/images/RL_Group_QR.jpeg" width="20%"/>
	</div>
	<div align="center"><caption>WeChat QR Code</caption></div>
</a>
<br/>
-->

This repository contains the implementation of most popular reinforcement learning algorithms with Tensorlayer 2.0, supporting [Tensorflow 2.0](https://www.tensorflow.org/alpha/guide/effective_tf2). We aim to make the reinforcement learning tutorial for each algorithm simple and straight-forward to use, as this would not only benefit new learners of reinforcement learning but also provide convenience for senior researchers to testify their new ideas quickly. In addition to this project, we also released a [RL zoo](https://github.com/tensorlayer/RLzoo) for industrial users.

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
### value-based
| Algorithms      | Action Space | Tutorial Env   | Papers |
| --------------- | ------------ | -------------- | -------|
|**value-based**||||
| Q-learning      | Discrete     | FrozenLake     | [Technical note: Q-learning. Watkins et al. 1992](http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)|
| Deep Q-Network (DQN)| Discrete     | FrozenLake     | [Human-level control through deep reinforcement learning, Mnih et al. 2015.](https://www.nature.com/articles/nature14236/) |
| Prioritized Experience Replay | Discrete     | Pong, CartPole | [Schaul et al. Prioritized experience replay. Schaul et al. 2015.](https://arxiv.org/abs/1511.05952) |
|Dueling DQN|Discrete     | Pong, CartPole |[Dueling network architectures for deep reinforcement learning. Wang et al. 2015.](https://arxiv.org/abs/1511.06581)|
|Double DQN| Discrete     | Pong, CartPole |[Deep reinforcement learning with double q-learning. Van et al. 2016.](https://arxiv.org/abs/1509.06461)|
|Noisy DQN|Discrete     | Pong, CartPole |[Noisy networks for exploration. Fortunato et al. 2017.](https://arxiv.org/pdf/1706.10295.pdf)|
| Distributed DQN (C51)| Discrete     | Pong, CartPole | [A distributional perspective on reinforcement learning. Bellemare et al. 2017.](https://arxiv.org/pdf/1707.06887.pdf) |
|**policy-based**||||
|REINFORCE(PG) |Discrete/Continuous|CartPole | [Reinforcement learning: An introduction. Sutton et al. 2011.](https://www.cambridge.org/core/journals/robotica/article/robot-learning-edited-by-jonathan-h-connell-and-sridhar-mahadevan-kluwer-boston-19931997-xii240-pp-isbn-0792393651-hardback-21800-guilders-12000-8995/737FD21CA908246DF17779E9C20B6DF6)|
| Trust Region Policy Optimization (TRPO)| Discrete/Continuous | Pendulum | [Abbeel et al. Trust region policy optimization. Schulman et al.2015.](https://arxiv.org/pdf/1502.05477.pdf) |
| Proximal Policy Optimization (PPO) |Discrete/Continuous |Pendulum| [Proximal policy optimization algorithms. Schulman et al. 2017.](https://arxiv.org/abs/1707.06347) |
|Distributed Proximal Policy Optimization (DPPO)|Discrete/Continuous |Pendulum|[Emergence of locomotion behaviours in rich environments. Heess et al. 2017.](https://arxiv.org/abs/1707.02286)|
|**actor-critic**||||
|Actor-Critic (AC)|Discrete/Continuous|CartPole| [Actor-critic algorithms. Konda er al. 2000.](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)|
| Asynchronous Advantage Actor-Critic (A3C)| Discrete/Continuous | BipedalWalker| [Asynchronous methods for deep reinforcement learning. Mnih et al. 2016.](https://arxiv.org/pdf/1602.01783.pdf) |
| DDPG|Discrete/Continuous |Pendulum| [Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016](https://arxiv.org/pdf/1509.02971.pdf) |
|TD3|Discrete/Continuous |Pendulum|[Addressing function approximation error in actor-critic methods. Fujimoto et al. 2018.](https://arxiv.org/pdf/1802.09477.pdf)|
|Soft Actor-Critic (SAC)|Discrete/Continuous |Pendulum|[Soft actor-critic algorithms and applications. Haarnoja et al. 2018.](https://arxiv.org/abs/1812.05905)|

## Examples of RL Algorithms:

* **Q-learning**

  Code: `./tutorial_Qlearning.py`

  <u>Paper</u>: [Technical  Note Q-Learning](http://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)

  <u>Description</u>:

  ```
  Q-learning is a non-deep-learning method with TD Learning, Off-Policy, e-Greedy Exploration.

  Central formula:
  Q(S, A) <- Q(S, A) + alpha * (R + lambda * Q(newS, newA) - Q(S, A))

  See David Silver RL Tutorial Lecture 5 - Q-Learning for more details.
  ```

  ​    

* **Deep Q-Network (DQN)**

  <u>Code:</u> `./tutorial_DQN.py`

  <u>Paper</u>: [Human-level control through deep reinforcementlearning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

  [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

  <u>Description</u>:

  ```
  Deep Q-Network (DQN) is a method of TD Learning, Off-Policy, e-Greedy Exploration (GLIE).

  Central formula:
  Q(S, A) <- Q(S, A) + alpha * (R + lambda * Q(newS, newA) - Q(S, A)),
  delta_w = R + lambda * Q(newS, newA).

  See David Silver RL Tutorial Lecture 5 - Q-Learning for more details.
  ```



* **Double DQN / Dueling DQN / Noisy DQN**

  <u>Code:</u> `./tutorial_DQN_variants.py`

  <u>Paper</u>: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

  <u>Description</u>:

  ```
  We implement Double DQN, Dueling DQN and Noisy DQN here.

  -The max operator in standard DQN uses the same values both to select and to evaluate an action by:

     Q(s_t, a_t) = R\_{t+1\} + gamma \* max\_{a}Q\_\{target\}(s_{t+1}, a).

  -Double DQN proposes to use following evaluation to address overestimation problem of max operator:

     Q(s_t, a_t) = R\_{t+1\} + gamma \* Q\_{target}(s\_\{t+1\}, max{a}Q(s_{t+1}, a)).

  -Dueling DQN uses dueling architecture where the value of state and the advantage of each action is estimated separately.

  -Noisy DQN propose to explore by adding parameter noises.


  ```




* **Prioritized Experience Replay**

  <u>Code</u>: `./tutorial_prioritized_replay.py`

  <u>Paper</u>: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

  <u>Description:</u>

  ```
  Prioritized experience replay is an efficient replay method that replay important transitions more frequently. Segment tree data structure is used to speed up indexing.
  ```



* **Distributed DQN (C51)**

  <u>Code</u>: `./tutorial_C51.py`

  <u>Paper</u>: [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf)

  <u>Description</u>:

  ```
  Categorical 51 distributional RL algorithm is a distrbuted DQN, where 51 means the number of atoms. In this algorithm, instead of estimating actual expected value, value distribution over a series of  continuous sub-intervals (atoms) is considered.
  ```


* **Actor-Critic (AC)**

  <u>Code</u>:`./tutorial_AC.py`

  <u>Paper</u>: [Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)

  <u>Description</u>:

  ```
  The implementation of Advantage Actor-Critic, using TD-error as the advantage.
  ```



* **Asynchronous Advantage Actor-Critic (A3C)**

  <u>Code</u>: `./tutorial_A3C.py`

  <u>Paper</u>: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)

  <u>Description</u>:

  ```
  The implementation of Asynchronous Advantage Actor-Critic (A3C), using multi-threading for distributed policy learning on Actor-Critic structure.
  ```



* **Soft Actor-Critic (SAC)**

  <u>Code</u>: `./tutorial_SAC.py`

  <u>Paper</u>: [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)

  <u>Description:</u>

  ```
  Actor policy in SAC is stochastic, with off-policy training.  And 'soft' in SAC indicates the trade-off between the entropy and expected return.  The additional consideration of entropy term helps with more explorative policy. And this implementation contains an automatic update for the entropy factor.

  This version of Soft Actor-Critic (SAC) implementation contains 5 networks:
  2 Q-networks, 2 target Q-networks and 1 policy network.
  ```




* **Vanilla Policy Gradient (PG or REINFORCE)**

  <u>Code</u>: `./tutorial_PG.py`

  <u>Paper</u>: [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

  <u>Description:</u>

  ```
  The policy gradient algorithm works by updating policy parameters via stochastic gradient ascent on policy performance. It's an on-policy algorithm can be used for environments with either discrete or continuous action spaces.

  To apply it on continuous action space, you need to change the last softmax layer and the choose_action function.
  ```



* **Deep Deterministic Policy Gradient (DDPG)**

  <u>Code:</u> `./tutorial_DDPG.py`

  <u>Paper:</u> [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)

  <u>Description:</u>

  ```
  An algorithm concurrently learns a Q-function and a policy.

  It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.
  ```




* **Twin Delayed DDPG (TD3)**

  <u>Code</u>: `./tutorial_TD3.py`

  <u>Paper</u>: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)

  <u>Description</u>:

  ```
  DDPG suffers from problems like overestimate of Q-values and sensitivity to hyper-parameters.

  Twin Delayed DDPG (TD3) is a variant of DDPG with several tricks:

  - Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”), and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.
  - Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently than the Q-function.
  - Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action.

  The implementation of TD3 includes 6 networks:
  2 Q-networks, 2 target Q-networks, 1 policy network, 1 target policy network.

  Actor policy in TD3 is deterministic, with Gaussian exploration noise.
  ```



* **Trust Region Policy Optimization (TRPO)**

  <u>Code</u>: `./tutorial_TRPO.py`

  <u>Paper</u>: [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)

  <u>Description:</u>

  ```
  PG method with a large step can crash the policy performance, even with a small step can lead a large differences in policy.

  TRPO constraints the step in policy space using KL divergence (rather than in parameter space), which can monotonically improve performance and avoid a collapsed update.
  ```



* **Proximal Policy Optimization (PPO)**

  <u>Code:</u> `./tutorial_PPO.py`

  <u>Paper</u>: [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)

  <u>Description:</u>

  ```
  A simple version of Proximal Policy Optimization (PPO) using single thread.

  PPO is a family of first-order methods that use a few other tricks to keep new policies close to old.

  PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.


  ```



* **Distributed Proximal Policy Optimization (DPPO)**

  <u>Code</u>: `./tutorial_DPPO.py`

  <u>Paper</u>: [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf)

  <u>Description:</u>

  ```
  A distributed version of OpenAI's Proximal Policy Optimization (PPO).

  Distribute the workers to collect data in parallel, then stop worker's roll-out and train PPO on collected data.
  ```



* **More in recent weeks**

## Environment:

We typically apply game environments in [Openai Gym](https://gym.openai.com/) for our tutorials. For other environment sources like [DeepMind Control Suite](https://github.com/deepmind/dm_control) and [Marathon-Envs in Unity](https://github.com/Unity-Technologies/marathon-envs), they all have wrappers to convert into format of Gym environments, see [here](https://github.com/martinseilair/dm_control2gym) and [here](https://github.com/Unity-Technologies/marathon-envs/tree/master/gym-unity).

Our env wrapper: `./tutorial_wrappers.py`

## Authors
- @zsdonghao Hao Dong: AC, A3C, Q-Learning, DQN, PG
- @quantumiracle Zihan Ding: SAC, TD3.
- @Tokarev-TT-33 Tianyang Yu @initial-h Hongming Zhang : PG, DDPG, PPO, DPPO, TRPO
- @Officium Yanhua Huang: C51, DQN_variants, prioritized_replay, wrappers.

