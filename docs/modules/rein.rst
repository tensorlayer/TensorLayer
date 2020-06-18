API - Reinforcement Learning
==============================

We provide two reinforcement learning libraries:

- `RL-tutorial <https://github.com/tensorlayer/tensorlayer/tree/master/examples/reinforcement_learning>`__ for professional users with low-level APIs.
- `RLzoo <https://rlzoo.readthedocs.io/en/latest/>`__ for simple usage with high-level APIs.

.. automodule:: tensorlayer.rein

.. autosummary::

  discount_episode_rewards
  cross_entropy_reward_loss
  log_weight
  choice_action_by_probs


Reward functions
---------------------
.. autofunction:: discount_episode_rewards

Cost functions
---------------------

Weighted Cross Entropy
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: cross_entropy_reward_loss

Log weight
^^^^^^^^^^^^^^
.. autofunction:: log_weight

Sampling functions
---------------------
.. autofunction:: choice_action_by_probs
