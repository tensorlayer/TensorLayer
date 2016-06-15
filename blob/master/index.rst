Welcome to DeeR's documentation!
==================================

DeeR (Deep Reinforcement) is a python library to train an agent how to behave in a given environement so as to maximize a cumulative sum of rewards.
It is based on the original deep Q learning algorithm described in :
Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533. (see :ref:`what-is-deer`)

Here are key advantages of the library:

* Contrary to the original code, this package provides a more general framework where observations are made up of any number of elements : scalars, vectors and frames (instead of one type of frame only in the above mentionned paper). The belief state on which the agent is based to build the Q function is made up of any length history of each element provided in the observation.
* You can easily add up a validation phase that allows to stop the training process before overfitting. This possibility is useful when the environment is dependent on scarce data (e.g. limited time series).
* You also have access to advanced techniques such as Double Q-learning and prioritized Experience Replay that are readily available in the library.

In addition, the framework is made in such a way that it is easy to

* build any environment
* modify any part of the learning process
* use your favorite python-based framework to code your own neural network architecture. The provided neural network architectures are based on Theano but you may easily use another one.

It is a work in progress and input is welcome. Please submit any contribution via pull request.

What is new
------------
Version 0.3 (in development)
****************************
- Choice between different exploration/exploitation policies and possibility to easily built your own.
- :ref:`naming_conv` has been updated. This may cause broken backward compatibility if you used old examples. In that case, make the changes to the new convention (if needed have a look at the API) and you'll easily be able to get it run smoothly.


Version 0.2
***********
- Standalone python package (you can simply do ``pip install deer``)
- Integration of new examples environments : :ref:`toy_env_pendulum`, :ref:`PLE` and :ref:`gym`
- Double Q-learning and prioritized Experience Replay
- Augmented documentation
- First automated tests

Future extensions:
******************

* Add planning (e.g. MCTS based when deterministic environment)
* Several agents interacting in the same environment
* ...


User Guide
------------

.. toctree::
  :maxdepth: 2

  user/installation
  user/tutorial
  user/environments
  user/development

API reference
-------------

If you are looking for information on a specific function, class or method, this API is for you.

.. toctree::
  :maxdepth: 2

  modules/agents
  modules/controllers
  modules/environments
  modules/q-networks



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/VinF/General_Deep_Q_RL
