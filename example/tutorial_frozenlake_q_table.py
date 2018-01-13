import time

import gym
import numpy as np

"""Q-Table learning algorithm, non deep learning - TD Learning, Off-Policy, e-Greedy Exploration

Q(S, A) <- Q(S, A) + alpha * (R + lambda * Q(newS, newA) - Q(S, A))

See David Silver RL Tutorial Lecture 5 - Q-Learning for more details.

For Q-Network, see tutorial_frozenlake_q_network.py

EN: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.5m3361vlw
CN: https://zhuanlan.zhihu.com/p/25710327
"""

## Load the environment
env = gym.make('FrozenLake-v0')
render = False           # display the game environment
running_reward = None

##================= Implement Q-Table learning algorithm =====================##
## Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
## Set learning parameters
lr = .85        # alpha, if use value function approximation, we can ignore it
lambd = .99     # decay factor
num_episodes = 10000
rList = []  # rewards for each episode
for i in range(num_episodes):
    ## Reset environment and get first new observation
    episode_time = time.time()
    s = env.reset()
    rAll = 0
    ## The Q-Table learning algorithm
    for j in range(99):
        if render: env.render()
        ## Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        ## Get new state and reward from environment
        s1, r, d, _ = env.step(a)
        ## Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + lambd * np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    rList.append(rAll)
    running_reward = r if running_reward is None else running_reward * 0.99 + r * 0.01
    print("Episode [%d/%d] sum reward:%f running reward:%f took:%.5fs %s" %
        (i, num_episodes, rAll, running_reward, time.time()-episode_time, '' if rAll == 0 else ' !!!!!!!!'))

print("Final Q-Table Values:/n %s" % Q)
