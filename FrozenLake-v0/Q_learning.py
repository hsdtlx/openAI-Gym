# results of 100000 episodes, 10000 tests:
# 74.72%
# 73.97%
# 74.18%

import gym
import numpy as np

env = gym.make('FrozenLake-v0')
LR = .01
y = .8

MAX_EPISODES = 100000
MAX_TEST = 10000
Q = np.zeros([env.observation_space.n, env.action_space.n])
test_reward = 0
training_reward = 0

epsilon = 0.9
decay_rate = 0.001
min_epsilon = 0

# training
for i in range(MAX_EPISODES):
    s = env.reset()
    reward = 0
    while True:
        random = np.random.uniform()
        if random > epsilon:
            a = np.argmax(Q[s, :])
        else:
            a = env.action_space.sample()
        s_, r, done, _ = env.step(a)
        training_reward += r
        if done and (r == 0):
            r = -1
        Q[s, a] = Q[s, a] + LR * (r + y * np.max(Q[s_, :]) - Q[s, a])
        s = s_
        if done:
            if i % 1000 == 0:
                print("Reward =", training_reward / 1000, 'in last 1000 episodes   Episode: ', i)
                training_reward = 0
            break

# test of training
for i in range(MAX_TEST):
    s = env.reset()
    reward = 0
    while True:
        a = np.argmax(Q[s, :])
        s_, r, done, _ = env.step(a)
        s = s_
        if done:
            test_reward += r
            break
print(Q)

print(test_reward/MAX_TEST*100, '%')