import gym
import numpy as np

env = gym.make('FrozenLake-v0')
LR = .05
y = .5
MAX_EPISODES = 10000
MAX_TEST = 1000
Q = np.zeros([env.observation_space.n, env.action_space.n])
test_reward = 0
training_reard = 0

for i in range(MAX_EPISODES):
    s = env.reset()
    reward = 0
    while True:
        a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
        s_, r, done, _ = env.step(a)
        next_a = np.argmax(Q[s_,:])  # choose next action
        training_reard += r
        if done:
            if r == 0:
                r = -1
        Q[s, a] = Q[s, a] + LR * (r + y * (Q[s_, next_a]) - Q[s, a]) # update with next action
        s = s_
        if done:
            if i % 1000 == 0:
                print("Reward =", training_reard / 1000, 'in last 1000 episodes   Episode: ', i)
                training_reard = 0
            break

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