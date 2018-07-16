import gym
from DQN2 import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped
print(env.observation_space.shape[0])
RL = DeepQNetwork(network_actions=env.action_space.n,
                  network_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.01,)

total_steps = 0

for i_episode in range(100):

    observation = env.reset()
    counter = 0
    while True:
        env.render()
        action = RL.choose_action(observation)
        counter += 1

        observation_, reward, done, info = env.step(action)
        position, velocity = observation_
        reward = abs(position + 0.46)

        RL.store_transition(observation, action, reward, observation_)
        if total_steps > 1000:
            RL.learn()
        if done:
            print('episode ', i_episode, ' done, total steps = ', counter)
            break
        observation = observation_
        total_steps += 1
