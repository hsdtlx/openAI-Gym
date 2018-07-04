import gym
from DQN import DeepQNetwork

env = gym.make('CartPole-v0')
env = env.unwrapped

RL = DeepQNetwork(network_actions=env.action_space.n,
                  network_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0


for i_episode in range(100):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('done, total steps = ', i_episode)
            break

        observation = observation_
        total_steps += 1

