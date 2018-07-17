import gym
from ActorCritic import Actor
from ActorCritic import Critic
import tensorflow as tf

MAX_STEPS = 1000
RENDER = False
LR_A = 0.001
LR_C = 0.01

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped


sess = tf.Session()

actor = Actor(
    sess,
    n_features=env.observation_space.shape[0],
    n_actions=env.action_space.n,
    learning_rate=LR_A
)

critic = Critic(
    sess,
    n_features=env.observation_space.shape[0],
    learning_rate=LR_C
)

sess.run(tf.global_variables_initializer())

tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(1000):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        # if RENDER: env.render()
        env.render()

        a = actor.choose_acton(s)

        s_, r, done, info = env.step(a)

        if done:
            r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)

        s = s_
        t += 1

        if done or t >= MAX_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > 200:
                RENDER = True

            print("episode: ", i_episode, " reward: ", int(running_reward))
            break
