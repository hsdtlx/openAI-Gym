import tensorflow as tf
import numpy as np


class DeepQNetwork:
    def __init__(
            self,
            network_actions,
            network_features,
            learning_rate = 0.01,
            gamma=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=300,
            batch_size=32,
            e_greedy_increment=None,
    ):
            self.network_actions = network_actions
            self.network_features = network_features
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.e_greedy = e_greedy
            self.replace_target_iter = replace_target_iter
            self.memory_size = memory_size
            self.batch_size = batch_size
            self.e_greedy_increment = e_greedy_increment
            self.epsilon = 0 if e_greedy_increment is not None else self.e_greedy

            self.learning_counter = 0

            self.memory = np.zeros((self.memory_size, self.network_features * 2 + 2))

            self._build_net()
            t_params = tf.get_collection('target_net_params')
            e_params = tf.get_collection('eval_net_params')
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            self.sess = tf.Session()

            self.sess.run(tf.global_variables_initializer())
            self.cost_his = []

    def _build_net(self):
        # building evaluate net

        with tf.name_scope('input'):
            self.obseravtions = tf.placeholder(tf.float32, [None, self.network_features], 'observations')
            self.q_target = tf.placeholder(tf.float32, [None, self.network_actions], 'Q_target')

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                        ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 100, \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            layer1 = tf.layers.dense(
                inputs=self.network_features,
                units=n_l1,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., 0.3),
                bias_initializer=tf.constant_initializer(0.1),
                collections=c_names,
                name='layer1'
            )

            self.q_eval = tf.layers.dense(
                inputs=layer1,
                units=self.network_actions,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., 0.3),
                bias_initializer=tf.constant_initializer(0.1),
                collections=c_names,
                name='layer2'
            )

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # building target net
        self.s_ = tf.placeholder(tf.float32, [None, self.network_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            layer1 = tf.layers.dense(
                inputs=self.network_features,
                units=n_l1,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., 0.3),
                bias_initializer=tf.constant_initializer(0.3),
                collections=c_names,
                name='layer1'
            )

            self.q_next = tf.layers.dense(
                inputs=layer1,
                units=self.network_actions,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., 0.3),
                bias_initializer=tf.constant_initializer(0.3),
                collections=c_names,
                name='layer2'
            )

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.network_actions)
        return action

    def learn(self):
        if self.learning_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.network_features:],
                self.s: batch_memory[:, :self.network_features],
            }
        )

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.network_features].astype(int)
        reward = batch_memory[:, self.network_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, self.cost = self.sess.run([self._train_op, self.loss],feed_dict={
            self.s: batch_memory[:, :self.network_features],
            self.q_target: q_target
        })

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.e_greedy else self.e_greedy
        self.learning_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig("one.png")
        plt.show()
