import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            print_graph=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        self.ep_observations, self.ep_actions, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if print_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.observations = tf.placeholder(tf.float32, [None, self.n_features], name='observations')
            self.actions = tf.placeholder(tf.int32, [None, ], name='actions')
            self.actions_value = tf.placeholder(tf.float32, [None, ], name='actions_value')

        layer1 = tf.layers.dense(
            inputs=self.observations,
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc_layer1'
        )

        layer2 = tf.layers.dense(
            inputs=layer1,
            units=self.n_actions,
            activation=tf.nn.softmax,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc_layer2'
        )

        self.layer2_result = layer2

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.layer2_result,
                                                                          labels=self.actions)
            loss = tf.reduce_mean(neg_log_prob*self.actions_value)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def choose_action(self, observation):
        weights = self.sess.run(self.layer2_result, feed_dict={self.observations: observation[np.newaxis, :]})
        action = np.random.choice(range(weights.shape[1]), p=weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_observations.append(s)
        self.ep_actions.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        self.sess.run(self.train_op, feed_dict={
            self.observations: np.vstack(self.ep_observations),  # shape=[None, n_obs]
            self.actions: np.array(self.ep_actions),  # shape=[None, ]
            self.actions_value: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_observations, self.ep_actions, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.reward_decay + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
