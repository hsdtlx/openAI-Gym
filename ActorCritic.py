import numpy as np
import tensorflow as tf


np.random.seed(2)
tf.set_random_seed(2)


class Actor(object):
    def __init__(
            self,
            sess,
            n_features,
            n_actions,
            learning_rate=0.001
    ):
        self.sess = sess

        self.state = tf.placeholder(tf.float32, [1, n_features], "state")
        self.action = tf.placeholder(tf.int32, None, "action")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope('Actor'):
            layer1 = tf.layers.dense(
                inputs=self.state,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='layer1'
            )

            layer2 = tf.layers.dense(
                inputs=layer1,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='layer2'
            )

            self.layer2_result = layer2

        with tf.variable_scope('exp'):
            log_prob = tf.log(self.layer2_result[0, self.action])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize((-self.exp_v))

    def learn(self, s, a, td_error):
        s = s[np.newaxis, :]
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict={
                                                                self.state: s,
                                                                self.action: a,
                                                                self.td_error: td_error
                                                                })
        return exp_v

    def choose_acton(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.layer2_result, {self.state: s})
        return np.random.choice(np.arange(probs.shape[1]))


class Critic(object):
    def __init__(
            self,
            sess,
            n_features,
            learning_rate=0.01,
            gamma=0.9
    ):
        self.sess = sess
        self.gamma = gamma

        self.state = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, "r")

        with tf.variable_scope('Critic'):
            layer1 = tf.layers.dense(
                inputs=self.state,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='layer1'
            )

            layer2 = tf.layers.dense(
                inputs=layer1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='layer2'
            )

            self.v = layer2

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.state: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], feed_dict={
                                                                        self.state: s,
                                                                        self.v_: v_,
                                                                        self.r: r})

        return td_error
