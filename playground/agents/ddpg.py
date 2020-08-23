import tensorflow as tf

class DeepQNetworkAgent(object):
    def __init__(self, lr, discount_rate, observation_shape: list, action_space_size: int, q_net, ckpt_name="ckpt/dqn"):
        self._buffer = deque()
        self._sess = tf.Session()

        with tf.name_scope("ddpg"):
            with tf.variable_scope("actor"):
                self._state = tf.placeholder(shape=[None, *observation_shape], dtype=tf.float32)





            with tf.variable_scope("critic"):
                pass


            self._qs = q_net(self._state, action_space_size)

            with tf.variable_scope("loss"):
                self._state_new = tf.placeholder(shape=[None, *observation_shape], dtype=tf.float32)
                self._reward = tf.placeholder(shape=[None, ], dtype=tf.float32)
                self._action = tf.placeholder(shape=[None, ], dtype=tf.int32)
                self._is_terminal = tf.placeholder(shape=[None, ], dtype=tf.bool)
                target = self._reward + \
                                discount_rate * (1. - tf.cast(self._is_terminal, tf.float32)) * \
                                tf.reduce_max(q_net(self._state_new, action_space_size), axis=-1)

                self._loss = tf.nn.l2_loss(target - tf.reduce_sum(self._qs * tf.one_hot(self._action, action_space_size)))
                self._train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self._loss)

        self._saver = tf.train.Saver()
        self._sess.run(tf.global_variables_initializer())
        self._ckpt_name = ckpt_name
        if tf.train.checkpoint_exists(ckpt_name):
            self._saver.restore(self._sess, ckpt_name)

    def learn(self):
        np.random.shuffle(self._buffer)

        states, actions, rewards, new_states, dones = [], [], [], [], []
        for sample in self._buffer:
            s, a, r, new_s, d = sample
            states.append(s)
            actions.append(a)
            rewards.append(r)
            new_states.append(new_s)
            dones.append(d)

            if len(states) >= BATCH_SIZE:
                break

        feeds = {self._state: states, self._reward: rewards, self._action: actions, self._state_new: new_states, self._is_terminal: dones}
        self._sess.run(self._train_op, feed_dict=feeds)

    def eval(self, observation: np.ndarray):
        observation = np.expand_dims(observation, 0)
        return self._sess.run(self._qs, feed_dict={self._state: observation})

    def buffer(self, data: tuple):
        self._buffer.append(data)
        while len(self._buffer) > BUFF_SIZE:
            self._buffer.popleft()

    def save(self, step):
        self._saver.save(self._sess, self._ckpt_name, global_step=step)
