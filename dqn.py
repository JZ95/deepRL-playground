import gym
import tensorflow as tf
import numpy as np
import argparse
from collections import deque


GAME = "LunarLander-v2"
EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 0.00001

BUFF_SIZE = 10000
BATCH_SIZE = 16
LR = 1e-4
GAMMA = 0.90

MAX_STEP = 10000000
LOG_STEP = 2000
SAVE_STEP = 10000
GRADIENT_STEP = 1


class DeepQNetworkAgent(object):
    def __init__(self, lr, discount_rate, observation_shape: list, action_space_size: int, q_net, ckpt_name="ckpt/dqn"):
        self._buffer = deque()
        self._sess = tf.Session()

        with tf.name_scope("dqn"):
            self._state = tf.placeholder(shape=[None, *observation_shape], dtype=tf.float32)
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


def eps_annealing(step):
    x = 1. / (EPS_DECAY * step)
    return max(EPS_END, min(EPS_START, x))


def conv_net(conv_in, dim_out, name='conv_net', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=reuse):
        for _ in range(3):
            conv_out = tf.layers.conv2d(conv_in, filters=16, kernel_size=[3, 3], activation=tf.nn.relu)
            pool_out = tf.layers.max_pooling2d(conv_out, pool_size=3, strides=2)
            conv_in = pool_out

        flatten = tf.layers.flatten(pool_out)
        x = tf.layers.dense(flatten, 512, activation=tf.nn.relu)
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        return tf.layers.dense(x, dim_out)


def ffn(x, dim_out, name='ffn', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=reuse):
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 32, activation=tf.nn.relu)
        return tf.layers.dense(x, dim_out)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", type=str, choices=["train", "play"], default="train")
    parser.add_argument("--ckpt", type=str)

    args = parser.parse_args()
    if args.mode == "play" and args.ckpt is None:
        print("must offer a ckpt for playing.")
        exit()

    return args


def main():
    args = get_args()
    env = gym.make(GAME)
    observation = env.reset()

    agent = DeepQNetworkAgent(LR, GAMMA, env.observation_space.shape, env.action_space.n, ffn)
    info = {'episode': 0, 'rewards': []}
    rewards = []

    for step in range(1, MAX_STEP):
        env.render()
        eps = eps_annealing(step) if args.mode == "train" else EPS_END
        if np.random.rand() < eps:
            action = np.random.randint(0, env.action_space.n)
        else:
            q_vals = agent.eval(observation)
            action = np.argmax(q_vals)

        new_observation, reward, done, _ = env.step(action)
        agent.buffer((observation, action, reward, new_observation, done))
        observation = new_observation
        rewards.append(reward)

        if args.mode == "train":
            if step > 2 * BATCH_SIZE and step % GRADIENT_STEP == 0:
                agent.learn()

            if step % SAVE_STEP == 0:
                agent.save(step)

        if step % LOG_STEP == 0:
            msg = '[STEP: {step}, EPISODE: {episode}] ' \
                  'eps: {eps:.3f}, ' \
                  'avg_reward: {avg_reward:.3f}, ' \
                  'max_reward: {max_reward:.3f}'.format(step=step,
                                                        episode=info['episode'],
                                                        eps=eps,
                                                        avg_reward=np.mean(info['rewards'][-100:]),
                                                        max_reward=np.max(info['rewards'][-100:]))
            print(msg)

        if done:
            observation = env.reset()
            info['episode'] += 1
            info['rewards'].append(np.sum(rewards))
            rewards.clear()

    env.close()
    exit()


if __name__ == '__main__':
    main()
