from collections import deque
import numpy as np


class Buffer(object):
    def __init__(self, buff_size):
        self._buff_size = buff_size
        self._buf = deque(maxlen=self._buff_size)

        self._mem_size = 2 * buff_size
        self._random_mem = np.random.randint(0, buff_size, size=self._mem_size)
        self._ptr = 0

    def _reset(self):
        self._ptr = 0
        self._random_mem = np.random.randint(0, self._buff_size, size=2 * self._buff_size)

    def sample(self, batch_size):
        states, actions, rewards, new_states, dones = [], [], [], [], []
        while len(states) < batch_size:
            i = np.random.randint(0, len(self._buf))
            s, a, r, ns, d = self._buf[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            new_states.append(ns)
            dones.append(d)

        return states, actions, rewards, new_states, dones


    def buffer(self, state, action, reward, new_state, done):
        self._buf.append((state, action, reward, new_state, done))