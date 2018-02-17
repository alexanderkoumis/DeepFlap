
# Lets keep the functions to store the agent experience
# and train form over here.

import random
import numpy as np


class Experience(object):

    def __init__(self, state_curr, action, reward, state_next, dead):
        self.state_curr = state_curr
        self.action = action
        self.state_next = state_next
        self.reward = reward
        self.terminal = dead


class Memories(object):

    def __init__(self, size):
        self.buffer = []
        self.size = size

        self.buffer_init = False
        self.buffer_items = 0
        self.buffer_idx = 0

        self.buffer_state_curr = None
        self.buffer_action = None
        self.buffer_reward = None
        self.buffer_state_next = None
        self.buffer_dead = None

    def init_buffers(self, state_shape):

        state_rows, state_cols, state_depth = state_shape
        state_size = (self.size, state_rows, state_cols, state_depth)
        int_size = self.size

        self.buffer_state_curr = np.zeros(state_size, dtype='uint8')
        self.buffer_action = np.zeros(int_size, dtype='uint8')
        self.buffer_reward = np.zeros(int_size, dtype='uint8')
        self.buffer_state_next = np.zeros(state_size, dtype='uint8')
        self.buffer_dead = np.zeros(int_size, dtype='uint8')

        self.buffer_init = True
        self.buffer_idx = 0

    def add(self, state_curr, action, reward, state_next, dead):

        if not self.buffer_init:
            self.init_buffers(state_curr.shape)

        self.buffer_state_curr[self.buffer_idx] = state_curr
        self.buffer_action[self.buffer_idx] = action
        self.buffer_reward[self.buffer_idx] = reward
        self.buffer_state_next[self.buffer_idx] = state_next
        self.buffer_dead[self.buffer_idx] = dead

        # Ring buffer
        self.buffer_items = min(self.buffer_items + 1, self.size)
        self.buffer_idx = (self.buffer_idx + 1) % self.size

    def get_batch(self, batch_size):

        if batch_size > self.buffer_items:
            exp = Experience(
                self.buffer_state_curr[0:self.buffer_items],
                self.buffer_action[0:self.buffer_items],
                self.buffer_reward[0:self.buffer_items],
                self.buffer_state_next[0:self.buffer_items],
                self.buffer_dead[0:self.buffer_items]
            )

        else:
            rand_idxs = np.random.choice(self.buffer_items, batch_size, replace=False)
            exp = Experience(
                self.buffer_state_curr[rand_idxs],
                self.buffer_action[rand_idxs],
                self.buffer_reward[rand_idxs],
                self.buffer_state_next[rand_idxs],
                self.buffer_dead[rand_idxs]
            )

        return exp
