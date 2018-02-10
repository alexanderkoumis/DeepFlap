
# Lets keep the functions to store the agent experience
# and train form over here.

import random
import numpy as np

class Experience:

    def __init__(self, state_curr, action, reward, state_next, dead):
        self.state_curr = state_curr
        self.action = action
        self.state_next = state_next
        self.reward = reward
        self.terminal = dead

class Memories:

    def __init__(self, size):
        self.buffer = []
        self.size = size

    def clean(self):
        self.buffer = []

    def add(self, state_curr, action, reward, state_next, dead):
        exp = Experience(state_curr, action, reward, state_next, dead)
        self.buffer.append(exp)
        if len(self.buffer) > self.size:
            del(self.buffer[0])

    def get_batch(self, size):
        batch = None
        if size > len(self.buffer):
            batch = self.buffer
        else:  
            batch = random.sample(self.buffer, size)
        
        e = Experience([],[],[],[],[])
        
        for experience in batch:
            e.state_curr.append(experience.state_curr)
            e.action.append(experience.action)
            e.reward.append(experience.reward)
            e.state_next.append(experience.state_next)
            e.terminal.append(experience.terminal)
        
        e.state_curr = np.array(e.state_curr)
        e.action     = np.array(e.action)
        e.reward     = np.array(e.reward)
        e.state_next = np.array(e.state_next)
        e.terminal   = np.array(e.terminal)
        
        return e
