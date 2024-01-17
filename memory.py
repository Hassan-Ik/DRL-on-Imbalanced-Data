import numpy as np
from collections import namedtuple
import random


class Memory:
    def __init__(self, max_size=50000, train_start_size=5000):
        """
        Custome created memory class to store random samples, actions, and correct labels etc.

        Args:
            max_size (int, optional): _description_. What should be the maximum size of our memory list.
            train_start_size (int, optional): _description_. What should the start size of the memory for starting training.
        """
        self.transition = namedtuple("Transition", ['s', 'a', 'r', 's_', 't'])
        self.buffer = []
        self.max_size = max_size
        self.train_start_size = train_start_size

    def check_train(self):
        return len(self.buffer) >= self.train_start_size

    def stack(self, sample):
        if len(self.buffer) >= self.max_size:
            del self.buffer[0]
        self.buffer.append(self.transition(*sample))

    def sample_batch(self, batch_size):
        # sampling with replacement
        sample = random.sample(self.buffer, batch_size)
        s = np.array([e.s for e in sample])  # (batch, size, size, channels, )
        a = np.array([e.a for e in sample])  # (batch, 1)
        r = np.array([e.r for e in sample])  # (batch, 1)
        s_ = np.array([e.s_ for e in sample])  # (batch, size, size, channels)
        t = np.array([e.t for e in sample])  # (batch, 1)

        return s, a, r, s_, t
