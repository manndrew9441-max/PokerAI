import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=250000): # Capacity is the number of moves AI can remember, 10000 is a good start for heads up
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Saves a single 'experience' triplet into memory
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Randomly Picks a group of memories for the Neural Network to study.
        This 'shuffling' prevens the AI from just memorizing the last hand played.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.uint8)
        )
    
    def __len__(self):
        return len(self.buffer)
    