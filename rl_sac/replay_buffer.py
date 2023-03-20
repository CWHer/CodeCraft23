from collections import deque
from typing import Dict, List

import numpy as np
from utils import Transition


class ReplayBuffer:
    def __init__(self, max_size) -> None:
        self.buffer: deque[Transition] = deque(maxlen=max_size)

    def add(self, episode: List[Transition]) -> None:
        self.buffer.extend(episode)

    def sample(self, batch_size) -> Dict[str, np.ndarray]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs = [self.buffer[i].obs for i in indices]
        action = [self.buffer[i].action for i in indices]
        reward = [self.buffer[i].reward for i in indices]
        done = [self.buffer[i].done for i in indices]
        next_obs = [self.buffer[i].next_obs for i in indices]

        batch = {
            "obs": np.array(obs),
            "act": np.array(action),
            "rew": np.array(reward),
            "done": np.array(done),
            "next_obs": np.array(next_obs),
        }
        return batch
