from collections import deque
from typing import Dict, List

import numpy as np
from utils import Transition


class ReplayBuffer:
    def __init__(self, max_size) -> None:
        self.buffer: deque[Transition] = deque(maxlen=max_size)

    def add(self, episode: List[Transition]) -> None:
        self.buffer.extend(episode)

    def sample(self, batch_size) -> Dict[str, List]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        obs = [self.buffer[i].obs for i in indices]
        action = [self.buffer[i].action for i in indices]
        candidate_actions = [self.buffer[i].candidate_actions for i in indices]
        reward = [self.buffer[i].reward for i in indices]
        done = [self.buffer[i].done for i in indices]
        next_obs = [self.buffer[i].next_obs for i in indices]

        return {
            "obs": obs,
            "action": action,
            "candidate_actions": candidate_actions,
            "reward": reward,
            "done": done,
            "next_obs": next_obs,
        }
