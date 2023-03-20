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
        # fmt: off
        all_indices = [
            i for i in range(len(self.buffer))
            if not self.buffer[i].done
        ]
        indices = np.random.choice(all_indices, batch_size, replace=True)
        obs = [self.buffer[i].obs for i in indices]
        action = [self.buffer[i].action for i in indices]
        candidate_actions = [self.buffer[i].candidate_actions for i in indices]
        reward = [self.buffer[i].reward for i in indices]
        done = [self.buffer[i].done for i in indices]
        obs_next = [self.buffer[i + 1].obs for i in indices]
        candidate_actions_next = [self.buffer[i + 1].candidate_actions for i in indices]
        # fmt: on

        return {
            "obs": obs,
            "action": action,
            "candidate_actions": candidate_actions,
            "reward": reward,
            "done": done,
            "obs_next": obs_next,
            "candidate_actions_next": candidate_actions_next,
        }
