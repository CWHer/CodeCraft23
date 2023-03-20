import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from Env.env_utils import obsToNumpy, taskToNumpy
from Env.task_utils import Task
from torch import nn
from torch.distributions import Categorical


class DiscreteSACPolicy(nn.Module):
    def __init__(self,
                 actor: torch.nn.Module,
                 actor_optim: torch.optim.Optimizer,
                 critic1: torch.nn.Module,
                 critic1_optim: torch.optim.Optimizer,
                 critic2: torch.nn.Module,
                 critic2_optim: torch.optim.Optimizer,
                 tau: float,
                 gamma: float,
                 alpha: float,
                 obs_padding: int,
                 task_padding: int,
                 ) -> None:
        super().__init__()
        self._tau = tau
        self._gamma = gamma

        self._actor = actor
        self._actor_optim = actor_optim
        self._critic1 = critic1
        self._critic1_old = copy.deepcopy(critic1)
        self._critic1_old.eval()
        self._critic1_optim = critic1_optim
        self._critic2 = critic2
        self._critic2_old = copy.deepcopy(critic2)
        self._critic2_old.eval()
        self._critic2_optim = critic2_optim

        self._alpha = alpha

        self.obs_padding = obs_padding
        self.task_padding = task_padding
    # APIs start

    def select(self,
               robot_id: int,
               tasks: List[Task],
               obs: Dict[str, Any]
               ) -> Task:
        obs_tensor = torch.as_tensor(
            obsToNumpy(obs, self.obs_padding))
        tasks_tensor = torch.tensor(
            [taskToNumpy(task, self.task_padding)
             for task in tasks]
        )
        logits = self._actor([obs_tensor], [tasks_tensor])[0]
        dist = Categorical(logits=logits)
        return tasks[dist.sample().item()]

    def train(self, mode: bool = True):
        self.training = mode
        self._actor.train(mode)
        self._critic1.train(mode)
        self._critic2.train(mode)
        return self

    def update(self, batch: Dict) -> Dict[str, Any]:
        return self.__learn(batch)

    # APIs end

    # fmt: off
    def __softUpdate(self, dst: nn.Module, src: nn.Module, tau: float):
        for dst_param, src_param in zip(dst.parameters(), src.parameters()):
            dst_param.data.copy_(tau * src_param.data + (1 - tau) * dst_param.data)
    # fmt: on

    def __targetQ(self, obs_next, candidate_actions_next) -> torch.Tensor:
        logits = self._actor(obs_next, candidate_actions_next)
        dist = Categorical(logits=logits)

        target_q = dist.probs * torch.min(
            self._critic1_old(obs_next, candidate_actions_next),
            self._critic2_old(obs_next, candidate_actions_next)
        )
        return target_q.sum(dim=-1) + self._alpha * dist.entropy()

    def __learn(self, batch: Dict) -> Dict[str, float]:
        def toFloat(x): return torch.as_tensor(x, dtype=torch.float32)

        # fmt: off
        obs = list(map(toFloat, batch["obs"]))
        action = list(map(toFloat, batch["action"]))
        candidate_actions = list(map(toFloat, batch["candidate_actions"]))
        reward = toFloat(batch["reward"])
        done = toFloat(batch["done"])
        obs_next = list(map(toFloat, batch["obs_next"]))
        candidate_actions_next = list(map(toFloat, batch["candidate_actions_next"]))
        # fmt: on

        with torch.no_grad():
            target_q = reward.view(-1) + \
                (1 - done.view(-1)) * self._gamma * \
                self.__targetQ(obs_next, candidate_actions_next)
            target_q = target_q.flatten()

        # critic 1
        self._critic1_old(obs, action)
        current_q1 = self._critic1(obs, action)
        td1 = current_q1 - target_q
        critic1_loss = td1.pow(2).mean()

        self._critic1_optim.zero_grad()
        critic1_loss.backward()
        self._critic1_optim.step()

        # critic 2
        self._critic2_old(obs, action)
        current_q2 = self._critic2(obs, action)
        td2 = current_q2 - target_q
        critic2_loss = td2.pow(2).mean()

        self._critic2_optim.zero_grad()
        critic2_loss.backward()
        self._critic2_optim.step()

        # actor
        logits = self._actor(obs, candidate_actions)
        dist = Categorical(logits=logits)
        entropy = dist.entropy()
        with torch.no_grad():
            current_q1a = self._critic1(obs, candidate_actions)
            current_q2a = self._critic2(obs, candidate_actions)
            q = torch.min(current_q1a, current_q2a)
        actor_loss = -(self._alpha * entropy +
                       (dist.probs * q).sum(dim=-1)).mean()
        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        with torch.no_grad():
            self.__softUpdate(self._critic1_old, self._critic1, self._tau)
            self.__softUpdate(self._critic2_old, self._critic2, self._tau)

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

        return result
