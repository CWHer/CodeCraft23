from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_sizes) -> None:
        super().__init__()
        self._net_list = []
        for hidden_size in hidden_sizes:
            self._net_list.append(nn.LazyLinear(hidden_size))
            self._net_list.append(nn.ReLU(inplace=True))
        self._net = nn.Sequential(*self._net_list)

    def forward(self, obs) -> torch.Tensor:
        return self._net(obs)


class ActorNet(nn.Module):
    def __init__(self,
                 obs_encoder: nn.Module,
                 action_encoder: nn.Module,
                 ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_encoder = action_encoder

    def forward(self,
                obs: List[torch.Tensor],
                act: List[torch.Tensor]
                ) -> torch.Tensor:
        batch_logits = []
        for k in range(len(obs)):
            # obs_embedding_shape: (E)
            # act_embedding_shape: (K', E)
            obs_embedding = self.obs_encoder(obs[k])
            act_embedding = self.action_encoder(act[k])
            # logits_shape: (K', 1)
            logits = torch.sum(obs_embedding * act_embedding,
                               dim=-1, keepdim=True)
            batch_logits.append(logits)
        # batch_logits_shape: (B, max_K, 1)
        max_dim = max([logits.shape[0] for logits in batch_logits])
        batch_logits = [
            F.pad(logits, (0, 0, 0, max_dim - logits.shape[0]), value=-1e10)
            for logits in batch_logits
        ]
        batch_logits = torch.stack(batch_logits, dim=0)
        return batch_logits


class CrticNet(nn.Module):
    def __init__(self,
                 obs_encoder: nn.Module,
                 action_encoder: nn.Module,
                 critic_mlp: nn.Module,
                 ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_encoder = action_encoder
        self.critic_mlp = critic_mlp

    def forward(self,
                obs: List[torch.Tensor],
                act: List[torch.Tensor]
                ) -> torch.Tensor:
        batch_value = []
        for k in range(len(obs)):
            # obs_embedding_shape: (E)
            # act_embedding_shape: (K', E)
            obs_embedding = self.obs_encoder(obs[k])
            act_embedding = self.action_encoder(act[k]).mean(dim=0)
            value = self.critic_mlp(
                torch.cat([obs_embedding, act_embedding], dim=-1))
            batch_value.append(value)

        return torch.stack(batch_value, dim=0)


if __name__ == "__main__":
    # test
    obs_encoder = MLP([128, 64])
    act_encoder = MLP([64, 64])
    actor_net = ActorNet(obs_encoder, act_encoder)
    critic_net = CrticNet(obs_encoder, act_encoder, MLP([128, 64, 1]))

    obs = [torch.randn(300), torch.randn(300)]
    act = [torch.randn(10, 300), torch.randn(20, 300)]
    print(actor_net(obs, act))
    print(critic_net(obs, act))
