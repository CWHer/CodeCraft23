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
        self._net_list.pop(-1)
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
            normed_obs_embedding = obs_embedding / torch.norm(obs_embedding)
            act_embedding = self.action_encoder(act[k])
            normed_act_embedding = act_embedding / \
                torch.norm(act_embedding, dim=-1, keepdim=True)
            # logits_shape: (K')
            logits = torch.sum(normed_obs_embedding * normed_act_embedding, dim=-1)
            batch_logits.append(logits)
        # batch_logits_shape: (B, max_K)
        max_dim = max([logits.shape[0] for logits in batch_logits])
        batch_logits = [
            F.pad(logits, (0, max_dim - logits.shape[0]), "constant", -1e10)
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
            normed_obs_embedding = obs_embedding / torch.norm(obs_embedding)
            act_embedding = self.action_encoder(act[k])
            normed_act_embedding = act_embedding / \
                torch.norm(act_embedding, dim=-1, keepdim=True)
            if normed_act_embedding.ndim == 1:
                normed_act_embedding = normed_act_embedding.unsqueeze(0)
            value = torch.cat([
                self.critic_mlp(
                    torch.cat([normed_obs_embedding, normed_act_embedding[i]], dim=-1))
                for i in range(normed_act_embedding.shape[0])
            ], dim=0)
            batch_value.append(value)

        max_dim = max([value.shape[0] for value in batch_value])
        batch_value = [
            F.pad(value, (0, max_dim - value.shape[0]), "constant", -1e10)
            for value in batch_value
        ]
        batch_value = torch.stack(batch_value, dim=0)
        return batch_value


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
