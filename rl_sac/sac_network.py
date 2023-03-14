import torch
from torch import nn


class PreprocessNet(nn.Module):
    def __init__(self, hidden_sizes=[256, 128, 64, 1]) -> None:
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
                 preprocess_net: nn.Module,
                 action_shape: int,
                 ) -> None:
        super().__init__()
        self._preprocess = preprocess_net
        self._last = nn.LazyLinear(action_shape)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        assert isinstance(obs, torch.Tensor)
        embeddings = self._preprocess(obs)
        logits = self._last(embeddings)
        return logits


class CriticNet(nn.Module):
    def __init__(self,
                 preprocess_net: nn.Module,
                 last_size: int = 1) -> None:
        super().__init__()
        self._preprocess = preprocess_net
        self._last = nn.LazyLinear(last_size)

    def forward(self, obs: torch.Tensor):
        assert isinstance(obs, torch.Tensor)
        embeddings = self._preprocess(obs)
        return self._last(embeddings)
