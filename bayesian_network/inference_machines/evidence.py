from typing import Callable, List

import torch
from torch.utils.data import DataLoader

from bayesian_network.common.torch_settings import TorchSettings


class Evidence:
    def __repr__(self) -> str:
        return f"""
num_observed_nodes: {self.num_observed_nodes},
num_observations: {self.num_observations}
"""

    def __len__(self) -> int:
        return self.num_observations

    def __getitem__(self, index) -> "Evidence":
        return Evidence(
            [x[index] for x in self._data],
            self.torch_settings,
        )

    def get_observation(self, index: int) -> torch.Tensor:
        return torch.stack(self[index].data)

    def __init__(
        self,
        data: List[torch.Tensor],
        torch_settings: TorchSettings,
    ):
        self.torch_settings = torch_settings
        self._data = [x.to(torch_settings.device, torch_settings.dtype) for x in data]
        self.num_observed_nodes = len(data)

        # Assert and set number of observations
        self.num_observations = data[0].shape[0]
        for d in data[1:]:
            assert d.shape[0] == self.num_observations, (
                "Number of observations not equal for each observed node"
            )

    @property
    def data(self) -> List[torch.Tensor]:
        return self._data


class EvidenceLoader:
    def __init__(
        self,
        data_loader: DataLoader,
        transform: Callable[[torch.Tensor], Evidence],
    ):
        self._data_loader = data_loader
        self._transform = transform

    def __len__(self):
        return len(self._data_loader)

    def __iter__(self):
        for batch, _ in iter(self._data_loader):
            evidence = self._transform(batch)

            yield evidence

    @property
    def num_observations(self) -> int:
        return len(self._data_loader.dataset)
