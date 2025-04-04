from typing import List

import torch

from bayesian_network.common.torch_settings import TorchSettings


class Evidence:
    def __repr__(self):
        return f"""
        num_observed_nodes: {self.num_observed_nodes},
        num_observations: {self.num_observations}
        """

    def __getitem__(self, index: int):
        return torch.stack([x[index] for x in self._data])

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
            assert (
                d.shape[0] == self.num_observations
            ), "Number of observations not equal for each observed node"

    @property
    def data(self) -> List[torch.Tensor]:
        return self._data
