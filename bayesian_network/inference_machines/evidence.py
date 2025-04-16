from abc import ABC, abstractmethod
from typing import List

import torch

from bayesian_network.common.torch_settings import TorchSettings


class Evidence:
    def __repr__(self) -> str:
        return f"""
        num_observed_nodes: {self.num_observed_nodes},
        num_observations: {self.num_observations}
        """

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


class IEvidenceBatches(ABC):
    @abstractmethod
    def next(self) -> Evidence:
        pass


class EvidenceBatches(IEvidenceBatches):
    def __init__(
        self,
        evidence: Evidence,
        batch_size: int,
    ):
        if batch_size > evidence.num_observations:
            raise ValueError("batch_size may not be larger than number of observations in evidence")

        self.evidence = evidence
        self.batch_size = batch_size

    def next(self) -> Evidence:
        num_observations = self.evidence.num_observations
        batch_size = self.batch_size

        indices = torch.randperm(num_observations)[0:batch_size]

        return self.evidence[indices]
