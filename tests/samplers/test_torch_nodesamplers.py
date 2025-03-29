from itertools import groupby
from unittest import TestCase

import torch
from scipy import stats

from bayesian_network.bayesian_network import Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.samplers.torch_sampler import NodeSampler


class TestTorchNodeSampler(TestCase):
    def get_torch_settings(self) -> TorchSettings:
        torch_settings = TorchSettings()

        device = torch_settings.device

        print(f"Running tests with configuration: {torch_settings}")

        if device == torch.device("cuda") and not torch.cuda.is_available():
            self.fail("Running tests for cuda, but cuda not available.")

        if device == torch.device("mps") and not torch.backends.mps.is_available():
            self.fail("Running tests for mps, but mps not available.")

        return torch_settings

    def setUp(self):
        self.num_samples = 10000
        self.alpha = 0.001

    def test_node_sampler(self):
        device = self.get_torch_settings().device
        dtype = self.get_torch_settings().dtype

        # Assign
        p_true = torch.tensor(
            [2 / 14, 4 / 14, 5 / 14, 2 / 14, 1 / 14],
            device=device,
            dtype=dtype,
        )
        node = Node(p_true)

        # Act
        sut = NodeSampler(node)

        samples = torch.empty(self.num_samples, device=device, dtype=torch.int32)
        for i in range(self.num_samples):
            samples[i] = sut.sample([])

        # Assert
        p_true_cpu = p_true.cpu().double()
        p_true_cpu /= p_true_cpu.sum()
        expected = p_true_cpu * self.num_samples

        samples, _ = samples.sort()
        actual = [len(list(group)) for _, group in groupby(samples)]

        _, p = stats.chisquare(actual, expected)

        self.assertGreater(p, self.alpha)
